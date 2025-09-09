"""
Bittle Walking Environment for PyBullet Simulation
深層強化学習用のBittle四足歩行ロボットシミュレーション環境
"""

import gym
import numpy as np
import pybullet as p
import pybullet_data
import yaml
from typing import Dict, Tuple, Any, Optional
import os
from pathlib import Path

from .reward_functions import RewardFunction


class BittleWalkingEnv(gym.Env):
    """
    Bittle四足歩行ロボットのPyBulletシミュレーション環境
    
    目標: 幅15cmの通路を1m直進する歩行制御を学習
    """
    
    def __init__(self, config_path: str = "config/env_config.yaml", 
                 bittle_config_path: str = "config/bittle_config.yaml",
                 render: bool = False):
        """
        環境の初期化
        
        Args:
            config_path: 環境設定ファイルのパス
            bittle_config_path: Bittleロボット設定ファイルのパス
            render: 可視化の有無
        """
        super().__init__()
        
        # 設定ファイルの読み込み
        self.config = self._load_config(config_path)
        self.bittle_config = self._load_config(bittle_config_path)
        
        # PyBulletの初期化
        self.render = render
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 環境パラメータ
        self.timestep = self.config['simulation']['timestep']
        self.control_frequency = self.config['environment']['action']['control_frequency']
        self.max_episode_steps = self.config['training']['episode']['max_steps']
        
        # 観測・行動空間の定義
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        
        # 環境状態
        self.robot_id = None
        self.corridor_id = None
        self.ground_id = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.initial_position = None
        self.target_position = None
        
        # 報酬関数
        self.reward_function = RewardFunction(self.config['reward'])
        
        # 物理パラメータ
        self._setup_physics()
        
        # 環境のリセット
        self.reset()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _define_observation_space(self) -> gym.Space:
        """観測空間の定義"""
        obs_dim = self.config['environment']['observation']['total_dimensions']
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _define_action_space(self) -> gym.Space:
        """行動空間の定義"""
        action_dim = self.config['environment']['action']['joint_targets']
        action_range = self.config['environment']['action']['action_range']
        return gym.spaces.Box(
            low=action_range[0], 
            high=action_range[1], 
            shape=(action_dim,), 
            dtype=np.float32
        )
    
    def _setup_physics(self):
        """物理シミュレーションの設定"""
        p.setTimeStep(self.timestep)
        p.setGravity(*self.config['simulation']['gravity'])
        
        # 接触パラメータの設定
        p.setPhysicsEngineParameter(
            contactERP=self.config['simulation']['contact_erp'],
            contactCFM=self.config['simulation']['contact_cfm'],
            numSolverIterations=self.config['simulation']['num_solver_iterations']
        )
    
    def reset(self) -> np.ndarray:
        """環境のリセット"""
        # 既存のオブジェクトをクリア
        p.resetSimulation()
        self._setup_physics()
        
        # 地面の作成
        self._create_ground()
        
        # 通路の作成
        self._create_corridor()
        
        # Bittleロボットの読み込み
        self._load_robot()
        
        # 初期位置の設定
        self._set_initial_pose()
        
        # 目標位置の設定
        self._set_target_position()
        
        # エピソード状態のリセット
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self._get_observation()
    
    def _create_ground(self):
        """地面の作成"""
        self.ground_id = p.loadURDF("plane.urdf")
        
        # 地面の物理パラメータ設定
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=self.config['simulation']['lateral_friction'],
            spinningFriction=self.config['simulation']['spinning_friction'],
            rollingFriction=self.config['simulation']['rolling_friction'],
            restitution=self.config['simulation']['restitution']
        )
    
    def _create_corridor(self):
        """通路の作成"""
        corridor_config = self.config['environment']['corridor']
        width = corridor_config['width']
        length = corridor_config['length']
        wall_height = corridor_config['wall_height']
        
        # 通路の壁を作成（簡易的な実装）
        # 実際の実装では、より詳細な通路モデルを作成する
        self.corridor_bounds = {
            'left': -width/2,
            'right': width/2,
            'front': length,
            'back': 0
        }
    
    def _load_robot(self):
        """Bittleロボットの読み込み"""
        urdf_path = self.bittle_config['robot']['urdf_path']
        full_urdf_path = os.path.join(os.getcwd(), urdf_path)
        
        # URDFファイルの存在確認
        if not os.path.exists(full_urdf_path):
            raise FileNotFoundError(f"URDF file not found: {full_urdf_path}")
        
        # ロボットの読み込み
        self.robot_id = p.loadURDF(
            full_urdf_path,
            basePosition=[0, 0, 0.1],  # 少し浮かせる
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        
        # ロボットの物理パラメータ設定
        self._setup_robot_dynamics()
    
    def _setup_robot_dynamics(self):
        """ロボットの動力学パラメータ設定"""
        # 各リンクの物理パラメータを設定
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            # 関節制限の設定
            if joint_name in self.bittle_config['robot']['joints']['joint_limits']:
                limits = self.bittle_config['robot']['joints']['joint_limits'][joint_name]
                p.resetJointState(self.robot_id, i, 0)  # 初期角度を0に設定
    
    def _set_initial_pose(self):
        """初期姿勢の設定"""
        # ロボットを通路の開始位置に配置
        initial_pos = [0, 0, 0.1]  # 通路の開始位置
        initial_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        p.resetBasePositionAndOrientation(self.robot_id, initial_pos, initial_orn)
        self.initial_position = np.array(initial_pos)
    
    def _set_target_position(self):
        """目標位置の設定"""
        # 通路の終端を目標位置に設定
        corridor_length = self.config['environment']['corridor']['length']
        self.target_position = np.array([corridor_length, 0, 0])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """環境のステップ実行"""
        # 行動の適用
        self._apply_action(action)
        
        # 物理シミュレーションの実行
        p.stepSimulation()
        
        # 観測の取得
        observation = self._get_observation()
        
        # 報酬の計算
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # 終了条件のチェック
        done = self._is_done()
        
        # 情報の収集
        info = self._get_info()
        
        self.current_step += 1
        
        return observation, reward, done, info
    
    def _apply_action(self, action: np.ndarray):
        """行動の適用"""
        # 行動を関節角度に変換
        joint_targets = self._action_to_joint_angles(action)
        
        # 各関節に目標角度を設定
        joint_names = self.bittle_config['robot']['joints']['joint_names']
        
        for i, joint_name in enumerate(joint_names):
            if i < len(joint_targets):
                # 関節IDを取得
                joint_id = self._get_joint_id(joint_name)
                if joint_id is not None:
                    # 関節制御
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=joint_targets[i],
                        force=self.bittle_config['robot']['joints']['control']['max_torque']
                    )
    
    def _action_to_joint_angles(self, action: np.ndarray) -> np.ndarray:
        """行動を関節角度に変換"""
        joint_names = self.bittle_config['robot']['joints']['joint_names']
        joint_angles = np.zeros(len(joint_names))
        
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.bittle_config['robot']['joints']['joint_limits']:
                limits = self.bittle_config['robot']['joints']['joint_limits'][joint_name]
                # 正規化された行動を関節角度範囲にマッピング
                joint_angles[i] = np.interp(action[i], [-1, 1], limits)
        
        return joint_angles
    
    def _get_joint_id(self, joint_name: str) -> Optional[int]:
        """関節名から関節IDを取得"""
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[1].decode('utf-8') == joint_name:
                return i
        
        return None
    
    def _get_observation(self) -> np.ndarray:
        """観測の取得"""
        obs = []
        
        # 関節角度と角速度
        joint_angles, joint_velocities = self._get_joint_states()
        obs.extend(joint_angles)
        obs.extend(joint_velocities)
        
        # IMUデータ（簡易実装）
        imu_data = self._get_imu_data()
        obs.extend(imu_data)
        
        # 位置・姿勢
        position, orientation = self._get_robot_pose()
        obs.extend(position)
        obs.extend(orientation)
        
        # 目標位置
        obs.extend(self.target_position[:2])  # X, Y座標のみ
        
        return np.array(obs, dtype=np.float32)
    
    def _get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """関節状態の取得"""
        joint_names = self.bittle_config['robot']['joints']['joint_names']
        joint_angles = np.zeros(len(joint_names))
        joint_velocities = np.zeros(len(joint_names))
        
        for i, joint_name in enumerate(joint_names):
            joint_id = self._get_joint_id(joint_name)
            if joint_id is not None:
                joint_state = p.getJointState(self.robot_id, joint_id)
                joint_angles[i] = joint_state[0]  # 角度
                joint_velocities[i] = joint_state[1]  # 角速度
        
        return joint_angles, joint_velocities
    
    def _get_imu_data(self) -> np.ndarray:
        """IMUデータの取得（簡易実装）"""
        # 実際の実装では、より詳細なIMUシミュレーションを行う
        base_velocity, base_angular_velocity = p.getBaseVelocity(self.robot_id)
        
        # 加速度（簡易計算）
        acceleration = np.array([0, 0, -9.81])  # 重力加速度
        
        # 角速度
        angular_velocity = np.array(base_angular_velocity)
        
        return np.concatenate([acceleration, angular_velocity])
    
    def _get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """ロボットの位置・姿勢の取得"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        return np.array(pos), np.array(euler)
    
    def _calculate_reward(self) -> float:
        """報酬の計算"""
        return self.reward_function.calculate_reward(
            robot_id=self.robot_id,
            target_position=self.target_position,
            corridor_bounds=self.corridor_bounds,
            step=self.current_step
        )
    
    def _is_done(self) -> bool:
        """終了条件のチェック"""
        # 最大ステップ数に達した場合
        if self.current_step >= self.max_episode_steps:
            return True
        
        # 転倒のチェック
        if self._is_fallen():
            return True
        
        # 目標到達のチェック
        if self._is_target_reached():
            return True
        
        # 通路外に出た場合
        if self._is_out_of_corridor():
            return True
        
        return False
    
    def _is_fallen(self) -> bool:
        """転倒のチェック"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # ピッチまたはロールが一定角度を超えた場合
        max_angle = 0.5  # 約30度
        return abs(euler[0]) > max_angle or abs(euler[1]) > max_angle
    
    def _is_target_reached(self) -> bool:
        """目標到達のチェック"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.linalg.norm(np.array(pos[:2]) - self.target_position[:2])
        
        return distance < 0.1  # 10cm以内
    
    def _is_out_of_corridor(self) -> bool:
        """通路外のチェック"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        x, y = pos[0], pos[1]
        
        return (y < self.corridor_bounds['left'] or 
                y > self.corridor_bounds['right'] or
                x < self.corridor_bounds['back'] or
                x > self.corridor_bounds['front'])
    
    def _get_info(self) -> Dict[str, Any]:
        """情報の取得"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        distance = np.linalg.norm(np.array(pos[:2]) - self.target_position[:2])
        
        return {
            'position': pos,
            'orientation': orn,
            'distance_to_target': distance,
            'episode_reward': self.episode_reward,
            'step': self.current_step
        }
    
    def render(self, mode='human'):
        """環境の可視化"""
        if mode == 'human':
            # PyBulletのGUIモードでは自動的に可視化される
            pass
        elif mode == 'rgb_array':
            # RGB配列の取得（実装が必要）
            pass
    
    def close(self):
        """環境の終了"""
        p.disconnect(self.physics_client)
    
    def seed(self, seed=None):
        """乱数シードの設定"""
        np.random.seed(seed)
        return [seed]
