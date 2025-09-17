"""
Bittle Walking Environment for PyBullet Simulation
深層強化学習用のBittle四足歩行ロボットシミュレーション環境
"""

import gymnasium as gym
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
    
    目標: 幅30cmの通路を1m直進する歩行制御を学習（手のひらサイズロボット用、安全マージン拡大）
    """
    
    def __init__(self, config_path: str = "config/env_config.yaml", 
                 bittle_config_path: str = "config/bittle_config.yaml",
                 render: bool = False,
                 render_mode: Optional[str] = None):
        """
        環境の初期化
        
        Args:
            config_path: 環境設定ファイルのパス
            bittle_config_path: Bittleロボット設定ファイルのパス
            render: 可視化の有無
            render_mode: Gymnasiumのレンダリングモード ("human", "rgb_array", None)
        """
        super().__init__()
        
        # 設定ファイルの読み込み
        self.config = self._load_config(config_path)
        self.bittle_config = self._load_config(bittle_config_path)
        
        # レンダリングモードの設定
        self.render_mode = render_mode
        self._render_enabled = render or (render_mode == "human")
        
        # PyBulletの初期化
        if self._render_enabled:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # レンダリング設定の改善
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        
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
        self.reward_function = RewardFunction(self.config['environment']['reward'])
        
        # 学習段階の管理
        self.learning_stage = 1
        self.stage_adaptation_enabled = self.bittle_config['robot']['joints']['initial_pose'].get('stage_adaptation', {}).get('enabled', False)
        
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
            fixedTimeStep=self.timestep,
            numSolverIterations=self.config['simulation']['num_solver_iterations'],
            numSubSteps=self.config['simulation']['num_substeps']
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
        
        # 安全マージンを追加（ロボットサイズを考慮）
        safety_margin = 0.10  # 10cmの安全マージン（手のひらサイズロボット用、2倍に拡大）
        
        # 通路の壁を作成（簡易的な実装）
        # 実際の実装では、より詳細な通路モデルを作成する
        self.corridor_bounds = {
            'left': -width/2 + safety_margin,    # 左端に安全マージン
            'right': width/2 - safety_margin,    # 右端に安全マージン
            'front': length - safety_margin,     # 前端に安全マージン
            'back': safety_margin                # 後端に安全マージン（0から0.1mに変更）
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
        # ロボットを通路内の安全な位置に配置（通路幅1.2m、安全マージン0.1m）
        initial_pos = [0.2, 0, 0.15]  # X=0.2m（通路内20cm）、Y=0m（中央）、Z=0.15m（高さ）
        initial_orn = p.getQuaternionFromEuler([0, 0, 0])
        
        p.resetBasePositionAndOrientation(self.robot_id, initial_pos, initial_orn)
        self.initial_position = np.array(initial_pos)
        
        # 適切な初期姿勢を設定
        self._set_proper_initial_joint_angles()
        
        # デバッグ: ロボットの境界ボックスを取得
        aabb_min, aabb_max = p.getAABB(self.robot_id)
        print(f"Robot AABB: min={aabb_min}, max={aabb_max}")
        print(f"Robot height: {aabb_max[2] - aabb_min[2]:.3f}m")
        print(f"Robot bottom: {aabb_min[2]:.3f}m (should be > 0)")
        
        # デバッグ: 各足の位置を確認
        joint_names = self.bittle_config['robot']['joints']['joint_names']
        for i, joint_name in enumerate(joint_names):
            joint_id = self._get_joint_id(joint_name)
            if joint_id is not None:
                joint_info = p.getJointInfo(self.robot_id, joint_id)
                joint_pos = p.getLinkState(self.robot_id, joint_id)[0]
                print(f"Joint {joint_name}: position={joint_pos}")
        
        # デバッグ: 全リンクの位置を確認
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Total joints: {num_joints}")
        min_z = float('inf')
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 固定関節以外
                link_pos = p.getLinkState(self.robot_id, i)[0]
                print(f"Joint {i} ({joint_info[1].decode()}): position={link_pos}")
                min_z = min(min_z, link_pos[2])
        
        print(f"Lowest joint position: {min_z:.3f}m")
        print(f"Required initial height: {abs(min_z) + 0.05:.3f}m (lowest + 5cm margin)")
    
    def _set_proper_initial_joint_angles(self):
        """適切な初期関節角度の設定（多様化対応）"""
        joint_names = self.bittle_config['robot']['joints']['joint_names']
        pose_config = self.bittle_config['robot']['joints']['initial_pose']
        
        # 多様化モードの取得
        diversity_mode = pose_config.get('diversity_mode', 'fixed')
        
        for joint_name in joint_names:
            joint_id = self._get_joint_id(joint_name)
            if joint_id is not None:
                # 基本角度の設定
                if 'shoulder' in joint_name:
                    base_angle = 1.57  # 90度（π/2ラジアン）
                    joint_type = 'shoulder'
                elif 'knee' in joint_name:
                    base_angle = 0.0   # 0度
                    joint_type = 'knee'
                else:
                    base_angle = 0.0   # その他は0度
                    joint_type = 'other'
                
                # 多様化の適用
                initial_angle = self._apply_pose_diversity(
                    base_angle, joint_type, diversity_mode, pose_config
                )
                
                # 関節角度を設定
                p.resetJointState(self.robot_id, joint_id, initial_angle)
                print(f"Set {joint_name} to {initial_angle:.2f} rad ({initial_angle * 180 / 3.14159:.1f}°) [mode: {diversity_mode}]")
    
    def _apply_pose_diversity(self, base_angle: float, joint_type: str, 
                             diversity_mode: str, pose_config: dict) -> float:
        """初期姿勢の多様化を適用"""
        # 学習段階適応の適用（段階適応が有効な場合のみ）
        if self.stage_adaptation_enabled:
            diversity_mode = self._get_stage_adapted_mode(diversity_mode, pose_config)
        
        if diversity_mode == 'fixed':
            return base_angle
        
        # 変動範囲の取得
        variation_ranges = pose_config.get('variation_ranges', {})
        constraints = pose_config.get('constraints', {})
        
        if joint_type not in variation_ranges:
            return base_angle
        
        # 変動範囲の設定
        if diversity_mode == 'light':
            variation = variation_ranges[joint_type].get('light', 0.0)
        elif diversity_mode == 'medium':
            variation = variation_ranges[joint_type].get('medium', 0.0)
        elif diversity_mode == 'heavy':
            variation = variation_ranges[joint_type].get('heavy', 0.0)
        elif diversity_mode == 'random':
            # 物理制約内で完全ランダム
            min_angle = constraints.get(f'{joint_type}_min', base_angle - 0.5)
            max_angle = constraints.get(f'{joint_type}_max', base_angle + 0.5)
            return np.random.uniform(min_angle, max_angle)
        else:
            return base_angle
        
        # ランダム変動の適用
        random_variation = np.random.uniform(-variation, variation)
        new_angle = base_angle + random_variation
        
        # 物理制約の適用
        min_angle = constraints.get(f'{joint_type}_min', new_angle - 1.0)
        max_angle = constraints.get(f'{joint_type}_max', new_angle + 1.0)
        new_angle = np.clip(new_angle, min_angle, max_angle)
        
        return new_angle
    
    def _get_stage_adapted_mode(self, base_mode: str, pose_config: dict) -> str:
        """学習段階に応じた多様化モードを取得"""
        stage_adaptation = pose_config.get('stage_adaptation', {})
        if not stage_adaptation.get('enabled', False):
            return base_mode
        
        stages = stage_adaptation.get('stages', {})
        
        # 学習段階に応じたモード選択
        if self.learning_stage == 1:
            return stages.get('stage1', base_mode)
        elif self.learning_stage == 2:
            return stages.get('stage2', base_mode)
        elif self.learning_stage == 3:
            return stages.get('stage3', base_mode)
        elif self.learning_stage >= 4:
            return stages.get('stage4', base_mode)
        else:
            return base_mode
    
    def update_learning_stage(self, stage: int):
        """学習段階を更新"""
        self.learning_stage = stage
        print(f"Learning stage updated to: {stage}")
    
    def get_learning_stage(self) -> int:
        """現在の学習段階を取得"""
        return self.learning_stage
    
    def _set_target_position(self):
        """目標位置の設定"""
        # 通路の終端を目標位置に設定（安全マージンを考慮）
        corridor_length = self.config['environment']['corridor']['length']
        safety_margin = 0.10  # 10cmの安全マージン（手のひらサイズロボット用、2倍に拡大）
        self.target_position = np.array([corridor_length - safety_margin, 0, 0])
    
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
        
        # デバッグ用: 最初の数ステップで行動と関節角度を出力
        if self.current_step < 5:
            print(f"Step {self.current_step}: Action={action[:3]}, Joint targets={joint_targets[:3]}")
        
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
                else:
                    if self.current_step < 5:
                        print(f"Warning: Joint {joint_name} not found")
    
    def _action_to_joint_angles(self, action: np.ndarray) -> np.ndarray:
        """行動を関節角度に変換"""
        # CUDA tensorの場合はCPUに移動してnumpyに変換
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        
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
        max_angle = 1.0  # 約60度（より寛容な設定）
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
    
    def render(self, mode: str = "human"):
        """
        Gymnasiumの標準的なレンダリング機能
        
        Args:
            mode: レンダリングモード ("human", "rgb_array")
            
        Returns:
            mode="rgb_array"の場合: RGB画像配列
            mode="human"の場合: None
        """
        if mode == "human":
            # PyBulletのGUIモードで表示
            if not self._render_enabled:
                # 現在DIRECTモードの場合は、GUIモードに切り替え
                p.disconnect(self.physics_client)
                self.physics_client = p.connect(p.GUI)
                self._render_enabled = True
                # 環境を再構築（resetメソッドを使用）
                self.reset()
            return None
            
        elif mode == "rgb_array":
            # RGB画像配列を返す
            try:
                # カメラパラメータの設定（ロボットの位置に追従）
                camera_params = self.config.get('visualization', {}).get('camera', {})
                distance = camera_params.get('distance', 2.0)
                yaw = camera_params.get('yaw', 0.0)
                pitch = camera_params.get('pitch', -30.0)
                
                # ロボットの現在位置を取得してカメラのターゲットに設定
                robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                target_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.1]  # ロボットの少し上をターゲット
                
                # カメラ画像の取得
                width, height = 640, 480
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=target_pos,
                    distance=distance,
                    yaw=yaw,
                    pitch=pitch,
                    roll=0,
                    upAxisIndex=2
                )
                projection_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=width/height,
                    nearVal=0.1,
                    farVal=100.0
                )
                
                # 画像の取得（DIRECTモードでも動作するように修正）
                _, _, rgb_array, _, _ = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=p.ER_TINY_RENDERER  # DIRECTモードでも動作するレンダラー
                )
                
                # RGB配列の整形
                rgb_array = np.array(rgb_array, dtype=np.uint8)
                rgb_array = rgb_array[:, :, :3]  # Alphaチャンネルを除去
                
                return rgb_array
                
            except Exception as e:
                # エラーが発生した場合はダミー画像を返す
                print(f"レンダリングエラー: {e}")
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # テスト用にグレーの背景を設定
                dummy_frame.fill(128)
                return dummy_frame
            
        else:
            raise ValueError(f"サポートされていないレンダリングモード: {mode}")
    
    def close(self):
        """環境の終了"""
        p.disconnect(self.physics_client)
    
    def seed(self, seed=None):
        """乱数シードの設定"""
        np.random.seed(seed)
        return [seed]
