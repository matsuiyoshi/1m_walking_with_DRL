"""
Optimized Reward Functions for Bittle Walking Environment
最適化されたBittle四足歩行ロボット用の報酬関数
"""

import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple


class OptimizedRewardFunction:
    """
    最適化されたBittle四足歩行ロボット用の報酬関数
    
    主な改善点:
    - より密な報酬設計（Dense Rewards）
    - ペナルティの軽減と適正化
    - 段階的学習サポート
    - 学習安定性の向上
    """
    
    def __init__(self, reward_config: Dict[str, Any]):
        """
        報酬関数の初期化
        
        Args:
            reward_config: 報酬設定辞書
        """
        self.config = reward_config
        
        # 前回の状態を保存（差分計算用）
        self.prev_position = None
        self.prev_orientation = None
        self.prev_joint_torques = None
        
        # 段階的学習の状態
        self.learning_stage = 0  # 0: 基本歩行, 1: 安定性, 2: 効率性
        self.stage_progress = 0.0
    
    def calculate_reward(self, robot_id: int, target_position: np.ndarray, 
                        corridor_bounds: Dict[str, float], step: int) -> float:
        """
        最適化された総合報酬の計算
        
        Args:
            robot_id: ロボットのID
            target_position: 目標位置
            corridor_bounds: 通路の境界
            step: 現在のステップ数
            
        Returns:
            計算された報酬値
        """
        # 現在の状態を取得
        current_state = self._get_current_state(robot_id)
        
        # ベース報酬（常に正の値で開始）
        base_reward = self.config.get('base_reward', 0.1)
        total_reward = base_reward
        
        # 段階的報酬計算
        if self.learning_stage == 0:
            # Stage 0: 基本歩行学習
            total_reward += self._calculate_basic_walking_reward(current_state, target_position)
        elif self.learning_stage == 1:
            # Stage 1: 安定性向上
            total_reward += self._calculate_stability_focused_reward(current_state, target_position, corridor_bounds)
        else:
            # Stage 2: 効率性最適化
            total_reward += self._calculate_efficiency_focused_reward(current_state, target_position, corridor_bounds)
        
        # 共通報酬（全段階共通）
        total_reward += self._calculate_common_rewards(current_state, target_position, corridor_bounds, step)
        
        # 軽減されたペナルティ
        penalty = self._calculate_soft_penalties(current_state, corridor_bounds)
        total_reward += penalty  # ペナルティは負の値だが軽減済み
        
        # 状態の更新
        self._update_previous_state(current_state)
        
        return total_reward
    
    def _calculate_basic_walking_reward(self, state: Dict[str, Any], 
                                      target_position: np.ndarray) -> float:
        """基本歩行学習用の報酬（Stage 0）"""
        reward = 0.0
        
        # 1. 前進報酬（シンプルで密な報酬）
        if self.prev_position is not None:
            forward_movement = state['position'][0] - self.prev_position[0]
            forward_reward_scale = self.config.get('basic_forward_reward', 10.0)
            reward += forward_reward_scale * max(0, forward_movement)  # 前進のみ報酬
        
        # 2. 生存報酬（エピソードを継続することの価値）
        survival_reward = self.config.get('survival_reward', 0.01)
        reward += survival_reward
        
        # 3. 高さ維持報酬（倒れない奨励）
        height_reward_scale = self.config.get('height_reward', 1.0)
        height_bonus = min(1.0, max(0.0, (state['position'][2] - 0.05) / 0.1))  # 5cm以上で報酬
        reward += height_reward_scale * height_bonus
        
        return reward
    
    def _calculate_stability_focused_reward(self, state: Dict[str, Any], 
                                          target_position: np.ndarray,
                                          corridor_bounds: Dict[str, float]) -> float:
        """安定性重視の報酬（Stage 1）"""
        reward = 0.0
        
        # 1. 前進報酬（継続）
        if self.prev_position is not None:
            forward_movement = state['position'][0] - self.prev_position[0]
            forward_reward_scale = self.config.get('stability_forward_reward', 8.0)
            reward += forward_reward_scale * max(0, forward_movement)
        
        # 2. 姿勢安定性報酬
        orientation_reward_scale = self.config.get('orientation_reward', 2.0)
        roll, pitch, yaw = state['orientation']
        
        # 理想的な姿勢に近いほど高い報酬
        orientation_penalty = abs(roll) + abs(pitch)
        orientation_bonus = max(0.0, 1.0 - orientation_penalty / (np.pi/4))  # 45度以内で正の報酬
        reward += orientation_reward_scale * orientation_bonus
        
        # 3. 通路中央維持報酬
        corridor_center_reward_scale = self.config.get('corridor_center_reward', 1.5)
        y_position = state['position'][1]
        # 互換性のためにキーを確認
        if 'y_min' in corridor_bounds and 'y_max' in corridor_bounds:
            corridor_width = corridor_bounds['y_max'] - corridor_bounds['y_min']
            corridor_center = (corridor_bounds['y_max'] + corridor_bounds['y_min']) / 2
        else:
            # 従来のキー形式（left/right）
            corridor_width = corridor_bounds['right'] - corridor_bounds['left']
            corridor_center = (corridor_bounds['right'] + corridor_bounds['left']) / 2
        
        distance_from_center = abs(y_position - corridor_center)
        center_bonus = max(0.0, 1.0 - (distance_from_center / (corridor_width / 2)))
        reward += corridor_center_reward_scale * center_bonus
        
        return reward
    
    def _calculate_efficiency_focused_reward(self, state: Dict[str, Any], 
                                           target_position: np.ndarray,
                                           corridor_bounds: Dict[str, float]) -> float:
        """効率性重視の報酬（Stage 2）"""
        reward = 0.0
        
        # 1. 速度効率報酬
        speed_reward_scale = self.config.get('speed_reward', 5.0)
        forward_velocity = state['linear_velocity'][0]
        target_speed = self.config.get('target_speed', 0.3)  # 30cm/s
        
        # 目標速度に近いほど高い報酬
        speed_efficiency = max(0.0, 1.0 - abs(forward_velocity - target_speed) / target_speed)
        reward += speed_reward_scale * speed_efficiency
        
        # 2. エネルギー効率報酬
        energy_reward_scale = self.config.get('energy_reward', 1.0)
        if len(state['joint_torques']) > 0:
            torque_magnitude = np.mean(np.abs(state['joint_torques']))
            energy_efficiency = max(0.0, 1.0 - torque_magnitude / 10.0)  # 10Nm基準
            reward += energy_reward_scale * energy_efficiency
        
        return reward
    
    def _calculate_common_rewards(self, state: Dict[str, Any], 
                                target_position: np.ndarray,
                                corridor_bounds: Dict[str, float], 
                                step: int) -> float:
        """全段階共通の報酬"""
        reward = 0.0
        
        # 1. ゴール距離報酬（密な報酬）
        goal_distance_reward_scale = self.config.get('goal_distance_reward', 3.0)
        current_distance = np.linalg.norm(state['position'][:2] - target_position[:2])
        max_distance = 1.5  # 最大距離
        
        # 距離に反比例する報酬（常に正の値）
        distance_reward = max(0.1, 1.0 - (current_distance / max_distance))
        reward += goal_distance_reward_scale * distance_reward
        
        # 2. ゴール到達ボーナス
        if current_distance < 0.1:  # 10cm以内
            goal_bonus = self.config.get('goal_bonus', 50.0)
            reward += goal_bonus
        
        return reward
    
    def _calculate_soft_penalties(self, state: Dict[str, Any], 
                                corridor_bounds: Dict[str, float]) -> float:
        """軽減されたペナルティ"""
        penalty = 0.0
        
        # 1. 通路逸脱ペナルティ（軽減）
        y_position = state['position'][1]
        # 互換性のためにキーを確認
        if 'y_min' in corridor_bounds and 'y_max' in corridor_bounds:
            out_of_bounds = y_position < corridor_bounds['y_min'] or y_position > corridor_bounds['y_max']
        else:
            # 従来のキー形式（left/right）
            out_of_bounds = y_position < corridor_bounds['left'] or y_position > corridor_bounds['right']
            
        if out_of_bounds:
            boundary_penalty = self.config.get('boundary_penalty', -0.5)  # 軽減済み
            penalty += boundary_penalty
        
        # 2. 極端な姿勢ペナルティ（軽減）
        roll, pitch, yaw = state['orientation']
        extreme_angle_threshold = np.pi / 3  # 60度
        
        if abs(roll) > extreme_angle_threshold or abs(pitch) > extreme_angle_threshold:
            pose_penalty = self.config.get('pose_penalty', -0.3)  # 軽減済み
            penalty += pose_penalty
        
        # 3. 高さペナルティ（軽減）
        if state['position'][2] < 0.02:  # 2cm以下
            height_penalty = self.config.get('height_penalty', -0.2)  # 軽減済み
            penalty += height_penalty
        
        return penalty
    
    def _get_current_state(self, robot_id: int) -> Dict[str, Any]:
        """現在の状態を取得"""
        # 位置・姿勢
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # 速度・角速度
        linear_vel, angular_vel = p.getBaseVelocity(robot_id)
        
        # 関節状態
        joint_states = self._get_joint_states(robot_id)
        
        return {
            'position': np.array(pos),
            'orientation': np.array(euler),
            'linear_velocity': np.array(linear_vel),
            'angular_velocity': np.array(angular_vel),
            'joint_angles': joint_states['angles'],
            'joint_velocities': joint_states['velocities'],
            'joint_torques': joint_states['torques']
        }
    
    def _get_joint_states(self, robot_id: int) -> Dict[str, np.ndarray]:
        """関節状態の取得"""
        num_joints = p.getNumJoints(robot_id)
        angles = []
        velocities = []
        torques = []
        
        for i in range(num_joints):
            joint_state = p.getJointState(robot_id, i)
            angles.append(joint_state[0])  # 角度
            velocities.append(joint_state[1])  # 角速度
            torques.append(joint_state[3])  # トルク
        
        return {
            'angles': np.array(angles),
            'velocities': np.array(velocities),
            'torques': np.array(torques)
        }
    
    def _update_previous_state(self, state: Dict[str, Any]):
        """前回の状態を更新"""
        self.prev_position = state['position'].copy()
        self.prev_orientation = state['orientation'].copy()
        self.prev_joint_torques = state['joint_torques'].copy()
    
    def update_learning_stage(self, success_rate: float, avg_reward: float):
        """学習段階の更新"""
        # Stage 0 -> 1: 基本歩行が安定したら
        if self.learning_stage == 0 and avg_reward > 5.0 and success_rate > 0.2:
            self.learning_stage = 1
            print(f"Learning Stage updated to 1 (Stability Focus)")
        
        # Stage 1 -> 2: 安定性が向上したら
        elif self.learning_stage == 1 and avg_reward > 10.0 and success_rate > 0.5:
            self.learning_stage = 2
            print(f"Learning Stage updated to 2 (Efficiency Focus)")
