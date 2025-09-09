"""
Reward Functions for Bittle Walking Environment
Bittle四足歩行ロボット用の報酬関数
"""

import numpy as np
import pybullet as p
from typing import Dict, Any, Tuple


class RewardFunction:
    """
    Bittle四足歩行ロボット用の報酬関数
    
    報酬構成:
    - 前進報酬: 前進距離と方向性
    - 安定性報酬: 転倒防止と姿勢維持
    - 効率性報酬: 速度とエネルギー効率
    - ペナルティ: 制約違反
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
    
    def calculate_reward(self, robot_id: int, target_position: np.ndarray, 
                        corridor_bounds: Dict[str, float], step: int) -> float:
        """
        総合報酬の計算
        
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
        
        # 各報酬成分の計算
        forward_reward = self._calculate_forward_reward(current_state, target_position)
        stability_reward = self._calculate_stability_reward(current_state, corridor_bounds)
        efficiency_reward = self._calculate_efficiency_reward(current_state)
        penalty = self._calculate_penalty(current_state, corridor_bounds)
        
        # 成功ボーナス
        success_bonus = self._calculate_success_bonus(current_state, target_position)
        
        # 総合報酬
        total_reward = (forward_reward + stability_reward + 
                       efficiency_reward + penalty + success_bonus)
        
        # 状態の更新
        self._update_previous_state(current_state)
        
        return total_reward
    
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
    
    def _calculate_forward_reward(self, state: Dict[str, Any], 
                                 target_position: np.ndarray) -> float:
        """前進報酬の計算"""
        reward = 0.0
        
        # 前進距離報酬
        if self.prev_position is not None:
            # X軸方向の前進距離
            forward_distance = state['position'][0] - self.prev_position[0]
            reward += self.config['forward_reward'] * forward_distance
            
            # 目標方向への移動報酬
            if forward_distance > 0:
                # 目標方向への角度
                target_direction = target_position - state['position']
                target_direction = target_direction / np.linalg.norm(target_direction)
                forward_direction = np.array([1, 0, 0])  # 前進方向
                
                direction_alignment = np.dot(target_direction, forward_direction)
                reward += self.config['direction_reward'] * direction_alignment * forward_distance
        
        return reward
    
    def _calculate_stability_reward(self, state: Dict[str, Any], 
                                   corridor_bounds: Dict[str, float]) -> float:
        """安定性報酬の計算"""
        reward = 0.0
        
        # 転倒ペナルティ
        pitch, roll = state['orientation'][0], state['orientation'][1]
        if abs(pitch) > 0.5 or abs(roll) > 0.5:  # 約30度
            reward += self.config['fall_penalty']
        
        # 姿勢安定性報酬
        stability = 1.0 - abs(pitch) - abs(roll)
        reward += self.config['stability_reward'] * stability
        
        # 通路内維持報酬
        x, y = state['position'][0], state['position'][1]
        if (corridor_bounds['left'] <= y <= corridor_bounds['right'] and
            corridor_bounds['back'] <= x <= corridor_bounds['front']):
            reward += self.config['corridor_reward']
        else:
            reward += self.config['corridor_penalty']
        
        return reward
    
    def _calculate_efficiency_reward(self, state: Dict[str, Any]) -> float:
        """効率性報酬の計算"""
        reward = 0.0
        
        # 速度報酬（前進速度）
        forward_velocity = state['linear_velocity'][0]
        reward += self.config['speed_reward'] * forward_velocity
        
        # エネルギー効率ペナルティ
        if state['joint_torques'] is not None:
            energy_consumption = np.sum(np.abs(state['joint_torques']))
            reward += self.config['energy_penalty'] * energy_consumption
        
        return reward
    
    def _calculate_penalty(self, state: Dict[str, Any], 
                          corridor_bounds: Dict[str, float]) -> float:
        """ペナルティの計算"""
        penalty = 0.0
        
        # 関節限界違反ペナルティ
        if state['joint_angles'] is not None:
            # 関節角度の制限チェック（簡易実装）
            for angle in state['joint_angles']:
                if abs(angle) > 1.57:  # 約90度
                    penalty += self.config['joint_limit_penalty']
        
        # 急激な動作ペナルティ
        if (self.prev_joint_torques is not None and 
            state['joint_torques'] is not None):
            torque_change = np.sum(np.abs(state['joint_torques'] - self.prev_joint_torques))
            penalty += self.config['torque_change_penalty'] * torque_change
        
        return penalty
    
    def _calculate_success_bonus(self, state: Dict[str, Any], 
                                target_position: np.ndarray) -> float:
        """成功ボーナスの計算"""
        bonus = 0.0
        
        # 目標到達ボーナス
        distance_to_target = np.linalg.norm(state['position'][:2] - target_position[:2])
        if distance_to_target < 0.1:  # 10cm以内
            bonus += self.config['success_bonus']
        
        # 時間ボーナス（早く完了した場合）
        # これは環境側で管理されるべきだが、簡易実装として含める
        if distance_to_target < 0.1:
            # 仮想的な時間ボーナス（実際の実装では環境から時間情報を受け取る）
            bonus += self.config['time_bonus']
        
        return bonus
    
    def _update_previous_state(self, current_state: Dict[str, Any]):
        """前回の状態を更新"""
        self.prev_position = current_state['position'].copy()
        self.prev_orientation = current_state['orientation'].copy()
        if current_state['joint_torques'] is not None:
            self.prev_joint_torques = current_state['joint_torques'].copy()
    
    def reset(self):
        """報酬関数のリセット"""
        self.prev_position = None
        self.prev_orientation = None
        self.prev_joint_torques = None


class ShapedRewardFunction(RewardFunction):
    """
    シェイプされた報酬関数
    
    より詳細な報酬設計により、学習の安定性と効率性を向上
    """
    
    def __init__(self, reward_config: Dict[str, Any]):
        super().__init__(reward_config)
        
        # 追加の報酬パラメータ
        self.velocity_smoothing_weight = 0.1
        self.joint_smoothness_weight = 0.05
        self.energy_efficiency_weight = 0.02
    
    def _calculate_efficiency_reward(self, state: Dict[str, Any]) -> float:
        """拡張された効率性報酬"""
        reward = super()._calculate_efficiency_reward(state)
        
        # 速度の滑らかさ
        if self.prev_position is not None:
            current_velocity = state['linear_velocity']
            prev_velocity = (state['position'] - self.prev_position) / self.timestep
            
            velocity_smoothness = -np.linalg.norm(current_velocity - prev_velocity)
            reward += self.velocity_smoothing_weight * velocity_smoothness
        
        # 関節動作の滑らかさ
        if (self.prev_joint_torques is not None and 
            state['joint_torques'] is not None):
            joint_smoothness = -np.sum(np.abs(state['joint_torques'] - self.prev_joint_torques))
            reward += self.joint_smoothness_weight * joint_smoothness
        
        return reward
    
    def _calculate_penalty(self, state: Dict[str, Any], 
                          corridor_bounds: Dict[str, float]) -> float:
        """拡張されたペナルティ"""
        penalty = super()._calculate_penalty(state, corridor_bounds)
        
        # エネルギー効率ペナルティ（より詳細）
        if state['joint_torques'] is not None:
            # 関節速度とトルクの積（パワー）を考慮
            power = np.sum(np.abs(state['joint_torques'] * state['joint_velocities']))
            penalty += self.energy_efficiency_weight * power
        
        return penalty
