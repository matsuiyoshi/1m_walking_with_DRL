#!/usr/bin/env python3
"""
学習中のモデル挙動分析スクリプト
学習プロセス中のモデルの具体的な挙動変化を詳細に記録・分析
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import torch
from collections import defaultdict, deque
import json
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.parallel_trainer import ParallelTrainer
from src.environment.bittle_env import BittleWalkingEnv
from src.models.ppo_agent import PPOAgent


class LearningBehaviorAnalyzer:
    """学習中のモデル挙動分析器"""
    
    def __init__(self, output_dir: str = "data/learning_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分析データの保存
        self.episode_data = []
        self.action_analysis = defaultdict(list)
        self.reward_breakdown = defaultdict(list)
        self.joint_angle_evolution = defaultdict(list)
        self.position_trajectories = []
        self.success_patterns = []
        self.failure_patterns = []
        
        # 統計情報
        self.step_count = 0
        self.episode_count = 0
        
    def analyze_episode(self, env, agent, episode_num: int, render: bool = False):
        """単一エピソードの詳細分析"""
        obs = env.reset()
        episode_data = {
            'episode': episode_num,
            'steps': [],
            'total_reward': 0,
            'success': False,
            'failure_reason': None,
            'final_distance': 0,
            'max_forward_distance': 0,
            'stability_metrics': [],
            'joint_angles_history': [],
            'joint_velocities_history': [],
            'position_history': [],
            'orientation_history': [],
            'action_history': [],
            'reward_breakdown': []
        }
        
        step = 0
        max_steps = 1000
        
        while step < max_steps:
            # 行動の取得
            action, log_prob, value = agent.get_action(obs, deterministic=False)
            
            # 環境ステップ
            next_obs, reward, done, info = env.step(action)
            
            # 詳細データの記録
            step_data = {
                'step': step,
                'observation': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'log_prob': log_prob,
                'value': value,
                'done': done,
                'info': info.copy()
            }
            
            # 関節角度と速度の記録
            joint_angles, joint_velocities = self._extract_joint_data(obs)
            step_data['joint_angles'] = joint_angles
            step_data['joint_velocities'] = joint_velocities
            
            # 位置・姿勢の記録
            position, orientation = self._extract_pose_data(obs)
            step_data['position'] = position
            step_data['orientation'] = orientation
            
            # 報酬の詳細分析
            reward_components = self._analyze_reward_components(env, obs, action, reward)
            step_data['reward_components'] = reward_components
            
            episode_data['steps'].append(step_data)
            episode_data['total_reward'] += reward
            episode_data['joint_angles_history'].append(joint_angles)
            episode_data['joint_velocities_history'].append(joint_velocities)
            episode_data['position_history'].append(position)
            episode_data['orientation_history'].append(orientation)
            episode_data['action_history'].append(action)
            episode_data['reward_breakdown'].append(reward_components)
            
            # 前進距離の追跡
            forward_distance = position[0]  # X座標
            episode_data['max_forward_distance'] = max(episode_data['max_forward_distance'], forward_distance)
            
            # 安定性指標の計算
            stability = self._calculate_stability_metrics(orientation)
            episode_data['stability_metrics'].append(stability)
            
            obs = next_obs
            step += 1
            
            if done:
                break
        
        # エピソード終了時の分析
        episode_data['final_distance'] = episode_data['max_forward_distance']
        episode_data['success'] = episode_data['final_distance'] >= 0.8  # 80cm以上で成功
        episode_data['episode_length'] = step
        
        if not episode_data['success']:
            episode_data['failure_reason'] = self._analyze_failure_reason(episode_data)
        
        # データの保存
        self.episode_data.append(episode_data)
        
        return episode_data
    
    def _extract_joint_data(self, obs):
        """観測から関節データを抽出"""
        # 観測の構造: [joint_angles(8), joint_velocities(8), imu(6), pose(6), target(2)]
        joint_angles = obs[:8]
        joint_velocities = obs[8:16]
        return joint_angles, joint_velocities
    
    def _extract_pose_data(self, obs):
        """観測から位置・姿勢データを抽出"""
        # IMUデータの後から位置・姿勢を取得
        position = obs[22:25]  # X, Y, Z
        orientation = obs[25:28]  # Roll, Pitch, Yaw
        return position, orientation
    
    def _analyze_reward_components(self, env, obs, action, reward):
        """報酬の詳細分析"""
        # 実際の報酬計算ロジックに基づいて分析
        # ここでは簡易的な分析を実装
        components = {
            'forward_progress': 0,
            'stability': 0,
            'corridor_stay': 0,
            'penalties': 0
        }
        
        # 前進進捗
        position, _ = self._extract_pose_data(obs)
        components['forward_progress'] = position[0] * 10  # 簡易計算
        
        # 安定性
        _, orientation = self._extract_pose_data(obs)
        pitch, roll = orientation[1], orientation[0]
        stability = 1 - (abs(pitch) + abs(roll)) / 2
        components['stability'] = stability * 5
        
        return components
    
    def _calculate_stability_metrics(self, orientation):
        """安定性指標の計算"""
        pitch, roll = orientation[1], orientation[0]
        return {
            'pitch_abs': abs(pitch),
            'roll_abs': abs(roll),
            'stability_score': 1 - (abs(pitch) + abs(roll)) / 2
        }
    
    def _analyze_failure_reason(self, episode_data):
        """失敗原因の分析"""
        if episode_data['max_forward_distance'] < 0.1:
            return "no_progress"
        elif max(episode_data['stability_metrics'])['pitch_abs'] > 1.0:
            return "unstable_pitch"
        elif max(episode_data['stability_metrics'])['roll_abs'] > 1.0:
            return "unstable_roll"
        else:
            return "insufficient_progress"
    
    def generate_analysis_report(self, model_path: str):
        """分析レポートの生成"""
        report = {
            'model_path': model_path,
            'analysis_time': datetime.now().isoformat(),
            'total_episodes': len(self.episode_data),
            'success_rate': sum(1 for ep in self.episode_data if ep['success']) / len(self.episode_data),
            'average_reward': np.mean([ep['total_reward'] for ep in self.episode_data]),
            'average_distance': np.mean([ep['final_distance'] for ep in self.episode_data]),
            'failure_reasons': self._analyze_failure_patterns(),
            'learning_progress': self._analyze_learning_progress()
        }
        
        # レポートの保存
        report_path = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _analyze_failure_patterns(self):
        """失敗パターンの分析"""
        failure_reasons = [ep['failure_reason'] for ep in self.episode_data if not ep['success']]
        return dict(zip(*np.unique(failure_reasons, return_counts=True)))
    
    def _analyze_learning_progress(self):
        """学習進捗の分析"""
        if len(self.episode_data) < 10:
            return {}
        
        # エピソードを10個ずつのグループに分割
        group_size = 10
        groups = [self.episode_data[i:i+group_size] for i in range(0, len(self.episode_data), group_size)]
        
        progress = []
        for i, group in enumerate(groups):
            if group:
                progress.append({
                    'group': i,
                    'episodes': f"{i*group_size}-{(i+1)*group_size-1}",
                    'success_rate': sum(1 for ep in group if ep['success']) / len(group),
                    'avg_reward': np.mean([ep['total_reward'] for ep in group]),
                    'avg_distance': np.mean([ep['final_distance'] for ep in group])
                })
        
        return progress
    
    def create_visualizations(self):
        """可視化の作成"""
        if not self.episode_data:
            return
        
        # 1. 報酬の推移
        self._plot_reward_progression()
        
        # 2. 関節角度の変化
        self._plot_joint_angle_evolution()
        
        # 3. 位置軌跡
        self._plot_position_trajectories()
        
        # 4. 行動の分布
        self._plot_action_distributions()
        
        # 5. 安定性指標
        self._plot_stability_metrics()
    
    def _plot_reward_progression(self):
        """報酬の推移プロット"""
        episodes = [ep['episode'] for ep in self.episode_data]
        rewards = [ep['total_reward'] for ep in self.episode_data]
        distances = [ep['final_distance'] for ep in self.episode_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 報酬の推移
        ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Total Reward')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward Progression')
        ax1.legend()
        ax1.grid(True)
        
        # 距離の推移
        ax2.plot(episodes, distances, 'g-', alpha=0.7, label='Final Distance')
        ax2.axhline(y=0.8, color='r', linestyle='--', label='Success Threshold (0.8m)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title('Distance Progression')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reward_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_joint_angle_evolution(self):
        """関節角度の変化プロット"""
        if not self.episode_data:
            return
        
        # 最新のエピソードの関節角度履歴を取得
        latest_episode = self.episode_data[-1]
        joint_history = latest_episode['joint_angles_history']
        
        if not joint_history:
            return
        
        joint_history = np.array(joint_history)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        joint_names = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 
                      'BL_hip', 'BL_knee', 'BR_hip', 'BR_knee']
        
        for i in range(8):
            row, col = i // 4, i % 4
            axes[row, col].plot(joint_history[:, i], label=joint_names[i])
            axes[row, col].set_title(f'{joint_names[i]} Angle')
            axes[row, col].set_xlabel('Step')
            axes[row, col].set_ylabel('Angle (rad)')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'joint_angle_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_trajectories(self):
        """位置軌跡のプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, episode in enumerate(self.episode_data[-5:]):  # 最新5エピソード
            if episode['position_history']:
                positions = np.array(episode['position_history'])
                ax.plot(positions[:, 0], positions[:, 1], 
                       label=f'Episode {episode["episode"]}', alpha=0.7)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Robot Trajectories (Latest 5 Episodes)')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'position_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_distributions(self):
        """行動分布のプロット"""
        if not self.episode_data:
            return
        
        # 全エピソードの行動を収集
        all_actions = []
        for episode in self.episode_data:
            if episode['action_history']:
                all_actions.extend(episode['action_history'])
        
        if not all_actions:
            return
        
        all_actions = np.array(all_actions)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        joint_names = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 
                      'BL_hip', 'BL_knee', 'BR_hip', 'BR_knee']
        
        for i in range(8):
            row, col = i // 4, i % 4
            axes[row, col].hist(all_actions[:, i], bins=50, alpha=0.7)
            axes[row, col].set_title(f'{joint_names[i]} Action Distribution')
            axes[row, col].set_xlabel('Action Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'action_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_metrics(self):
        """安定性指標のプロット"""
        if not self.episode_data:
            return
        
        # 全エピソードの安定性指標を収集
        all_stability = []
        for episode in self.episode_data:
            if episode['stability_metrics']:
                all_stability.extend(episode['stability_metrics'])
        
        if not all_stability:
            return
        
        stability_scores = [s['stability_score'] for s in all_stability]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 安定性スコアの分布
        ax1.hist(stability_scores, bins=50, alpha=0.7)
        ax1.set_xlabel('Stability Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Stability Score Distribution')
        ax1.grid(True)
        
        # エピソードごとの平均安定性
        episode_stability = []
        for episode in self.episode_data:
            if episode['stability_metrics']:
                avg_stability = np.mean([s['stability_score'] for s in episode['stability_metrics']])
                episode_stability.append(avg_stability)
        
        if episode_stability:
            ax2.plot(episode_stability, 'b-', alpha=0.7)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Stability Score')
            ax2.set_title('Stability Progression')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Learning Behavior Analysis')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of episodes to analyze')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during analysis')
    parser.add_argument('--output-dir', type=str, default='data/learning_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 分析器の初期化
        analyzer = LearningBehaviorAnalyzer(args.output_dir)
        
        # 環境とエージェントの設定
        env = BittleWalkingEnv(render=args.render)
        
        # エージェントの初期化
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
        agent.load(args.model_path)
        
        logger.info(f"学習挙動分析を開始します（エピソード数: {args.num_episodes}）")
        
        # エピソード分析の実行
        for episode in range(args.num_episodes):
            logger.info(f"エピソード {episode + 1}/{args.num_episodes} を分析中...")
            episode_data = analyzer.analyze_episode(env, agent, episode, args.render)
            
            logger.info(f"  報酬: {episode_data['total_reward']:.2f}")
            logger.info(f"  距離: {episode_data['final_distance']:.3f}m")
            logger.info(f"  成功: {episode_data['success']}")
            if not episode_data['success']:
                logger.info(f"  失敗理由: {episode_data['failure_reason']}")
        
        # 分析レポートの生成
        logger.info("分析レポートを生成中...")
        report = analyzer.generate_analysis_report(args.model_path)
        
        # 可視化の作成
        logger.info("可視化を作成中...")
        analyzer.create_visualizations()
        
        # 結果の表示
        logger.info("=" * 50)
        logger.info("学習挙動分析結果:")
        logger.info(f"  総エピソード数: {report['total_episodes']}")
        logger.info(f"  成功率: {report['success_rate']:.2%}")
        logger.info(f"  平均報酬: {report['average_reward']:.2f}")
        logger.info(f"  平均距離: {report['average_distance']:.3f}m")
        logger.info(f"  失敗理由分布: {report['failure_reasons']}")
        logger.info("=" * 50)
        
        logger.info(f"分析結果を保存しました: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
