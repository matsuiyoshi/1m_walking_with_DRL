#!/usr/bin/env python3
"""
詳細学習監視スクリプト
学習中のモデルの挙動をリアルタイムで詳細監視
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import torch
from collections import deque
import json
from datetime import datetime
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.parallel_trainer import ParallelTrainer


class DetailedLearningMonitor:
    """詳細学習監視器"""
    
    def __init__(self, trainer: ParallelTrainer, output_dir: str = "data/detailed_monitoring"):
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 監視データの保存
        self.monitoring_data = {
            'episode_rewards': deque(maxlen=1000),
            'episode_lengths': deque(maxlen=1000),
            'action_statistics': deque(maxlen=1000),
            'joint_angle_statistics': deque(maxlen=1000),
            'stability_metrics': deque(maxlen=1000),
            'reward_components': deque(maxlen=1000),
            'learning_metrics': deque(maxlen=1000)
        }
        
        # リアルタイム監視設定
        self.monitor_frequency = 10  # 10エピソードごとに詳細分析
        self.last_analysis_time = time.time()
        
    def monitor_episode(self, episode_data: dict):
        """エピソードの詳細監視"""
        # 基本統計の記録
        self.monitoring_data['episode_rewards'].append(episode_data.get('total_reward', 0))
        self.monitoring_data['episode_lengths'].append(episode_data.get('episode_length', 0))
        
        # 行動統計の分析
        if 'action_history' in episode_data:
            actions = np.array(episode_data['action_history'])
            action_stats = {
                'mean': np.mean(actions, axis=0).tolist(),
                'std': np.std(actions, axis=0).tolist(),
                'min': np.min(actions, axis=0).tolist(),
                'max': np.max(actions, axis=0).tolist(),
                'range': (np.max(actions, axis=0) - np.min(actions, axis=0)).tolist()
            }
            self.monitoring_data['action_statistics'].append(action_stats)
        
        # 関節角度統計の分析
        if 'joint_angles_history' in episode_data:
            joint_angles = np.array(episode_data['joint_angles_history'])
            joint_stats = {
                'mean': np.mean(joint_angles, axis=0).tolist(),
                'std': np.std(joint_angles, axis=0).tolist(),
                'range': (np.max(joint_angles, axis=0) - np.min(joint_angles, axis=0)).tolist()
            }
            self.monitoring_data['joint_angle_statistics'].append(joint_stats)
        
        # 安定性指標の分析
        if 'stability_metrics' in episode_data:
            stability_data = episode_data['stability_metrics']
            if stability_data:
                stability_stats = {
                    'avg_stability': np.mean([s['stability_score'] for s in stability_data]),
                    'min_stability': np.min([s['stability_score'] for s in stability_data]),
                    'max_pitch': np.max([s['pitch_abs'] for s in stability_data]),
                    'max_roll': np.max([s['roll_abs'] for s in stability_data])
                }
                self.monitoring_data['stability_metrics'].append(stability_stats)
        
        # 報酬成分の分析
        if 'reward_breakdown' in episode_data:
            reward_components = episode_data['reward_breakdown']
            if reward_components:
                component_stats = {}
                for component in ['forward_progress', 'stability', 'corridor_stay', 'penalties']:
                    values = [r.get(component, 0) for r in reward_components]
                    component_stats[component] = {
                        'total': np.sum(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                self.monitoring_data['reward_components'].append(component_stats)
        
        # 定期的な詳細分析
        if len(self.monitoring_data['episode_rewards']) % self.monitor_frequency == 0:
            self._perform_detailed_analysis()
    
    def _perform_detailed_analysis(self):
        """詳細分析の実行"""
        current_time = time.time()
        
        # 学習指標の計算
        recent_rewards = list(self.monitoring_data['episode_rewards'])[-50:]  # 直近50エピソード
        recent_lengths = list(self.monitoring_data['episode_lengths'])[-50:]
        
        learning_metrics = {
            'timestamp': current_time,
            'episode_count': len(self.monitoring_data['episode_rewards']),
            'avg_reward_50': np.mean(recent_rewards),
            'std_reward_50': np.std(recent_rewards),
            'avg_length_50': np.mean(recent_lengths),
            'success_rate_50': sum(1 for r in recent_rewards if r > 100) / len(recent_rewards),  # 成功閾値
            'reward_trend': self._calculate_trend(recent_rewards),
            'length_trend': self._calculate_trend(recent_lengths)
        }
        
        # 行動の多様性分析
        if self.monitoring_data['action_statistics']:
            recent_actions = list(self.monitoring_data['action_statistics'])[-20:]
            action_diversity = self._analyze_action_diversity(recent_actions)
            learning_metrics['action_diversity'] = action_diversity
        
        # 関節角度の変化分析
        if self.monitoring_data['joint_angle_statistics']:
            recent_joints = list(self.monitoring_data['joint_angle_statistics'])[-20:]
            joint_evolution = self._analyze_joint_evolution(recent_joints)
            learning_metrics['joint_evolution'] = joint_evolution
        
        # 安定性の変化分析
        if self.monitoring_data['stability_metrics']:
            recent_stability = list(self.monitoring_data['stability_metrics'])[-20:]
            stability_evolution = self._analyze_stability_evolution(recent_stability)
            learning_metrics['stability_evolution'] = stability_evolution
        
        self.monitoring_data['learning_metrics'].append(learning_metrics)
        
        # 詳細分析レポートの生成
        self._generate_detailed_report(learning_metrics)
        
        # アラートのチェック
        self._check_learning_alerts(learning_metrics)
    
    def _calculate_trend(self, data):
        """データのトレンド計算"""
        if len(data) < 2:
            return 0
        
        x = np.arange(len(data))
        y = np.array(data)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _analyze_action_diversity(self, action_stats_list):
        """行動の多様性分析"""
        if not action_stats_list:
            return {}
        
        # 各関節の行動範囲の変化を分析
        diversity_metrics = {}
        for joint_idx in range(8):  # 8関節
            ranges = [stats['range'][joint_idx] for stats in action_stats_list]
            diversity_metrics[f'joint_{joint_idx}_range_std'] = np.std(ranges)
            diversity_metrics[f'joint_{joint_idx}_range_trend'] = self._calculate_trend(ranges)
        
        return diversity_metrics
    
    def _analyze_joint_evolution(self, joint_stats_list):
        """関節角度の進化分析"""
        if not joint_stats_list:
            return {}
        
        evolution_metrics = {}
        for joint_idx in range(8):
            means = [stats['mean'][joint_idx] for stats in joint_stats_list]
            stds = [stats['std'][joint_idx] for stats in joint_stats_list]
            
            evolution_metrics[f'joint_{joint_idx}_mean_trend'] = self._calculate_trend(means)
            evolution_metrics[f'joint_{joint_idx}_std_trend'] = self._calculate_trend(stds)
        
        return evolution_metrics
    
    def _analyze_stability_evolution(self, stability_stats_list):
        """安定性の進化分析"""
        if not stability_stats_list:
            return {}
        
        avg_stabilities = [s['avg_stability'] for s in stability_stats_list]
        max_pitches = [s['max_pitch'] for s in stability_stats_list]
        max_rolls = [s['max_roll'] for s in stability_stats_list]
        
        return {
            'stability_trend': self._calculate_trend(avg_stabilities),
            'pitch_instability_trend': self._calculate_trend(max_pitches),
            'roll_instability_trend': self._calculate_trend(max_rolls)
        }
    
    def _generate_detailed_report(self, learning_metrics):
        """詳細レポートの生成"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'learning_metrics': learning_metrics,
            'monitoring_summary': {
                'total_episodes': len(self.monitoring_data['episode_rewards']),
                'avg_reward_all': np.mean(self.monitoring_data['episode_rewards']),
                'std_reward_all': np.std(self.monitoring_data['episode_rewards']),
                'avg_length_all': np.mean(self.monitoring_data['episode_lengths']),
                'success_rate_all': sum(1 for r in self.monitoring_data['episode_rewards'] if r > 100) / len(self.monitoring_data['episode_rewards'])
            }
        }
        
        # レポートの保存
        report_path = self.output_dir / f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _check_learning_alerts(self, learning_metrics):
        """学習アラートのチェック"""
        alerts = []
        
        # 報酬の停滞アラート
        if learning_metrics['reward_trend'] < -0.1:
            alerts.append("REWARD_DECLINING: 報酬が減少傾向にあります")
        
        # 成功率の低下アラート
        if learning_metrics['success_rate_50'] < 0.1:
            alerts.append("LOW_SUCCESS_RATE: 成功率が低すぎます")
        
        # 行動の多様性低下アラート
        if 'action_diversity' in learning_metrics:
            action_diversity = learning_metrics['action_diversity']
            low_diversity_joints = [k for k, v in action_diversity.items() if 'range_std' in k and v < 0.01]
            if len(low_diversity_joints) > 4:
                alerts.append("LOW_ACTION_DIVERSITY: 行動の多様性が低下しています")
        
        # 安定性の悪化アラート
        if 'stability_evolution' in learning_metrics:
            stability_evolution = learning_metrics['stability_evolution']
            if stability_evolution.get('stability_trend', 0) < -0.01:
                alerts.append("STABILITY_DECLINING: 安定性が悪化しています")
        
        # アラートの出力
        if alerts:
            print("\n" + "="*50)
            print("学習アラート:")
            for alert in alerts:
                print(f"  ⚠️  {alert}")
            print("="*50 + "\n")
    
    def create_monitoring_visualizations(self):
        """監視可視化の作成"""
        if not self.monitoring_data['episode_rewards']:
            return
        
        # 1. 学習進捗の総合可視化
        self._plot_learning_progress()
        
        # 2. 行動の進化
        self._plot_action_evolution()
        
        # 3. 関節角度の進化
        self._plot_joint_evolution()
        
        # 4. 安定性の進化
        self._plot_stability_evolution()
    
    def _plot_learning_progress(self):
        """学習進捗の可視化"""
        episodes = list(range(len(self.monitoring_data['episode_rewards'])))
        rewards = list(self.monitoring_data['episode_rewards'])
        lengths = list(self.monitoring_data['episode_lengths'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 報酬の推移
        ax1.plot(episodes, rewards, 'b-', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward Progression')
        ax1.grid(True)
        
        # エピソード長の推移
        ax2.plot(episodes, lengths, 'g-', alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length')
        ax2.set_title('Episode Length Progression')
        ax2.grid(True)
        
        # 移動平均
        if len(rewards) > 10:
            window = min(50, len(rewards) // 4)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax3.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            ax3.plot(episodes, rewards, 'b-', alpha=0.3)
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward Moving Average')
            ax3.legend()
            ax3.grid(True)
        
        # 成功率の推移
        if len(rewards) > 20:
            success_rate_window = 20
            success_rates = []
            for i in range(success_rate_window, len(rewards) + 1):
                window_rewards = rewards[i-success_rate_window:i]
                success_rate = sum(1 for r in window_rewards if r > 100) / len(window_rewards)
                success_rates.append(success_rate)
            
            ax4.plot(episodes[success_rate_window-1:], success_rates, 'purple', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Success Rate')
            ax4.set_title(f'Success Rate (window={success_rate_window})')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_evolution(self):
        """行動の進化可視化"""
        if not self.monitoring_data['action_statistics']:
            return
        
        action_stats = list(self.monitoring_data['action_statistics'])
        joint_names = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 
                      'BL_hip', 'BL_knee', 'BR_hip', 'BR_knee']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for joint_idx in range(8):
            row, col = joint_idx // 4, joint_idx % 4
            
            # 各関節の行動範囲の変化
            ranges = [stats['range'][joint_idx] for stats in action_stats]
            means = [stats['mean'][joint_idx] for stats in action_stats]
            
            ax = axes[row, col]
            ax2 = ax.twinx()
            
            ax.plot(ranges, 'b-', label='Action Range', alpha=0.7)
            ax2.plot(means, 'r-', label='Action Mean', alpha=0.7)
            
            ax.set_title(f'{joint_names[joint_idx]} Action Evolution')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Range', color='b')
            ax2.set_ylabel('Mean', color='r')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'action_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_joint_evolution(self):
        """関節角度の進化可視化"""
        if not self.monitoring_data['joint_angle_statistics']:
            return
        
        joint_stats = list(self.monitoring_data['joint_angle_statistics'])
        joint_names = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 
                      'BL_hip', 'BL_knee', 'BR_hip', 'BR_knee']
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for joint_idx in range(8):
            row, col = joint_idx // 4, joint_idx % 4
            
            means = [stats['mean'][joint_idx] for stats in joint_stats]
            stds = [stats['std'][joint_idx] for stats in joint_stats]
            
            ax = axes[row, col]
            ax2 = ax.twinx()
            
            ax.plot(means, 'b-', label='Mean Angle', alpha=0.7)
            ax2.plot(stds, 'r-', label='Angle Std', alpha=0.7)
            
            ax.set_title(f'{joint_names[joint_idx]} Angle Evolution')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Angle (rad)', color='b')
            ax2.set_ylabel('Std Dev (rad)', color='r')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'joint_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_evolution(self):
        """安定性の進化可視化"""
        if not self.monitoring_data['stability_metrics']:
            return
        
        stability_stats = list(self.monitoring_data['stability_metrics'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 平均安定性
        avg_stabilities = [s['avg_stability'] for s in stability_stats]
        ax1.plot(avg_stabilities, 'b-', linewidth=2)
        ax1.set_title('Average Stability Score')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Stability Score')
        ax1.grid(True)
        
        # 最小安定性
        min_stabilities = [s['min_stability'] for s in stability_stats]
        ax2.plot(min_stabilities, 'r-', linewidth=2)
        ax2.set_title('Minimum Stability Score')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Min Stability Score')
        ax2.grid(True)
        
        # 最大ピッチ角度
        max_pitches = [s['max_pitch'] for s in stability_stats]
        ax3.plot(max_pitches, 'g-', linewidth=2)
        ax3.set_title('Maximum Pitch Angle')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Max Pitch (rad)')
        ax3.grid(True)
        
        # 最大ロール角度
        max_rolls = [s['max_roll'] for s in stability_stats]
        ax4.plot(max_rolls, 'orange', linewidth=2)
        ax4.set_title('Maximum Roll Angle')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Max Roll (rad)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Detailed Learning Monitor')
    parser.add_argument('--config', type=str, default='config/training_config_12h.yaml',
                       help='Training configuration file')
    parser.add_argument('--output-dir', type=str, default='data/detailed_monitoring',
                       help='Output directory for monitoring results')
    parser.add_argument('--monitor-frequency', type=int, default=10,
                       help='Monitoring frequency (episodes)')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # トレーナーの初期化
        trainer = ParallelTrainer(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # 監視器の初期化
        monitor = DetailedLearningMonitor(trainer, args.output_dir)
        monitor.monitor_frequency = args.monitor_frequency
        
        logger.info("詳細学習監視を開始します...")
        logger.info(f"監視頻度: {args.monitor_frequency}エピソードごと")
        logger.info(f"出力ディレクトリ: {args.output_dir}")
        
        # 学習の実行（監視付き）
        # 実際の学習ループでは、各エピソード後にmonitor.monitor_episode()を呼び出す
        
        logger.info("詳細学習監視が完了しました")
        
    except Exception as e:
        logger.error(f"監視中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
