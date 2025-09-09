"""
Evaluation System for Bittle Walking DRL
Bittle四足歩行ロボット用の評価システム
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from ..environment import BittleWalkingEnv
from ..models import PPOAgent


class Evaluator:
    """
    Bittle四足歩行ロボットの評価システム
    
    学習済みモデルの性能評価と分析
    """
    
    def __init__(self, 
                 env_config_path: str = "config/env_config.yaml",
                 bittle_config_path: str = "config/bittle_config.yaml",
                 output_dir: Optional[Path] = None):
        """
        評価器の初期化
        
        Args:
            env_config_path: 環境設定ファイルのパス
            bittle_config_path: Bittle設定ファイルのパス
            output_dir: 出力ディレクトリ
        """
        # 設定の読み込み
        self.env_config = self._load_config(env_config_path)
        self.bittle_config = self._load_config(bittle_config_path)
        
        # 出力ディレクトリの設定
        self.output_dir = output_dir
        
        # ログの設定
        self.logger = logging.getLogger(__name__)
        
        # 評価環境の作成
        self.eval_env = BittleWalkingEnv(
            config_path=env_config_path,
            bittle_config_path=bittle_config_path,
            render=False  # 評価時は可視化しない（必要に応じて変更）
        )
        
        # 評価統計
        self.evaluation_history = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def evaluate(self, 
                 agent: PPOAgent,
                 num_episodes: int = 10,
                 render: bool = False,
                 deterministic: bool = True,
                 save_trajectories: bool = False) -> Dict[str, Any]:
        """
        モデルの評価
        
        Args:
            agent: 評価するエージェント
            num_episodes: 評価エピソード数
            render: 可視化の有無
            deterministic: 決定論的行動かどうか
            save_trajectories: 軌道の保存
            
        Returns:
            評価結果の辞書
        """
        self.logger.info(f"評価を開始します（エピソード数: {num_episodes}）")
        
        # 評価統計の初期化
        episode_rewards = []
        episode_lengths = []
        episode_times = []
        success_flags = []
        final_distances = []
        trajectories = []
        
        # 詳細統計
        detailed_stats = defaultdict(list)
        
        # 評価ループ
        for episode in range(num_episodes):
            episode_result = self._evaluate_episode(
                agent=agent,
                episode_num=episode,
                render=render,
                deterministic=deterministic,
                save_trajectory=save_trajectories
            )
            
            # 統計の収集
            episode_rewards.append(episode_result['total_reward'])
            episode_lengths.append(episode_result['episode_length'])
            episode_times.append(episode_result['episode_time'])
            success_flags.append(episode_result['success'])
            final_distances.append(episode_result['final_distance'])
            
            if save_trajectories:
                trajectories.append(episode_result['trajectory'])
            
            # 詳細統計の収集
            for key, value in episode_result['detailed_stats'].items():
                detailed_stats[key].append(value)
            
            self.logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"Reward={episode_result['total_reward']:.2f}, "
                f"Success={episode_result['success']}, "
                f"Distance={episode_result['final_distance']:.3f}m, "
                f"Time={episode_result['episode_time']:.2f}s"
            )
        
        # 評価結果の計算
        results = self._compute_evaluation_results(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episode_times=episode_times,
            success_flags=success_flags,
            final_distances=final_distances,
            detailed_stats=detailed_stats
        )
        
        # 軌道の保存
        if save_trajectories and self.output_dir:
            self._save_trajectories(trajectories)
        
        # 評価結果の保存
        if self.output_dir:
            self._save_evaluation_results(results)
            self._plot_evaluation_results(results, detailed_stats)
        
        # 評価履歴の更新
        self.evaluation_history.append(results)
        
        self.logger.info("評価完了")
        return results
    
    def _evaluate_episode(self, 
                         agent: PPOAgent,
                         episode_num: int,
                         render: bool = False,
                         deterministic: bool = True,
                         save_trajectory: bool = False) -> Dict[str, Any]:
        """
        単一エピソードの評価
        
        Args:
            agent: 評価するエージェント
            episode_num: エピソード番号
            render: 可視化の有無
            deterministic: 決定論的行動かどうか
            save_trajectory: 軌道の保存
            
        Returns:
            エピソード結果の辞書
        """
        obs = self.eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # 軌道の記録
        trajectory = [] if save_trajectory else None
        
        # 詳細統計
        detailed_stats = {
            'forward_rewards': [],
            'stability_rewards': [],
            'efficiency_rewards': [],
            'penalties': [],
            'positions': [],
            'orientations': [],
            'joint_angles': [],
            'joint_velocities': [],
            'actions': []
        }
        
        start_time = time.time()
        
        while not done:
            # 行動の取得
            action, _, _ = agent.get_action(obs, deterministic=deterministic)
            
            # 環境のステップ実行
            next_obs, reward, done, info = self.eval_env.step(action)
            
            # 統計の更新
            episode_reward += reward
            episode_length += 1
            
            # 軌道の記録
            if save_trajectory:
                trajectory.append({
                    'step': episode_length,
                    'observation': obs.copy(),
                    'action': action.copy(),
                    'reward': reward,
                    'next_observation': next_obs.copy(),
                    'done': done,
                    'info': info.copy()
                })
            
            # 詳細統計の記録
            if hasattr(self.eval_env, 'reward_function'):
                # 報酬の詳細分解（実装に応じて調整）
                detailed_stats['forward_rewards'].append(reward * 0.4)  # 仮の値
                detailed_stats['stability_rewards'].append(reward * 0.3)
                detailed_stats['efficiency_rewards'].append(reward * 0.2)
                detailed_stats['penalties'].append(reward * 0.1)
            
            # 状態情報の記録
            detailed_stats['positions'].append(info.get('position', [0, 0, 0]))
            detailed_stats['orientations'].append(info.get('orientation', [0, 0, 0]))
            detailed_stats['actions'].append(action.copy())
            
            obs = next_obs
        
        episode_time = time.time() - start_time
        
        # エピソード結果
        result = {
            'episode_num': episode_num,
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'episode_time': episode_time,
            'success': info.get('distance_to_target', float('inf')) < 0.1,
            'final_distance': info.get('distance_to_target', float('inf')),
            'detailed_stats': detailed_stats
        }
        
        if save_trajectory:
            result['trajectory'] = trajectory
        
        return result
    
    def _compute_evaluation_results(self, 
                                   episode_rewards: List[float],
                                   episode_lengths: List[int],
                                   episode_times: List[float],
                                   success_flags: List[bool],
                                   final_distances: List[float],
                                   detailed_stats: Dict[str, List]) -> Dict[str, Any]:
        """
        評価結果の計算
        
        Args:
            episode_rewards: エピソード報酬のリスト
            episode_lengths: エピソード長のリスト
            episode_times: エピソード時間のリスト
            success_flags: 成功フラグのリスト
            final_distances: 最終距離のリスト
            detailed_stats: 詳細統計
            
        Returns:
            評価結果の辞書
        """
        # 基本統計
        results = {
            'num_episodes': len(episode_rewards),
            'success_rate': np.mean(success_flags),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'avg_time': np.mean(episode_times),
            'std_time': np.std(episode_times),
            'avg_distance': np.mean(final_distances),
            'std_distance': np.std(final_distances),
            'min_distance': np.min(final_distances),
            'max_distance': np.max(final_distances)
        }
        
        # 成功率の詳細
        results['success_count'] = sum(success_flags)
        results['failure_count'] = len(success_flags) - sum(success_flags)
        
        # 詳細統計の計算
        for key, values in detailed_stats.items():
            if values:
                results[f'avg_{key}'] = np.mean(values)
                results[f'std_{key}'] = np.std(values)
        
        # 性能指標の計算
        results['efficiency'] = results['success_rate'] / results['avg_time'] if results['avg_time'] > 0 else 0
        results['energy_efficiency'] = results['avg_distance'] / results['avg_reward'] if results['avg_reward'] != 0 else 0
        
        return results
    
    def _save_trajectories(self, trajectories: List[List[Dict]]):
        """軌道の保存"""
        if not trajectories:
            return
        
        trajectories_file = self.output_dir / "evaluation_trajectories.yaml"
        
        # 軌道データの簡略化（メモリ効率のため）
        simplified_trajectories = []
        for i, trajectory in enumerate(trajectories):
            simplified_trajectory = []
            for step_data in trajectory[::10]:  # 10ステップごとにサンプリング
                simplified_trajectory.append({
                    'step': step_data['step'],
                    'position': step_data['info'].get('position', [0, 0, 0]),
                    'reward': step_data['reward'],
                    'done': step_data['done']
                })
            simplified_trajectories.append(simplified_trajectory)
        
        with open(trajectories_file, 'w', encoding='utf-8') as f:
            yaml.dump(simplified_trajectories, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"軌道データを保存しました: {trajectories_file}")
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """評価結果の保存"""
        results_file = self.output_dir / "evaluation_results.yaml"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"評価結果を保存しました: {results_file}")
    
    def _plot_evaluation_results(self, results: Dict[str, Any], detailed_stats: Dict[str, List]):
        """評価結果の可視化"""
        # プロットの作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. エピソード報酬の分布
        episode_rewards = [results['avg_reward']] * results['num_episodes']  # 簡略化
        axes[0, 0].hist(episode_rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Episode Rewards Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率の可視化
        success_data = [1 if i < results['success_count'] else 0 for i in range(results['num_episodes'])]
        axes[0, 1].bar(['Success', 'Failure'], [results['success_count'], results['failure_count']])
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylabel('Count')
        
        # 3. 距離の分布
        if 'avg_distance' in results:
            axes[0, 2].bar(['Avg Distance'], [results['avg_distance']])
            axes[0, 2].set_title('Average Distance to Target')
            axes[0, 2].set_ylabel('Distance (m)')
        
        # 4. 時間の分布
        if 'avg_time' in results:
            axes[1, 0].bar(['Avg Time'], [results['avg_time']])
            axes[1, 0].set_title('Average Episode Time')
            axes[1, 0].set_ylabel('Time (s)')
        
        # 5. 性能指標
        metrics = ['Success Rate', 'Efficiency', 'Energy Efficiency']
        values = [results['success_rate'], results.get('efficiency', 0), results.get('energy_efficiency', 0)]
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 統計サマリー
        axes[1, 2].axis('off')
        summary_text = f"""
        Evaluation Summary:
        
        Episodes: {results['num_episodes']}
        Success Rate: {results['success_rate']:.2%}
        Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}
        Avg Distance: {results['avg_distance']:.3f}m ± {results['std_distance']:.3f}m
        Avg Time: {results['avg_time']:.2f}s ± {results['std_time']:.2f}s
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("評価結果の可視化を保存しました")
    
    def compare_models(self, agents: List[PPOAgent], 
                      agent_names: List[str],
                      num_episodes: int = 10) -> Dict[str, Any]:
        """
        複数モデルの比較評価
        
        Args:
            agents: 比較するエージェントのリスト
            agent_names: エージェント名のリスト
            num_episodes: 評価エピソード数
            
        Returns:
            比較結果の辞書
        """
        self.logger.info(f"複数モデルの比較評価を開始します（{len(agents)}個のモデル）")
        
        comparison_results = {}
        
        for agent, name in zip(agents, agent_names):
            self.logger.info(f"モデル '{name}' を評価中...")
            
            results = self.evaluate(
                agent=agent,
                num_episodes=num_episodes,
                render=False,
                deterministic=True
            )
            
            comparison_results[name] = results
        
        # 比較結果の可視化
        if self.output_dir:
            self._plot_model_comparison(comparison_results)
        
        return comparison_results
    
    def _plot_model_comparison(self, comparison_results: Dict[str, Dict[str, Any]]):
        """モデル比較の可視化"""
        models = list(comparison_results.keys())
        
        # 比較指標
        metrics = ['success_rate', 'avg_reward', 'avg_distance', 'avg_time']
        metric_names = ['Success Rate', 'Average Reward', 'Average Distance', 'Average Time']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [comparison_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values)
            axes[i].set_title(f'{metric_name} Comparison')
            axes[i].set_ylabel(metric_name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # 値の表示
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("モデル比較結果を保存しました")
    
    def close(self):
        """リソースの解放"""
        self.eval_env.close()
