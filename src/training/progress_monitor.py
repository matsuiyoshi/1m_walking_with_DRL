#!/usr/bin/env python3
"""
進捗監視クラス
Progress Monitor for Curriculum Learning
"""

import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class ProgressMonitor:
    """進捗監視クラス"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        初期化
        
        Args:
            config: 監視設定
            output_dir: 出力ディレクトリ
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 監視設定
        self.check_interval = config.get('stage_progress', {}).get('check_interval', 300000)
        self.log_interval = config.get('stage_progress', {}).get('log_interval', 10000)
        self.window_size = config.get('success_monitoring', {}).get('window_size', 100)
        self.min_episodes = config.get('success_monitoring', {}).get('min_episodes', 50)
        
        # 監視データ
        self.episode_data = []
        self.stage_start_time = time.time()
        self.last_check_time = time.time()
        self.last_log_time = time.time()
        
    def update_episode(self, episode_length: float, reward: float, forward_distance: float, 
                      success: bool, timestep: int):
        """
        エピソードデータを更新
        
        Args:
            episode_length: エピソード長
            reward: 報酬
            forward_distance: 前進距離
            success: 成功かどうか
            timestep: 現在のタイムステップ
        """
        episode_info = {
            'timestep': timestep,
            'episode_length': episode_length,
            'reward': reward,
            'forward_distance': forward_distance,
            'success': success,
            'timestamp': time.time()
        }
        
        self.episode_data.append(episode_info)
        
        # ウィンドウサイズを超えた場合は古いデータを削除
        if len(self.episode_data) > self.window_size * 2:
            self.episode_data = self.episode_data[-self.window_size:]
            
    def should_check_progress(self, timestep: int) -> bool:
        """進捗チェックが必要かどうか"""
        return timestep % self.check_interval == 0
        
    def should_log_progress(self, timestep: int) -> bool:
        """ログ出力が必要かどうか"""
        return timestep % self.log_interval == 0
        
    def get_recent_stats(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        最近の統計を取得
        
        Args:
            window_size: ウィンドウサイズ（Noneの場合は設定値を使用）
            
        Returns:
            統計情報
        """
        if window_size is None:
            window_size = self.window_size
            
        if len(self.episode_data) < self.min_episodes:
            return {
                'episode_count': len(self.episode_data),
                'avg_episode_length': 0.0,
                'avg_reward': 0.0,
                'avg_forward_distance': 0.0,
                'success_rate': 0.0,
                'improvement_rate': 0.0
            }
            
        # 最近のデータを取得
        recent_data = self.episode_data[-window_size:]
        
        # 統計を計算
        episode_lengths = [d['episode_length'] for d in recent_data]
        rewards = [d['reward'] for d in recent_data]
        forward_distances = [d['forward_distance'] for d in recent_data]
        successes = [d['success'] for d in recent_data]
        
        # 改善率を計算（前半と後半の比較）
        if len(recent_data) >= 20:
            first_half = recent_data[:len(recent_data)//2]
            second_half = recent_data[len(recent_data)//2:]
            
            first_avg_reward = np.mean([d['reward'] for d in first_half])
            second_avg_reward = np.mean([d['reward'] for d in second_half])
            
            if first_avg_reward > 0:
                improvement_rate = (second_avg_reward - first_avg_reward) / first_avg_reward
            else:
                improvement_rate = 0.0
        else:
            improvement_rate = 0.0
            
        return {
            'episode_count': len(recent_data),
            'avg_episode_length': np.mean(episode_lengths),
            'avg_reward': np.mean(rewards),
            'avg_forward_distance': np.mean(forward_distances),
            'success_rate': np.mean(successes),
            'improvement_rate': improvement_rate,
            'std_episode_length': np.std(episode_lengths),
            'std_reward': np.std(rewards),
            'std_forward_distance': np.std(forward_distances)
        }
        
    def check_stage_completion_criteria(self, criteria: Dict[str, Any]) -> Dict[str, bool]:
        """
        段階完了条件をチェック
        
        Args:
            criteria: 完了条件
            
        Returns:
            各条件の達成状況
        """
        stats = self.get_recent_stats()
        results = {}
        
        if 'min_episode_length' in criteria:
            results['episode_length'] = stats['avg_episode_length'] >= criteria['min_episode_length']
            
        if 'min_stability_reward' in criteria:
            results['stability_reward'] = stats['avg_reward'] >= criteria['min_stability_reward']
            
        if 'min_forward_distance' in criteria:
            results['forward_distance'] = stats['avg_forward_distance'] >= criteria['min_forward_distance']
            
        if 'success_rate' in criteria:
            results['success_rate'] = stats['success_rate'] >= criteria['success_rate']
            
        return results
        
    def log_progress(self, stage: int, timestep: int):
        """進捗をログ出力"""
        stats = self.get_recent_stats()
        elapsed_time = time.time() - self.stage_start_time
        
        self.logger.info(f"=== 段階 {stage} 進捗レポート ===")
        self.logger.info(f"経過時間: {elapsed_time:.1f}秒")
        self.logger.info(f"タイムステップ: {timestep}")
        self.logger.info(f"エピソード数: {stats['episode_count']}")
        self.logger.info(f"平均エピソード長: {stats['avg_episode_length']:.2f}秒")
        self.logger.info(f"平均報酬: {stats['avg_reward']:.2f}")
        self.logger.info(f"平均前進距離: {stats['avg_forward_distance']:.2f}m")
        self.logger.info(f"成功率: {stats['success_rate']:.2%}")
        self.logger.info(f"改善率: {stats['improvement_rate']:.2%}")
        
    def save_progress_report(self, stage: int, timestep: int):
        """進捗レポートを保存"""
        stats = self.get_recent_stats()
        elapsed_time = time.time() - self.stage_start_time
        
        report = {
            'stage': stage,
            'timestep': timestep,
            'elapsed_time': elapsed_time,
            'episode_count': stats['episode_count'],
            'avg_episode_length': stats['avg_episode_length'],
            'avg_reward': stats['avg_reward'],
            'avg_forward_distance': stats['avg_forward_distance'],
            'success_rate': stats['success_rate'],
            'improvement_rate': stats['improvement_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
        report_path = self.output_dir / f"stage_{stage}_progress_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
            
        self.logger.info(f"進捗レポートを保存しました: {report_path}")
        
    def reset_for_new_stage(self, stage: int):
        """新しい段階のためにリセット"""
        self.episode_data = []
        self.stage_start_time = time.time()
        self.last_check_time = time.time()
        self.last_log_time = time.time()
        
        self.logger.info(f"段階 {stage} の監視を開始しました")
        
    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """学習曲線データを取得"""
        if len(self.episode_data) < 10:
            return {
                'timesteps': [],
                'rewards': [],
                'episode_lengths': [],
                'forward_distances': []
            }
            
        # データを時系列で整理
        timesteps = [d['timestep'] for d in self.episode_data]
        rewards = [d['reward'] for d in self.episode_data]
        episode_lengths = [d['episode_length'] for d in self.episode_data]
        forward_distances = [d['forward_distance'] for d in self.episode_data]
        
        return {
            'timesteps': timesteps,
            'rewards': rewards,
            'episode_lengths': episode_lengths,
            'forward_distances': forward_distances
        }
        
    def detect_learning_plateau(self, window_size: int = 50, threshold: float = 0.01) -> bool:
        """
        学習の停滞を検出
        
        Args:
            window_size: ウィンドウサイズ
            threshold: 改善閾値
            
        Returns:
            停滞しているかどうか
        """
        if len(self.episode_data) < window_size * 2:
            return False
            
        # 前半と後半の比較
        first_half = self.episode_data[-window_size*2:-window_size]
        second_half = self.episode_data[-window_size:]
        
        first_avg_reward = np.mean([d['reward'] for d in first_half])
        second_avg_reward = np.mean([d['reward'] for d in second_half])
        
        improvement = (second_avg_reward - first_avg_reward) / max(abs(first_avg_reward), 1e-6)
        
        return improvement < threshold
