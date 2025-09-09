"""
Training Pipeline for Bittle Walking DRL
Bittle四足歩行ロボット用の学習パイプライン
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
from collections import deque

from ..environment import BittleWalkingEnv
from ..models import PPOAgent
from .evaluator import Evaluator


class Trainer:
    """
    Bittle四足歩行ロボットの学習管理クラス
    
    PPOアルゴリズムを使用した学習プロセスの管理
    """
    
    def __init__(self, 
                 config_path: str = "config/training_config.yaml",
                 env_config_path: str = "config/env_config.yaml",
                 bittle_config_path: str = "config/bittle_config.yaml",
                 output_dir: str = "data/experiments",
                 experiment_name: Optional[str] = None):
        """
        学習器の初期化
        
        Args:
            config_path: 学習設定ファイルのパス
            env_config_path: 環境設定ファイルのパス
            bittle_config_path: Bittle設定ファイルのパス
            output_dir: 出力ディレクトリ
            experiment_name: 実験名（Noneの場合は自動生成）
        """
        # 設定の読み込み
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)
        self.bittle_config = self._load_config(bittle_config_path)
        
        # 出力ディレクトリの設定
        if experiment_name is None:
            experiment_name = f"bittle_walking_{int(time.time())}"
        
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ログの設定
        self._setup_logging()
        
        # 環境の作成
        self.env = BittleWalkingEnv(
            config_path=env_config_path,
            bittle_config_path=bittle_config_path,
            render=False  # 学習時は可視化しない
        )
        
        # エージェントの作成
        self.agent = PPOAgent(
            obs_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            config_path=config_path
        )
        
        # 評価器の作成
        self.evaluator = Evaluator(
            env_config_path=env_config_path,
            bittle_config_path=bittle_config_path,
            output_dir=self.output_dir
        )
        
        # 学習状態
        self.episode_count = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        
        # 統計情報
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.training_stats = []
        
        # 設定の保存
        self._save_configs()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """ログの設定"""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _save_configs(self):
        """設定ファイルの保存"""
        configs = {
            'training_config.yaml': self.config,
            'env_config.yaml': self.env_config,
            'bittle_config.yaml': self.bittle_config
        }
        
        for filename, config in configs.items():
            with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def train(self) -> Dict[str, Any]:
        """
        学習の実行
        
        Returns:
            学習結果の統計情報
        """
        self.logger.info("学習を開始します")
        self.logger.info(f"出力ディレクトリ: {self.output_dir}")
        
        # 学習パラメータ
        total_timesteps = self.config['training']['total_timesteps']
        eval_freq = self.config['training']['eval_freq']
        save_freq = self.config['training']['save_freq']
        log_interval = self.config['training']['log_interval']
        
        # 学習ループ
        while self.total_timesteps < total_timesteps:
            # エピソードの実行
            episode_stats = self._run_episode()
            
            # 統計の更新
            self._update_stats(episode_stats)
            
            # ログ出力
            if self.episode_count % log_interval == 0:
                self._log_progress()
            
            # 評価
            if self.episode_count % eval_freq == 0:
                self._evaluate()
            
            # モデルの保存
            if self.episode_count % save_freq == 0:
                self._save_model()
        
        # 最終評価
        self.logger.info("学習完了。最終評価を実行します")
        final_results = self._evaluate()
        
        # 学習結果の保存
        self._save_training_results()
        
        return final_results
    
    def _run_episode(self) -> Dict[str, Any]:
        """エピソードの実行"""
        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # 行動の取得
            action, log_prob, value = self.agent.get_action(obs)
            
            # 環境のステップ実行
            next_obs, reward, done, info = self.env.step(action)
            
            # 遷移の保存
            self.agent.store_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                log_prob=log_prob,
                value=value
            )
            
            # 統計の更新
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            obs = next_obs
        
        # エピソード統計
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': info.get('distance_to_target', float('inf')) < 0.1,
            'final_distance': info.get('distance_to_target', float('inf')),
            'episode_time': info.get('step', 0) * self.env.timestep
        }
        
        # エージェントの更新
        if self.agent.rollout_buffer.is_full():
            training_stats = self.agent.update()
            episode_stats.update(training_stats)
        
        self.episode_count += 1
        
        return episode_stats
    
    def _update_stats(self, episode_stats: Dict[str, Any]):
        """統計情報の更新"""
        self.episode_rewards.append(episode_stats['episode_reward'])
        self.episode_lengths.append(episode_stats['episode_length'])
        self.success_rates.append(episode_stats['success'])
        
        # 学習統計の保存
        if 'actor_loss' in episode_stats:
            self.training_stats.append({
                'episode': self.episode_count,
                'timestep': self.total_timesteps,
                'actor_loss': episode_stats.get('actor_loss', 0),
                'critic_loss': episode_stats.get('critic_loss', 0),
                'entropy_loss': episode_stats.get('entropy_loss', 0),
                'total_loss': episode_stats.get('total_loss', 0),
                'kl_divergence': episode_stats.get('kl_divergence', 0),
                'clip_fraction': episode_stats.get('clip_fraction', 0)
            })
    
    def _log_progress(self):
        """進捗のログ出力"""
        if len(self.episode_rewards) == 0:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_rates)
        
        elapsed_time = time.time() - self.start_time
        
        self.logger.info(
            f"Episode {self.episode_count} | "
            f"Timesteps {self.total_timesteps} | "
            f"Avg Reward {avg_reward:.2f} | "
            f"Avg Length {avg_length:.1f} | "
            f"Success Rate {success_rate:.2f} | "
            f"Time {elapsed_time:.1f}s"
        )
    
    def _evaluate(self) -> Dict[str, Any]:
        """モデルの評価"""
        self.logger.info(f"Episode {self.episode_count}: 評価を実行中...")
        
        eval_results = self.evaluator.evaluate(
            agent=self.agent,
            num_episodes=self.config['evaluation']['num_episodes'],
            render=False
        )
        
        # 評価結果のログ
        self.logger.info(
            f"評価結果 - "
            f"Success Rate: {eval_results['success_rate']:.2f} | "
            f"Avg Reward: {eval_results['avg_reward']:.2f} | "
            f"Avg Time: {eval_results['avg_time']:.2f}s | "
            f"Avg Distance: {eval_results['avg_distance']:.2f}m"
        )
        
        # 評価結果の保存
        eval_file = self.output_dir / f"evaluation_episode_{self.episode_count}.yaml"
        with open(eval_file, 'w', encoding='utf-8') as f:
            yaml.dump(eval_results, f, default_flow_style=False, allow_unicode=True)
        
        return eval_results
    
    def _save_model(self):
        """モデルの保存"""
        model_path = self.output_dir / f"model_episode_{self.episode_count}.pth"
        self.agent.save(str(model_path))
        self.logger.info(f"モデルを保存しました: {model_path}")
    
    def _save_training_results(self):
        """学習結果の保存"""
        # 学習曲線のプロット
        self._plot_training_curves()
        
        # 統計情報の保存
        results = {
            'total_episodes': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'training_time': time.time() - self.start_time,
            'final_avg_reward': np.mean(list(self.episode_rewards)[-100:]) if self.episode_rewards else 0,
            'final_success_rate': np.mean(list(self.success_rates)[-100:]) if self.success_rates else 0,
            'training_stats': self.training_stats
        }
        
        results_file = self.output_dir / "training_results.yaml"
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        self.logger.info(f"学習結果を保存しました: {results_file}")
    
    def _plot_training_curves(self):
        """学習曲線のプロット"""
        if not self.training_stats:
            return
        
        # データの準備
        episodes = [stat['episode'] for stat in self.training_stats]
        actor_losses = [stat['actor_loss'] for stat in self.training_stats]
        critic_losses = [stat['critic_loss'] for stat in self.training_stats]
        total_losses = [stat['total_loss'] for stat in self.training_stats]
        
        # プロットの作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失のプロット
        axes[0, 0].plot(episodes, actor_losses, label='Actor Loss')
        axes[0, 0].set_title('Actor Loss')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(episodes, critic_losses, label='Critic Loss')
        axes[0, 1].set_title('Critic Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(episodes, total_losses, label='Total Loss')
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # エピソード報酬のプロット
        if self.episode_rewards:
            reward_episodes = list(range(len(self.episode_rewards)))
            axes[1, 1].plot(reward_episodes, list(self.episode_rewards), alpha=0.3)
            
            # 移動平均
            window_size = min(100, len(self.episode_rewards))
            if window_size > 1:
                moving_avg = np.convolve(list(self.episode_rewards), 
                                       np.ones(window_size)/window_size, mode='valid')
                axes[1, 1].plot(range(window_size-1, len(self.episode_rewards)), 
                               moving_avg, label=f'Moving Avg ({window_size})')
            
            axes[1, 1].set_title('Episode Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("学習曲線を保存しました")
    
    def close(self):
        """リソースの解放"""
        self.env.close()
        self.evaluator.close()
