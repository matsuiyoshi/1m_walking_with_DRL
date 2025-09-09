"""
PPO Agent Implementation
Proximal Policy Optimization エージェントの実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
from collections import deque
import logging

from .networks import PPONetwork, ActorNetwork, CriticNetwork, NetworkFactory


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) エージェント
    
    連続制御タスク用のPPOアルゴリズム実装
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 config_path: str = "config/training_config.yaml",
                 device: str = "auto"):
        """
        PPO Agentの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            config_path: 設定ファイルのパス
            device: 使用デバイス ("auto", "cpu", "cuda")
        """
        # 設定の読み込み
        self.config = self._load_config(config_path)
        self.algorithm_config = self.config['algorithm']
        self.hyperparams = self.algorithm_config['hyperparameters']
        
        # デバイスの設定
        self.device = self._setup_device(device)
        
        # ネットワークの作成
        self.network = NetworkFactory.create_ppo_network(
            obs_dim, action_dim, self.algorithm_config['network']
        ).to(self.device)
        
        # オプティマイザーの設定
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.hyperparams['learning_rate']
        )
        
        # 学習状態
        self.learning_steps = 0
        self.episode_count = 0
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        # バッファ
        self.rollout_buffer = RolloutBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            buffer_size=self.hyperparams.get('buffer_size', 2048),
            device=self.device
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self, device: str) -> torch.device:
        """デバイスの設定"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        行動の取得
        
        Args:
            obs: 観測
            deterministic: 決定論的行動かどうか
            
        Returns:
            (action, log_prob, value): 行動、対数確率、状態価値
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action_and_value(
                obs_tensor, deterministic
            )
            
            return (
                action.cpu().numpy().flatten(),
                log_prob.cpu().item(),
                value.cpu().item()
            )
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, 
                        reward: float, next_obs: np.ndarray, 
                        done: bool, log_prob: float, value: float):
        """
        遷移の保存
        
        Args:
            obs: 現在の観測
            action: 行動
            reward: 報酬
            next_obs: 次の観測
            done: エピソード終了フラグ
            log_prob: 行動の対数確率
            value: 状態価値
        """
        self.rollout_buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            log_prob=log_prob,
            value=value
        )
    
    def update(self) -> Dict[str, float]:
        """
        ネットワークの更新
        
        Returns:
            学習統計情報
        """
        if not self.rollout_buffer.is_full():
            return {}
        
        # バッファからデータを取得
        batch = self.rollout_buffer.get_batch()
        
        # アドバンテージの計算
        advantages = self._compute_advantages(batch)
        returns = advantages + batch['values']
        
        # データの正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 学習統計
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0
        }
        
        # 複数エポックでの学習
        n_epochs = self.hyperparams['n_epochs']
        batch_size = self.hyperparams['batch_size']
        
        for epoch in range(n_epochs):
            # バッチの作成
            indices = torch.randperm(len(batch['observations']))
            
            for start_idx in range(0, len(batch['observations']), batch_size):
                end_idx = min(start_idx + batch_size, len(batch['observations']))
                batch_indices = indices[start_idx:end_idx]
                
                # ミニバッチの取得
                mini_batch = {
                    key: value[batch_indices] for key, value in batch.items()
                }
                mini_advantages = advantages[batch_indices]
                mini_returns = returns[batch_indices]
                
                # 損失の計算と更新
                epoch_stats = self._update_network(mini_batch, mini_advantages, mini_returns)
                
                # 統計の更新
                for key, value in epoch_stats.items():
                    stats[key] += value
        
        # 統計の平均化
        for key in stats:
            stats[key] /= (n_epochs * (len(batch['observations']) // batch_size + 1))
        
        # バッファのクリア
        self.rollout_buffer.clear()
        
        self.learning_steps += 1
        
        return stats
    
    def _compute_advantages(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        アドバンテージの計算（GAE）
        
        Args:
            batch: バッチデータ
            
        Returns:
            アドバンテージ
        """
        gamma = self.hyperparams['gamma']
        gae_lambda = self.hyperparams['gae_lambda']
        
        advantages = torch.zeros_like(batch['rewards'])
        last_advantage = 0
        
        # 後ろから計算
        for t in reversed(range(len(batch['rewards']))):
            if t == len(batch['rewards']) - 1:
                next_value = 0 if batch['dones'][t] else batch['values'][t]
            else:
                next_value = batch['values'][t + 1]
            
            delta = batch['rewards'][t] + gamma * next_value * (1 - batch['dones'][t]) - batch['values'][t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - batch['dones'][t]) * last_advantage
        
        return advantages
    
    def _update_network(self, batch: Dict[str, torch.Tensor], 
                       advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """
        ネットワークの更新
        
        Args:
            batch: ミニバッチデータ
            advantages: アドバンテージ
            returns: リターン
            
        Returns:
            学習統計
        """
        # 現在のポリシーでの行動と価値の取得
        current_actions, current_log_probs, current_values = self.network.get_action_and_value(
            batch['observations']
        )
        
        # ポリシー比の計算
        ratio = torch.exp(current_log_probs - batch['log_probs'])
        
        # クリップされた目的関数
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.hyperparams['clip_range'], 
                           1 + self.hyperparams['clip_range']) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 価値関数の損失
        value_pred_clipped = batch['values'] + torch.clamp(
            current_values - batch['values'], 
            -self.hyperparams['clip_range'], 
            self.hyperparams['clip_range']
        )
        value_loss1 = (current_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        critic_loss = torch.max(value_loss1, value_loss2).mean()
        
        # エントロピー損失
        entropy_loss = -current_log_probs.mean()
        
        # 総損失
        total_loss = (actor_loss + 
                     self.hyperparams['vf_coef'] * critic_loss + 
                     self.hyperparams['ent_coef'] * entropy_loss)
        
        # 勾配クリッピング
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.hyperparams['max_grad_norm'])
        self.optimizer.step()
        
        # 統計の計算
        with torch.no_grad():
            kl_divergence = (batch['log_probs'] - current_log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.hyperparams['clip_range']).float().mean()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'clip_fraction': clip_fraction.item()
        }
    
    def save(self, filepath: str):
        """モデルの保存"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_steps': self.learning_steps,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """モデルの読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_steps = checkpoint['learning_steps']
        self.episode_count = checkpoint['episode_count']


class RolloutBuffer:
    """
    ロールアウトバッファ
    
    エピソードデータの保存と管理
    """
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: torch.device):
        """
        バッファの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            buffer_size: バッファサイズ
            device: デバイス
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # バッファの初期化
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool, log_prob: float, value: float):
        """
        データの追加
        
        Args:
            obs: 観測
            action: 行動
            reward: 報酬
            next_obs: 次の観測
            done: エピソード終了フラグ
            log_prob: 対数確率
            value: 状態価値
        """
        self.observations[self.ptr] = torch.FloatTensor(obs)
        self.actions[self.ptr] = torch.FloatTensor(action)
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = torch.FloatTensor(next_obs)
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        バッチデータの取得
        
        Returns:
            バッチデータ辞書
        """
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_observations': self.next_observations[:self.size],
            'dones': self.dones[:self.size],
            'log_probs': self.log_probs[:self.size],
            'values': self.values[:self.size]
        }
    
    def is_full(self) -> bool:
        """バッファが満杯かどうか"""
        return self.size == self.buffer_size
    
    def clear(self):
        """バッファのクリア"""
        self.ptr = 0
        self.size = 0
