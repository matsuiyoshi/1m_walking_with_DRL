"""
並列トレーナーの実装
GPU並列処理を活用した高速学習
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from ..models.ppo_agent import PPOAgent
from ..environment.parallel_env import GPUParallelBittleEnv
from .trainer import Trainer


class ParallelTrainer(Trainer):
    """並列処理を活用したトレーナー"""
    
    def __init__(self, 
                 config_path: str = "config/training_config.yaml",
                 env_config_path: str = "config/env_config.yaml",
                 bittle_config_path: str = "config/bittle_config.yaml",
                 output_dir: str = "data/experiments"):
        """
        並列トレーナーの初期化
        
        Args:
            config_path: 学習設定ファイルのパス
            env_config_path: 環境設定ファイルのパス
            bittle_config_path: Bittle設定ファイルのパス
            output_dir: 出力ディレクトリ
        """
        super().__init__(config_path, env_config_path, bittle_config_path, output_dir)
        
        # 並列環境の初期化
        self.num_envs = self.config['training']['env']['num_envs']
        self.parallel_env = GPUParallelBittleEnv(
            num_envs=self.num_envs,
            env_config_path=env_config_path,
            bittle_config_path=bittle_config_path,
            device=self.agent.device
        )
        
        # 並列処理用のバッファ
        buffer_size = self.config.get('algorithm', {}).get('hyperparameters', {}).get('buffer_size', 2048)
        self.obs_dim = self.parallel_env.observation_space.shape[0]
        self.action_dim = self.parallel_env.action_space.shape[0]
        
        self.obs_buffer = torch.zeros(
            (buffer_size, self.num_envs, self.obs_dim),
            device=self.agent.device
        )
        self.action_buffer = torch.zeros(
            (buffer_size, self.num_envs, self.action_dim),
            device=self.agent.device
        )
        self.reward_buffer = torch.zeros(
            (buffer_size, self.num_envs),
            device=self.agent.device
        )
        self.value_buffer = torch.zeros(
            (buffer_size, self.num_envs),
            device=self.agent.device
        )
        self.log_prob_buffer = torch.zeros(
            (buffer_size, self.num_envs),
            device=self.agent.device
        )
        self.done_buffer = torch.zeros(
            (buffer_size, self.num_envs),
            device=self.agent.device
        )
        
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer_full = False
        
        # TensorBoardライターの初期化
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # TensorBoard用の統計情報
        self.tensorboard_step = 0
        self.episode_rewards_history = []
        self.episode_lengths_history = []
        
        # Early Stopping用の統計情報
        self.early_stopping_config = self.config.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_config.get('enabled', False)
        self.patience = self.early_stopping_config.get('patience', 100000)  # デフォルト: 100kステップ
        self.min_delta = self.early_stopping_config.get('min_delta', 0.01)  # デフォルト: 0.01
        self.monitor_metric = self.early_stopping_config.get('monitor_metric', 'avg_reward')  # 監視する指標
        self.restore_best_weights = self.early_stopping_config.get('restore_best_weights', True)
        
        # Early Stopping用の状態変数
        self.best_metric_value = float('-inf')
        self.wait_count = 0
        self.best_weights = None
        self.early_stop_triggered = False
        
        # 過学習検出用の設定
        self.overfitting_config = self.early_stopping_config.get('overfitting_detection', {})
        self.overfitting_enabled = self.overfitting_config.get('enabled', False)
        self.policy_loss_threshold = self.overfitting_config.get('policy_loss_threshold', 200.0)
        self.value_loss_threshold = self.overfitting_config.get('value_loss_threshold', 2.0)
        self.reward_std_threshold = self.overfitting_config.get('reward_std_threshold', 0.5)
        
        # 過学習検出用の履歴
        self.recent_rewards = []
        self.recent_policy_losses = []
        self.recent_value_losses = []
        self.history_size = 100  # 直近100回の履歴を保持
        
        self.logger.info(f"並列トレーナーを初期化しました（環境数: {self.num_envs}）")
        self.logger.info(f"TensorBoardログディレクトリ: {self.tensorboard_dir}")
        self.logger.info(f"TensorBoard起動コマンド: tensorboard --logdir={self.tensorboard_dir}")
        
        if self.early_stopping_enabled:
            self.logger.info(f"Early Stopping有効: patience={self.patience}, min_delta={self.min_delta}, monitor={self.monitor_metric}")
        
        if self.overfitting_enabled:
            self.logger.info(f"過学習検出有効: policy_loss_threshold={self.policy_loss_threshold}, value_loss_threshold={self.value_loss_threshold}, reward_std_threshold={self.reward_std_threshold}")
    
    def _detect_overfitting(self, metrics: Dict[str, float]) -> bool:
        """
        過学習の検出
        
        Args:
            metrics: 現在の学習指標
            
        Returns:
            bool: 過学習が検出されたかどうか
        """
        if not self.overfitting_enabled:
            return False
        
        # 履歴を更新
        self.recent_rewards.append(metrics.get('avg_reward', 0.0))
        self.recent_policy_losses.append(metrics.get('policy_loss', 0.0))
        self.recent_value_losses.append(metrics.get('value_loss', 0.0))
        
        # 履歴サイズを制限
        if len(self.recent_rewards) > self.history_size:
            self.recent_rewards.pop(0)
            self.recent_policy_losses.pop(0)
            self.recent_value_losses.pop(0)
        
        # 十分な履歴がない場合は判定しない
        if len(self.recent_rewards) < 20:
            return False
        
        # 過学習の判定
        overfitting_detected = False
        
        # Policy Lossの異常な上昇
        if len(self.recent_policy_losses) >= 10:
            recent_policy_loss = np.mean(self.recent_policy_losses[-10:])
            if recent_policy_loss > self.policy_loss_threshold:
                self.logger.warning(f"過学習検出: Policy Lossが異常に高い ({recent_policy_loss:.2f} > {self.policy_loss_threshold})")
                overfitting_detected = True
        
        # Value Lossの異常な上昇
        if len(self.recent_value_losses) >= 10:
            recent_value_loss = np.mean(self.recent_value_losses[-10:])
            if recent_value_loss > self.value_loss_threshold:
                self.logger.warning(f"過学習検出: Value Lossが異常に高い ({recent_value_loss:.2f} > {self.value_loss_threshold})")
                overfitting_detected = True
        
        # 報酬の分散の異常な増加
        if len(self.recent_rewards) >= 20:
            reward_std = np.std(self.recent_rewards[-20:])
            if reward_std > self.reward_std_threshold:
                self.logger.warning(f"過学習検出: 報酬の分散が異常に大きい ({reward_std:.3f} > {self.reward_std_threshold})")
                overfitting_detected = True
        
        return overfitting_detected
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """
        Early Stoppingの判定
        
        Args:
            metrics: 現在の学習指標
            
        Returns:
            bool: Early Stoppingがトリガーされたかどうか
        """
        if not self.early_stopping_enabled:
            return False
        
        current_metric = metrics.get(self.monitor_metric, 0.0)
        
        # 改善の判定
        if current_metric > self.best_metric_value + self.min_delta:
            self.best_metric_value = current_metric
            self.wait_count = 0
            
            # 最良の重みを保存
            if self.restore_best_weights:
                self.best_weights = {
                    'actor': self.agent.network.actor.state_dict().copy(),
                    'critic': self.agent.network.critic.state_dict().copy()
                }
            
            self.logger.info(f"Early Stopping: 新しい最良値 {self.monitor_metric}={current_metric:.4f}")
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                self.early_stop_triggered = True
                self.logger.warning(f"Early Stopping: {self.patience}ステップ改善なし。学習を停止します。")
                self.logger.warning(f"最良値: {self.monitor_metric}={self.best_metric_value:.4f}")
                
                # 最良の重みを復元
                if self.restore_best_weights and self.best_weights is not None:
                    self.agent.network.actor.load_state_dict(self.best_weights['actor'])
                    self.agent.network.critic.load_state_dict(self.best_weights['critic'])
                    self.logger.info("最良の重みを復元しました")
                
                return True
        
        return False
    
    def _get_action_batch(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """並列環境用のバッチ行動取得"""
        with torch.no_grad():
            actions, log_probs, values = self.agent.network.get_action_and_value(obs, deterministic=False)
            # 形状を確実に調整
            if log_probs.dim() > 1:
                log_probs = log_probs.squeeze(-1)
            if values.dim() > 1:
                values = values.squeeze(-1)
            return actions, log_probs, values
    
    def _collect_experience(self) -> Dict[str, Any]:
        """並列環境で経験を収集"""
        start_time = time.time()
        
        # 初期観測を取得
        obs = self.parallel_env.reset()
        
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
        while total_steps < self.buffer_size:
            # 行動を取得
            with torch.no_grad():
                actions, log_probs, values = self._get_action_batch(obs)
            
            # 環境でステップ実行
            next_obs, rewards, dones, infos = self.parallel_env.step(actions)
            
            # バッファに保存（テンソル形状を調整）
            self.obs_buffer[self.buffer_idx] = obs
            self.action_buffer[self.buffer_idx] = actions
            self.reward_buffer[self.buffer_idx] = rewards
            # valuesの形状を調整（[batch_size, 1] -> [batch_size]）
            if values.dim() > 1:
                values = values.squeeze(-1)
            self.value_buffer[self.buffer_idx] = values
            # log_probsの形状を調整（[batch_size, 1] -> [batch_size]）
            if log_probs.dim() > 1:
                log_probs = log_probs.squeeze(-1)
            self.log_prob_buffer[self.buffer_idx] = log_probs
            self.done_buffer[self.buffer_idx] = dones
            
            # エピソード統計を更新
            for i, (reward, done, info) in enumerate(zip(rewards, dones, infos)):
                if done:
                    episode_rewards.append(reward.item())
                    episode_lengths.append(info.get('episode_length', 0))
            
            obs = next_obs
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
            total_steps += self.num_envs
            
            if self.buffer_idx == 0:
                self.buffer_full = True
        
        collection_time = time.time() - start_time
        
        # TensorBoardにログ
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            self.writer.add_scalar('Training/Average_Reward', avg_reward, self.tensorboard_step)
            self.writer.add_scalar('Training/Max_Reward', max_reward, self.tensorboard_step)
            self.writer.add_scalar('Training/Min_Reward', min_reward, self.tensorboard_step)
            self.writer.add_scalar('Training/Reward_Std', std_reward, self.tensorboard_step)
            self.writer.add_scalar('Training/Episode_Count', len(episode_rewards), self.tensorboard_step)
            
            # 報酬のヒストグラム
            self.writer.add_histogram('Training/Reward_Distribution', np.array(episode_rewards), self.tensorboard_step)
            
            # 履歴に追加
            self.episode_rewards_history.extend(episode_rewards)
            
        if episode_lengths:
            avg_length = np.mean(episode_lengths)
            max_length = np.max(episode_lengths)
            min_length = np.min(episode_lengths)
            
            self.writer.add_scalar('Training/Average_Length', avg_length, self.tensorboard_step)
            self.writer.add_scalar('Training/Max_Length', max_length, self.tensorboard_step)
            self.writer.add_scalar('Training/Min_Length', min_length, self.tensorboard_step)
            
            # 履歴に追加
            self.episode_lengths_history.extend(episode_lengths)
        
        self.writer.add_scalar('Training/Total_Steps', total_steps, self.tensorboard_step)
        self.writer.add_scalar('Training/Collection_Time', collection_time, self.tensorboard_step)
        self.writer.add_scalar('Training/Steps_Per_Second', total_steps / collection_time, self.tensorboard_step)
        self.writer.add_scalar('Training/Buffer_Index', self.buffer_idx, self.tensorboard_step)
        self.writer.add_scalar('Training/Parallel_Envs', self.num_envs, self.tensorboard_step)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': total_steps,
            'collection_time': collection_time,
            'steps_per_second': total_steps / collection_time
        }
    
    def _compute_advantages(self) -> torch.Tensor:
        """並列環境用のアドバンテージ計算"""
        advantages = torch.zeros_like(self.reward_buffer)
        
        # 最後の値の推定
        with torch.no_grad():
            last_obs = self.obs_buffer[-1]  # 形状: (num_envs, obs_dim)
            # ネットワークに適切な形状で渡す
            _, _, last_values = self.agent.network.get_action_and_value(last_obs, deterministic=False)
            # 値の形状を確実に(num_envs,)にする
            if last_values.dim() > 1:
                last_values = last_values.squeeze(-1)  # 最後の次元を削除
            elif last_values.dim() == 0:
                last_values = last_values.unsqueeze(0).repeat(self.num_envs)  # スカラーの場合
        
        # 逆順でアドバンテージを計算
        next_value = last_values
        next_advantage = torch.zeros(self.num_envs, device=self.agent.device)
        
        for t in reversed(range(self.buffer_size)):
            if self.buffer_full or t < self.buffer_idx:
                # 終了フラグの処理
                mask = 1.0 - self.done_buffer[t].float()
                
                # 報酬と値の計算（形状を明示的に調整）
                gamma = self.config['algorithm']['hyperparameters']['gamma']
                gae_lambda = self.config['algorithm']['hyperparameters']['gae_lambda']
                
                delta = self.reward_buffer[t] + gamma * next_value * mask - self.value_buffer[t]
                next_advantage = delta + gamma * gae_lambda * next_advantage * mask
                
                # 形状を確実に(num_envs,)にする
                if next_advantage.dim() > 1:
                    next_advantage = next_advantage.squeeze(-1)
                elif next_advantage.dim() == 0:
                    next_advantage = next_advantage.unsqueeze(0).repeat(self.num_envs)
                
                advantages[t] = next_advantage
                next_value = self.value_buffer[t]
                
                # next_valueの形状も調整
                if next_value.dim() > 1:
                    next_value = next_value.squeeze(-1)
                elif next_value.dim() == 0:
                    next_value = next_value.unsqueeze(0).repeat(self.num_envs)
        
        return advantages
    
    def _update_policy(self, advantages: torch.Tensor) -> Dict[str, float]:
        """並列環境用のポリシー更新"""
        # バッファのデータを平坦化
        obs_flat = self.obs_buffer.view(-1, self.obs_dim)
        action_flat = self.action_buffer.view(-1, self.action_dim)
        old_log_prob_flat = self.log_prob_buffer.view(-1)
        value_flat = self.value_buffer.view(-1)
        advantage_flat = advantages.view(-1)
        
        # 正規化
        advantage_flat = (advantage_flat - advantage_flat.mean()) / (advantage_flat.std() + 1e-8)
        
        # エポックごとに更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.config['algorithm']['hyperparameters']['n_epochs']):
            # バッチサイズで分割
            batch_size = self.config['algorithm']['hyperparameters']['batch_size']
            num_batches = len(obs_flat) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_obs = obs_flat[start_idx:end_idx]
                batch_action = action_flat[start_idx:end_idx]
                batch_old_log_prob = old_log_prob_flat[start_idx:end_idx]
                batch_value = value_flat[start_idx:end_idx]
                batch_advantage = advantage_flat[start_idx:end_idx]
                
                # ポリシー更新（勾配計算用）
                actions, log_probs, values = self.agent.network.get_action_and_value(batch_obs, deterministic=False)
                # 形状を調整（[batch_size, 1] -> [batch_size]）
                new_log_prob = log_probs.squeeze(-1) if log_probs.dim() > 1 else log_probs
                new_value = values.squeeze(-1) if values.dim() > 1 else values
                # エントロピーは対数確率の負の値として近似
                entropy = -new_log_prob
                
                # 比率の計算
                ratio = torch.exp(new_log_prob - batch_old_log_prob)
                
                # クリッピング
                clipped_ratio = torch.clamp(ratio, 1 - self.config['algorithm']['hyperparameters']['clip_range'], 
                                          1 + self.config['algorithm']['hyperparameters']['clip_range'])
                
                # ポリシー損失
                policy_loss = -torch.min(ratio * batch_advantage, clipped_ratio * batch_advantage).mean()
                
                # 値関数損失
                value_loss = nn.MSELoss()(new_value, batch_value + batch_advantage)
                
                # エントロピー損失
                entropy_loss = -entropy.mean()
                
                # 総損失
                total_loss = (policy_loss + 
                            self.config['algorithm']['hyperparameters']['vf_coef'] * value_loss + 
                            self.config['algorithm']['hyperparameters']['ent_coef'] * entropy_loss)
                
                # 勾配更新
                self.agent.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), 
                                             self.config['algorithm']['hyperparameters']['max_grad_norm'])
                self.agent.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        avg_policy_loss = total_policy_loss / (self.config['algorithm']['hyperparameters']['n_epochs'] * num_batches)
        avg_value_loss = total_value_loss / (self.config['algorithm']['hyperparameters']['n_epochs'] * num_batches)
        avg_entropy_loss = total_entropy_loss / (self.config['algorithm']['hyperparameters']['n_epochs'] * num_batches)
        
        # TensorBoardにログ
        self.writer.add_scalar('Training/Policy_Loss', avg_policy_loss, self.tensorboard_step)
        self.writer.add_scalar('Training/Value_Loss', avg_value_loss, self.tensorboard_step)
        self.writer.add_scalar('Training/Entropy_Loss', avg_entropy_loss, self.tensorboard_step)
        self.writer.add_scalar('Training/Total_Loss', avg_policy_loss + avg_value_loss + avg_entropy_loss, self.tensorboard_step)
        
        # 学習率のログ
        current_lr = self.agent.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/Learning_Rate', current_lr, self.tensorboard_step)
        
        # 勾配ノルムのログ
        total_grad_norm = 0
        for param in self.agent.network.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.writer.add_scalar('Training/Gradient_Norm', total_grad_norm, self.tensorboard_step)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss
        }
    
    def train(self) -> Dict[str, Any]:
        """並列学習の実行"""
        self.logger.info("並列学習を開始します...")
        
        start_time = time.time()
        total_timesteps = self.config['training']['total_timesteps']
        eval_freq = self.config['training']['eval_freq']
        save_freq = self.config['training']['save_freq']
        log_interval = self.config['training']['log_interval']
        
        self.total_timesteps = 0
        self.episode_count = 0
        
        while self.total_timesteps < total_timesteps:
            # 経験収集
            collection_stats = self._collect_experience()
            
            # アドバンテージ計算
            advantages = self._compute_advantages()
            
            # ポリシー更新
            update_stats = self._update_policy(advantages)
            
            # 統計更新
            self.total_timesteps += collection_stats['total_steps']
            self.episode_count += len(collection_stats['episode_rewards'])
            self.tensorboard_step += 1
            
            # ログ出力（並列環境対応）
            if self.total_timesteps // log_interval > (self.total_timesteps - collection_stats['total_steps']) // log_interval:
                avg_reward = np.mean(collection_stats['episode_rewards']) if collection_stats['episode_rewards'] else 0
                avg_length = np.mean(collection_stats['episode_lengths']) if collection_stats['episode_lengths'] else 0
                
                self.logger.info(
                    f"Timesteps: {self.total_timesteps}/{total_timesteps}, "
                    f"Episodes: {self.episode_count}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Avg Length: {avg_length:.1f}, "
                    f"Steps/sec: {collection_stats['steps_per_second']:.1f}, "
                    f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                    f"Value Loss: {update_stats['value_loss']:.4f}"
                )
                
                # Early Stoppingと過学習検出のチェック
                if self.early_stopping_enabled or self.overfitting_enabled:
                    metrics = {
                        'avg_reward': avg_reward,
                        'avg_length': avg_length,
                        'policy_loss': update_stats['policy_loss'],
                        'value_loss': update_stats['value_loss']
                    }
                    
                    # 過学習検出
                    if self._detect_overfitting(metrics):
                        self.logger.warning("過学習が検出されました。学習を停止します。")
                        break
                    
                    # Early Stoppingのチェック
                    if self.early_stopping_enabled and self._check_early_stopping(metrics):
                        self.logger.warning("Early Stoppingがトリガーされました。学習を停止します。")
                        break
            
            # 評価
            if eval_freq > 0 and self.total_timesteps % eval_freq == 0:
                self.logger.info("評価を実行します...")
                eval_results = self._evaluate()
                self.logger.info(f"評価結果: {eval_results}")
            
            # モデル保存
            if save_freq > 0 and self.total_timesteps % save_freq == 0:
                self._save_model()
        
        # 最終評価と保存
        self.logger.info("学習完了。最終評価を実行します")
        final_results = self._evaluate()
        
        # 最終モデルの保存
        final_model_path = self.output_dir / "final_model.pth"
        self.agent.save(str(final_model_path))
        self.logger.info(f"最終モデルを保存しました: {final_model_path}")
        
        # 学習結果の保存
        self._save_training_results()
        
        # 並列環境を閉じる
        self.parallel_env.close()
        
        # TensorBoardライターを閉じる
        self.writer.close()
        
        total_time = time.time() - start_time
        self.logger.info(f"並列学習完了。総時間: {total_time:.1f}秒")
        self.logger.info(f"TensorBoardログ: {self.tensorboard_dir}")
        
        return final_results
