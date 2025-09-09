"""
Neural Network Architectures for PPO
PPOアルゴリズム用のニューラルネットワーク構造
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class ActorNetwork(nn.Module):
    """
    PPO Actor Network (Policy Network)
    
    観測から行動（関節目標角度）を出力するネットワーク
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "ReLU",
                 output_activation: str = "Tanh",
                 dropout_rate: float = 0.0):
        """
        Actor Networkの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            hidden_dims: 隠れ層の次元数リスト
            activation: 隠れ層の活性化関数
            output_activation: 出力層の活性化関数
            dropout_rate: ドロップアウト率
        """
        super(ActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 活性化関数の設定
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation)
        
        # ネットワーク層の構築
        self.layers = self._build_layers()
        
        # 重みの初期化
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str):
        """活性化関数の取得"""
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _build_layers(self) -> nn.ModuleList:
        """ネットワーク層の構築"""
        layers = nn.ModuleList()
        
        # 入力層
        input_dim = self.obs_dim
        
        # 隠れ層
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            input_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(input_dim, self.action_dim))
        layers.append(self.output_activation)
        
        return layers
    
    def _initialize_weights(self):
        """重みの初期化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier初期化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            obs: 観測テンソル [batch_size, obs_dim]
            
        Returns:
            行動テンソル [batch_size, action_dim]
        """
        x = obs
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        行動の取得
        
        Args:
            obs: 観測テンソル
            deterministic: 決定論的行動かどうか
            
        Returns:
            (action, log_prob): 行動と対数確率
        """
        action_mean = self.forward(obs)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean)
        
        # 行動の分散（学習可能パラメータとして追加することも可能）
        action_std = torch.ones_like(action_mean) * 0.1
        
        # 正規分布からサンプリング
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    PPO Critic Network (Value Network)
    
    観測から状態価値を出力するネットワーク
    """
    
    def __init__(self, 
                 obs_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "ReLU",
                 dropout_rate: float = 0.0):
        """
        Critic Networkの初期化
        
        Args:
            obs_dim: 観測次元数
            hidden_dims: 隠れ層の次元数リスト
            activation: 活性化関数
            dropout_rate: ドロップアウト率
        """
        super(CriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 活性化関数の設定
        self.activation = self._get_activation(activation)
        
        # ネットワーク層の構築
        self.layers = self._build_layers()
        
        # 重みの初期化
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str):
        """活性化関数の取得"""
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _build_layers(self) -> nn.ModuleList:
        """ネットワーク層の構築"""
        layers = nn.ModuleList()
        
        # 入力層
        input_dim = self.obs_dim
        
        # 隠れ層
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            input_dim = hidden_dim
        
        # 出力層（状態価値）
        layers.append(nn.Linear(input_dim, 1))
        
        return layers
    
    def _initialize_weights(self):
        """重みの初期化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier初期化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            obs: 観測テンソル [batch_size, obs_dim]
            
        Returns:
            状態価値テンソル [batch_size, 1]
        """
        x = obs
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class PPONetwork(nn.Module):
    """
    PPO用の統合ネットワーク
    
    ActorとCriticを組み合わせたネットワーク
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 actor_hidden_dims: List[int] = [256, 256, 128],
                 critic_hidden_dims: List[int] = [256, 256, 128],
                 shared_layers: bool = False,
                 activation: str = "ReLU",
                 output_activation: str = "Tanh",
                 dropout_rate: float = 0.0):
        """
        PPO Networkの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            actor_hidden_dims: Actor隠れ層の次元数リスト
            critic_hidden_dims: Critic隠れ層の次元数リスト
            shared_layers: 共有層を使用するかどうか
            activation: 活性化関数
            output_activation: 出力層の活性化関数
            dropout_rate: ドロップアウト率
        """
        super(PPONetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.shared_layers = shared_layers
        
        if shared_layers:
            # 共有層を使用する場合
            self.shared_network = self._build_shared_network(
                obs_dim, actor_hidden_dims[0], activation, dropout_rate
            )
            
            # ActorとCriticの出力層
            self.actor_head = nn.Linear(actor_hidden_dims[0], action_dim)
            self.critic_head = nn.Linear(actor_hidden_dims[0], 1)
            
            # 出力活性化関数
            self.output_activation = self._get_activation(output_activation)
        else:
            # 独立したネットワーク
            self.actor = ActorNetwork(
                obs_dim, action_dim, actor_hidden_dims, 
                activation, output_activation, dropout_rate
            )
            self.critic = CriticNetwork(
                obs_dim, critic_hidden_dims, activation, dropout_rate
            )
    
    def _get_activation(self, activation_name: str):
        """活性化関数の取得"""
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(),
            "ELU": nn.ELU()
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _build_shared_network(self, input_dim: int, output_dim: int, 
                             activation: str, dropout_rate: float) -> nn.Module:
        """共有ネットワークの構築"""
        layers = []
        
        # 共有層
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(self._get_activation(activation))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            obs: 観測テンソル
            
        Returns:
            (action, value): 行動と状態価値
        """
        if self.shared_layers:
            shared_features = self.shared_network(obs)
            action = self.output_activation(self.actor_head(shared_features))
            value = self.critic_head(shared_features)
            return action, value
        else:
            action = self.actor(obs)
            value = self.critic(obs)
            return action, value
    
    def get_action_and_value(self, obs: torch.Tensor, 
                            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        行動と状態価値の取得
        
        Args:
            obs: 観測テンソル
            deterministic: 決定論的行動かどうか
            
        Returns:
            (action, log_prob, value): 行動、対数確率、状態価値
        """
        if self.shared_layers:
            shared_features = self.shared_network(obs)
            action_mean = self.output_activation(self.actor_head(shared_features))
            value = self.critic_head(shared_features)
        else:
            action_mean = self.actor(obs)
            value = self.critic(obs)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean), value
        
        # 行動の分散
        action_std = torch.ones_like(action_mean) * 0.1
        
        # 正規分布からサンプリング
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class NetworkFactory:
    """
    ネットワーク作成のファクトリクラス
    """
    
    @staticmethod
    def create_actor(obs_dim: int, action_dim: int, config: dict) -> ActorNetwork:
        """Actor Networkの作成"""
        return ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.get('actor', {}).get('hidden_layers', [256, 256, 128]),
            activation=config.get('actor', {}).get('activation', 'ReLU'),
            output_activation=config.get('actor', {}).get('output_activation', 'Tanh'),
            dropout_rate=config.get('dropout_rate', 0.0)
        )
    
    @staticmethod
    def create_critic(obs_dim: int, config: dict) -> CriticNetwork:
        """Critic Networkの作成"""
        return CriticNetwork(
            obs_dim=obs_dim,
            hidden_dims=config.get('critic', {}).get('hidden_layers', [256, 256, 128]),
            activation=config.get('critic', {}).get('activation', 'ReLU'),
            dropout_rate=config.get('dropout_rate', 0.0)
        )
    
    @staticmethod
    def create_ppo_network(obs_dim: int, action_dim: int, config: dict) -> PPONetwork:
        """PPO Networkの作成"""
        return PPONetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=config.get('actor', {}).get('hidden_layers', [256, 256, 128]),
            critic_hidden_dims=config.get('critic', {}).get('hidden_layers', [256, 256, 128]),
            shared_layers=config.get('shared_layers', False),
            activation=config.get('actor', {}).get('activation', 'ReLU'),
            output_activation=config.get('actor', {}).get('output_activation', 'Tanh'),
            dropout_rate=config.get('dropout_rate', 0.0)
        )
