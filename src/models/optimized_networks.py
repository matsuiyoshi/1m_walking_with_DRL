"""
Optimized Neural Network Architectures for PPO
最適化されたPPOアルゴリズム用のニューラルネットワーク構造
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import math


class ResidualBlock(nn.Module):
    """残差接続ブロック（安定性向上）"""
    
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = out + residual  # 残差接続
        out = self.layer_norm(out)  # レイヤー正規化
        return F.relu(out)


class AttentionModule(nn.Module):
    """アテンション機構（重要な特徴量に注目）"""
    
    def __init__(self, feature_dim: int, attention_dim: int = 64):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, feature_dim)
        attention_weights = self.attention(x)  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_features = x * attention_weights
        return weighted_features.sum(dim=1)  # (batch_size, feature_dim)


class OptimizedActorNetwork(nn.Module):
    """
    最適化されたActor Network
    
    改善点:
    - 残差接続による勾配流の改善
    - レイヤー正規化による学習安定性
    - ドロップアウト正則化による過学習防止
    - アテンション機構による重要特徴の強調
    - 適応的活性化関数
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "Swish",
                 output_activation: str = "Tanh",
                 dropout_rate: float = 0.1,
                 use_residual: bool = True,
                 use_attention: bool = False,
                 use_layer_norm: bool = True):
        """
        最適化されたActor Networkの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            hidden_dims: 隠れ層の次元数リスト
            activation: 隠れ層の活性化関数
            output_activation: 出力層の活性化関数
            dropout_rate: ドロップアウト率
            use_residual: 残差接続を使用するかどうか
            use_attention: アテンション機構を使用するかどうか
            use_layer_norm: レイヤー正規化を使用するかどうか
        """
        super(OptimizedActorNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        
        # 活性化関数の設定
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation)
        
        # ネットワーク層の構築
        self.network = self._build_network()
        
        # 重みの初期化
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str):
        """最適化された活性化関数の取得"""
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "LeakyReLU": nn.LeakyReLU(0.01),
            "ELU": nn.ELU(),
            "Swish": nn.SiLU(),  # Swish活性化関数（最適化）
            "GELU": nn.GELU(),   # GELU活性化関数（Transformer系で人気）
            "Mish": lambda x: x * torch.tanh(F.softplus(x))  # Mish活性化関数
        }
        return activations.get(activation_name, nn.SiLU())
    
    def _build_network(self) -> nn.ModuleDict:
        """最適化されたネットワーク構造の構築"""
        network = nn.ModuleDict()
        
        # 入力層
        input_dim = self.obs_dim
        network['input_layer'] = nn.Linear(input_dim, self.hidden_dims[0])
        if self.use_layer_norm:
            network['input_norm'] = nn.LayerNorm(self.hidden_dims[0])
        
        # 隠れ層（残差接続付き）
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:  # 最初の層はすでに作成済み
                network[f'hidden_{i}'] = nn.Linear(self.hidden_dims[i-1], hidden_dim)
                if self.use_layer_norm:
                    network[f'hidden_norm_{i}'] = nn.LayerNorm(hidden_dim)
            
            # 残差ブロック
            if self.use_residual and hidden_dim == self.hidden_dims[0]:
                network[f'residual_{i}'] = ResidualBlock(hidden_dim, self.dropout_rate)
            
            # ドロップアウト
            if self.dropout_rate > 0:
                network[f'dropout_{i}'] = nn.Dropout(self.dropout_rate)
        
        # アテンション機構
        if self.use_attention:
            network['attention'] = AttentionModule(self.hidden_dims[-1])
        
        # 出力層
        network['output_layer'] = nn.Linear(self.hidden_dims[-1], self.action_dim)
        
        return network
    
    def _initialize_weights(self):
        """最適化された重み初期化"""
        for name, module in self.network.items():
            if isinstance(module, nn.Linear):
                # He初期化（ReLU系活性化関数に最適）
                if 'output' in name:
                    # 出力層は小さな値で初期化（安定性）
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        最適化された順伝播
        
        Args:
            obs: 観測データ (batch_size, obs_dim)
            
        Returns:
            action: 行動出力 (batch_size, action_dim)
        """
        x = obs
        
        # 入力層
        x = self.network['input_layer'](x)
        if 'input_norm' in self.network:
            x = self.network['input_norm'](x)
        x = self.activation(x)
        
        # 隠れ層
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:
                x = self.network[f'hidden_{i}'](x)
                if f'hidden_norm_{i}' in self.network:
                    x = self.network[f'hidden_norm_{i}'](x)
                x = self.activation(x)
            
            # 残差ブロック
            if f'residual_{i}' in self.network:
                x = self.network[f'residual_{i}'](x)
            
            # ドロップアウト
            if f'dropout_{i}' in self.network:
                x = self.network[f'dropout_{i}'](x)
        
        # アテンション機構
        if 'attention' in self.network:
            # アテンション用に次元を拡張
            x_expanded = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
            x = self.network['attention'](x_expanded)
        
        # 出力層
        x = self.network['output_layer'](x)
        x = self.output_activation(x)
        
        return x


class OptimizedCriticNetwork(nn.Module):
    """
    最適化されたCritic Network
    
    改善点:
    - より深いネットワーク構造
    - 値関数の安定性向上
    - 正則化による過学習防止
    """
    
    def __init__(self, 
                 obs_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 activation: str = "Swish",
                 dropout_rate: float = 0.1,
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        """
        最適化されたCritic Networkの初期化
        """
        super(OptimizedCriticNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 活性化関数の設定
        self.activation = self._get_activation(activation)
        
        # ネットワーク層の構築
        self.network = self._build_network()
        
        # 重みの初期化
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str):
        """活性化関数の取得"""
        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "LeakyReLU": nn.LeakyReLU(0.01),
            "ELU": nn.ELU(),
            "Swish": nn.SiLU(),
            "GELU": nn.GELU()
        }
        return activations.get(activation_name, nn.SiLU())
    
    def _build_network(self) -> nn.ModuleDict:
        """ネットワーク構造の構築"""
        network = nn.ModuleDict()
        
        # 入力層
        input_dim = self.obs_dim
        network['input_layer'] = nn.Linear(input_dim, self.hidden_dims[0])
        if self.use_layer_norm:
            network['input_norm'] = nn.LayerNorm(self.hidden_dims[0])
        
        # 隠れ層
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:
                network[f'hidden_{i}'] = nn.Linear(self.hidden_dims[i-1], hidden_dim)
                if self.use_layer_norm:
                    network[f'hidden_norm_{i}'] = nn.LayerNorm(hidden_dim)
            
            # 残差ブロック
            if self.use_residual and hidden_dim == self.hidden_dims[0]:
                network[f'residual_{i}'] = ResidualBlock(hidden_dim, self.dropout_rate)
            
            # ドロップアウト
            if self.dropout_rate > 0:
                network[f'dropout_{i}'] = nn.Dropout(self.dropout_rate)
        
        # 出力層（値関数は1次元出力）
        network['output_layer'] = nn.Linear(self.hidden_dims[-1], 1)
        
        return network
    
    def _initialize_weights(self):
        """重み初期化"""
        for name, module in self.network.items():
            if isinstance(module, nn.Linear):
                if 'output' in name:
                    # 出力層は小さな値で初期化
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            obs: 観測データ (batch_size, obs_dim)
            
        Returns:
            value: 状態価値 (batch_size, 1)
        """
        x = obs
        
        # 入力層
        x = self.network['input_layer'](x)
        if 'input_norm' in self.network:
            x = self.network['input_norm'](x)
        x = self.activation(x)
        
        # 隠れ層
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i > 0:
                x = self.network[f'hidden_{i}'](x)
                if f'hidden_norm_{i}' in self.network:
                    x = self.network[f'hidden_norm_{i}'](x)
                x = self.activation(x)
            
            # 残差ブロック
            if f'residual_{i}' in self.network:
                x = self.network[f'residual_{i}'](x)
            
            # ドロップアウト
            if f'dropout_{i}' in self.network:
                x = self.network[f'dropout_{i}'](x)
        
        # 出力層
        x = self.network['output_layer'](x)
        
        return x


class OptimizedPPONetwork(nn.Module):
    """
    最適化されたPPOネットワーク（Actor-Critic統合）
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        最適化されたPPOネットワークの初期化
        
        Args:
            obs_dim: 観測次元数
            action_dim: 行動次元数
            config: ネットワーク設定
        """
        super(OptimizedPPONetwork, self).__init__()
        
        # 設定の取得
        actor_config = config.get('actor', {})
        critic_config = config.get('critic', {})
        
        # Actorネットワーク
        self.actor = OptimizedActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_config.get('hidden_layers', [256, 256, 128]),
            activation=actor_config.get('activation', 'Swish'),
            output_activation=actor_config.get('output_activation', 'Tanh'),
            dropout_rate=actor_config.get('dropout_rate', 0.1),
            use_residual=actor_config.get('use_residual', True),
            use_attention=actor_config.get('use_attention', False),
            use_layer_norm=actor_config.get('use_layer_norm', True)
        )
        
        # Criticネットワーク
        self.critic = OptimizedCriticNetwork(
            obs_dim=obs_dim,
            hidden_dims=critic_config.get('hidden_layers', [256, 256, 128]),
            activation=critic_config.get('activation', 'Swish'),
            dropout_rate=critic_config.get('dropout_rate', 0.1),
            use_residual=critic_config.get('use_residual', True),
            use_layer_norm=critic_config.get('use_layer_norm', True)
        )
        
        # 行動分布のパラメータ（対数標準偏差）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        行動と価値を同時に取得
        
        Args:
            obs: 観測データ
            
        Returns:
            action: サンプリングされた行動
            log_prob: 行動の対数確率
            value: 状態価値
        """
        # 行動の平均値
        action_mean = self.actor(obs)
        
        # 状態価値
        value = self.critic(obs)
        
        # 行動分布（正規分布）
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        # 行動サンプリング
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        既存の行動を評価
        
        Args:
            obs: 観測データ
            actions: 評価する行動
            
        Returns:
            log_prob: 行動の対数確率
            value: 状態価値
            entropy: エントロピー
        """
        # 行動の平均値
        action_mean = self.actor(obs)
        
        # 状態価値
        value = self.critic(obs)
        
        # 行動分布
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        # 対数確率とエントロピー
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy
