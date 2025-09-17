"""
並列環境の実装
GPU並列処理を活用した複数環境の同時実行
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from gymnasium import Env
from .bittle_env import BittleWalkingEnv


class ParallelBittleEnv:
    """並列Bittle環境の実装"""
    
    def __init__(self, 
                 num_envs: int = 8,
                 env_config_path: str = "config/env_config.yaml",
                 bittle_config_path: str = "config/bittle_config.yaml"):
        """
        並列環境の初期化
        
        Args:
            num_envs: 並列環境数
            env_config_path: 環境設定ファイルのパス
            bittle_config_path: Bittle設定ファイルのパス
        """
        self.num_envs = num_envs
        self.envs = []
        
        # 各環境を初期化
        print(f"並列環境を初期化中... (環境数: {num_envs})")
        for i in range(num_envs):
            print(f"環境 {i+1}/{num_envs} を初期化中...")
            env = BittleWalkingEnv(
                config_path=env_config_path,
                bittle_config_path=bittle_config_path,
                render=False,  # 並列環境ではレンダリングを無効化
                render_mode=None
            )
            self.envs.append(env)
            print(f"環境 {i+1} の初期化完了")
        
        # 環境の状態を取得
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # 現在の状態を保存
        self.current_obs = np.zeros((num_envs, self.observation_space.shape[0]))
        self.current_dones = np.zeros(num_envs, dtype=bool)
        
        # 初期化
        self.reset()
    
    def reset(self) -> np.ndarray:
        """全環境をリセット"""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs_list.append(obs)
            self.current_obs[i] = obs
            self.current_dones[i] = False
        
        return np.array(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        並列ステップ実行
        
        Args:
            actions: 各環境の行動 (num_envs, action_dim)
            
        Returns:
            observations: 各環境の観測 (num_envs, obs_dim)
            rewards: 各環境の報酬 (num_envs,)
            dones: 各環境の終了フラグ (num_envs,)
            infos: 各環境の情報
        """
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        
        # 各環境でステップ実行
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if not self.current_dones[i]:
                obs, reward, done, info = env.step(action)
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
                
                self.current_obs[i] = obs
                self.current_dones[i] = done
            else:
                # 終了済みの環境は前の状態を維持
                obs_list.append(self.current_obs[i])
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({})
        
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(done_list),
            info_list
        )
    
    def close(self):
        """全環境を閉じる"""
        for env in self.envs:
            env.close()


class GPUParallelBittleEnv(ParallelBittleEnv):
    """GPU並列処理を活用したBittle環境"""
    
    def __init__(self, 
                 num_envs: int = 8,
                 env_config_path: str = "config/env_config.yaml",
                 bittle_config_path: str = "config/bittle_config.yaml",
                 device: str = "cuda"):
        """
        GPU並列環境の初期化
        
        Args:
            num_envs: 並列環境数
            env_config_path: 環境設定ファイルのパス
            bittle_config_path: Bittle設定ファイルのパス
            device: 使用するデバイス ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        super().__init__(num_envs, env_config_path, bittle_config_path)
        
        # GPU並列処理の最適化設定
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def step(self, actions: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        GPU並列ステップ実行
        
        Args:
            actions: 各環境の行動 (num_envs, action_dim)
            
        Returns:
            observations: 各環境の観測 (num_envs, obs_dim) - GPU tensor
            rewards: 各環境の報酬 (num_envs,) - GPU tensor
            dones: 各環境の終了フラグ (num_envs,) - GPU tensor
            infos: 各環境の情報
        """
        # CPUで環境を実行
        obs, rewards, dones, infos = super().step(actions)
        
        # GPU tensorに変換
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).bool().to(self.device)
        
        return obs_tensor, rewards_tensor, dones_tensor, infos
    
    def reset(self) -> torch.Tensor:
        """全環境をリセット（GPU tensorで返す）"""
        obs = super().reset()
        return torch.from_numpy(obs).float().to(self.device)
