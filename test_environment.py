#!/usr/bin/env python3
"""
Environment Test Script
環境のテスト用スクリプト
"""

import sys
import os
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 環境の基本テスト
def test_basic_environment():
    """基本的な環境テスト"""
    print("=== 基本環境テスト ===")
    
    try:
        from src.environment.bittle_env import BittleWalkingEnv
        
        # 環境の作成
        print("環境を作成中...")
        env = BittleWalkingEnv(
            config_path="config/env_config.yaml",
            bittle_config_path="config/bittle_config.yaml",
            render=False
        )
        
        print(f"観測空間: {env.observation_space}")
        print(f"行動空間: {env.action_space}")
        
        # 環境のリセット
        print("環境をリセット中...")
        obs = env.reset()
        print(f"初期観測形状: {obs.shape}")
        print(f"初期観測例: {obs[:5]}...")  # 最初の5要素
        
        # ランダム行動でのテスト
        print("ランダム行動でテスト中...")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            print(f"Step {step+1}: reward={reward:.3f}, done={done}")
            if 'distance_to_target' in info:
                print(f"  目標までの距離: {info['distance_to_target']:.3f}m")
            
            if done:
                print("エピソード終了")
                break
        
        env.close()
        print("✅ 基本環境テスト成功!")
        return True
        
    except Exception as e:
        print(f"❌ 基本環境テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """モデルのテスト"""
    print("\n=== モデルテスト ===")
    
    try:
        from src.models.ppo_agent import PPOAgent
        
        # エージェントの作成
        print("PPOエージェントを作成中...")
        agent = PPOAgent(
            obs_dim=32,
            action_dim=9,
            config_path="config/training_config.yaml"
        )
        
        # ダミー観測でテスト
        obs = np.random.randn(32)
        action, log_prob, value = agent.get_action(obs)
        
        print(f"行動形状: {action.shape}")
        print(f"対数確率: {log_prob}")
        print(f"状態価値: {value}")
        
        print("✅ モデルテスト成功!")
        return True
        
    except Exception as e:
        print(f"❌ モデルテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """学習コンポーネントのテスト"""
    print("\n=== 学習コンポーネントテスト ===")
    
    try:
        from src.training.trainer import Trainer
        
        # 学習器の作成（実際の学習は行わない）
        print("学習器を作成中...")
        trainer = Trainer(
            config_path="config/training_config.yaml",
            env_config_path="config/env_config.yaml", 
            bittle_config_path="config/bittle_config.yaml",
            output_dir="data/test_experiments",
            experiment_name="test_run"
        )
        
        print("✅ 学習コンポーネントテスト成功!")
        trainer.close()
        return True
        
    except Exception as e:
        print(f"❌ 学習コンポーネントテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    print("Bittle DRL 環境テストを開始します\n")
    
    # 基本テスト
    success_count = 0
    total_tests = 3
    
    if test_basic_environment():
        success_count += 1
    
    if test_models():
        success_count += 1
    
    if test_training_components():
        success_count += 1
    
    # 結果表示
    print(f"\n=== テスト結果 ===")
    print(f"成功: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 すべてのテストが成功しました! 学習を開始できます。")
        return True
    else:
        print("⚠️  一部のテストが失敗しました。実装を確認してください。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
