#!/usr/bin/env python3
"""
最終的な可視化修正を適用した評価スクリプト
"""

import os
import sys
import pybullet as p
import pybullet_data
import numpy as np
import imageio
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.ppo_agent import PPOAgent


class FinalFixedEvaluator:
    """最終的な可視化修正を適用した評価器"""
    
    def __init__(self, model_path: str, output_dir: str = "data/evaluations/final_fix"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"デバイス: {self.device}")
        
        # PyBulletの初期化
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 基本的な環境設定
        p.setGravity(0, 0, -9.81)
        
        # 地面の追加（濃いグレー）
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])
        
        # ロボットの読み込み
        self.robot_id = self._load_robot()
        
        # エージェントの読み込み
        self.agent = self._load_agent()
        
        print("✅ 最終的な可視化修正を適用した評価器の初期化完了")
    
    def _load_robot(self):
        """ロボットの読み込み（修正版）"""
        urdf_path = "assets/bittle-urdf/bittle.urdf"
        full_urdf_path = os.path.join(os.getcwd(), urdf_path)
        
        robot_id = p.loadURDF(
            full_urdf_path,
            basePosition=[0, 0, 0.2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        
        # 根本的な色設定の修正
        p.changeVisualShape(robot_id, -1, rgbaColor=[0.0, 0.0, 1.0, 1.0])  # ベースリンク: 濃い青
        
        # 各関節の色設定
        num_joints = p.getNumJoints(robot_id)
        colors = [
            [1.0, 0.0, 0.0, 1.0],  # 濃い赤
            [0.0, 1.0, 0.0, 1.0],  # 濃い緑
            [0.0, 0.0, 1.0, 1.0],  # 濃い青
            [1.0, 1.0, 0.0, 1.0],  # 濃い黄
            [1.0, 0.0, 1.0, 1.0],  # 濃いマゼンタ
            [0.0, 1.0, 1.0, 1.0],  # 濃いシアン
            [1.0, 0.5, 0.0, 1.0],  # 濃いオレンジ
            [0.5, 0.0, 1.0, 1.0],  # 濃い紫
        ]
        
        for i in range(num_joints):
            color = colors[i % len(colors)]
            p.changeVisualShape(robot_id, i, rgbaColor=color)
        
        print(f"✅ ロボット読み込み成功: ID={robot_id}")
        return robot_id
    
    def _load_agent(self):
        """エージェントの読み込み"""
        agent = PPOAgent(
            obs_dim=32,
            action_dim=9,
            device=self.device
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print("✅ エージェント読み込み成功")
        return agent
    
    def _get_observation(self):
        """観測データの取得（32次元）"""
        # ロボットの位置と姿勢
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 関節角度と速度（8関節）
        joint_states = p.getJointStates(self.robot_id, range(8))
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # 位置と姿勢の情報（6次元）
        position_orientation = list(pos) + list(orn)[:3]  # 姿勢の最初の3要素のみ使用
        
        # 目標位置（前方1m）（2次元）
        target_position = [1.0, 0.0]
        
        # 簡易IMUデータ（6次元）
        imu_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 観測ベクトルの構築（32次元）
        observation = np.concatenate([
            joint_angles,        # 8次元
            joint_velocities,    # 8次元
            imu_data,           # 6次元
            position_orientation, # 6次元
            target_position,     # 2次元
            [0.0, 0.0]          # 2次元追加
        ])
        
        return observation.astype(np.float32)
    
    def _execute_action(self, action):
        """行動の実行（8関節制御）"""
        for i in range(min(8, len(action))):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=10.0
            )
    
    def _render_frame(self):
        """フレームのレンダリング（修正版）"""
        # カメラ設定の最適化
        camera_pos = [0, -1.5, 0.8]
        camera_target = [0, 0, 0.2]
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=1.5,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=640/480,
            nearVal=0.1,
            farVal=10.0
        )
        
        # 画像取得
        width, height = 640, 480
        _, _, rgb_array, depth_array, seg_array = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        return rgb_array
    
    def evaluate(self, num_episodes: int = 2):
        """評価の実行"""
        print(f"\n=== 最終的な可視化修正を適用した評価開始 ===")
        print(f"エピソード数: {num_episodes}")
        
        results = []
        
        for episode in range(num_episodes):
            print(f"\n--- エピソード {episode} ---")
            
            # エピソードの初期化
            p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.2], [0, 0, 0, 1])
            
            # 関節の初期化
            for i in range(8):
                p.resetJointState(self.robot_id, i, 0, 0)
            
            # 動画記録の準備
            frames = []
            episode_reward = 0
            episode_steps = 0
            
            # エピソードの実行
            for step in range(100):  # 最大100ステップ
                # 観測の取得
                observation = self._get_observation()
                
                # 行動の選択
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(observation, deterministic=True)
                
                # 行動の実行
                self._execute_action(action)
                
                # 物理シミュレーションの実行
                p.stepSimulation()
                
                # フレームのレンダリング
                frame = self._render_frame()
                frames.append(frame)
                
                # 報酬の計算（簡易版）
                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                distance = np.sqrt(pos[0]**2 + pos[1]**2)
                reward = -distance  # 距離に基づく報酬
                episode_reward += reward
                episode_steps += 1
                
                # 終了条件のチェック
                if distance > 1.0:  # 1m以上進んだら成功
                    print(f"✅ エピソード {episode}: 成功！距離: {distance:.2f}m")
                    break
            
            # エピソード結果の保存
            episode_result = {
                'episode': episode,
                'steps': episode_steps,
                'reward': episode_reward,
                'success': episode_steps < 100
            }
            results.append(episode_result)
            
            # 動画の保存
            if frames:
                video_path = self.output_dir / f"episode_{episode}_video_final_fix.mp4"
                imageio.mimsave(video_path, frames, fps=30)
                print(f"✅ 動画保存: {video_path}")
        
        # 結果の保存
        results_path = self.output_dir / "evaluation_results_final_fix.yaml"
        import yaml
        with open(results_path, 'w') as f:
            yaml.dump(results, f)
        
        print(f"\n=== 評価完了 ===")
        print(f"結果保存: {results_path}")
        
        # 統計の表示
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        avg_reward = sum(r['reward'] for r in results) / len(results)
        
        print(f"成功率: {success_rate:.2%}")
        print(f"平均ステップ数: {avg_steps:.1f}")
        print(f"平均報酬: {avg_reward:.2f}")
        
        return results
    
    def close(self):
        """リソースの解放"""
        p.disconnect(self.physics_client)


def main():
    """メイン関数"""
    model_path = "data/final_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    
    evaluator = FinalFixedEvaluator(model_path)
    
    try:
        results = evaluator.evaluate(num_episodes=2)
        print("✅ 評価完了")
    except Exception as e:
        print(f"❌ 評価中にエラーが発生しました: {e}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
