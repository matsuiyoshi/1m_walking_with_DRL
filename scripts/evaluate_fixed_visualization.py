#!/usr/bin/env python3
"""
修正された可視化評価スクリプト
URDFの構造問題と色設定問題を解決
"""

import os
import sys
import pybullet as p
import pybullet_data
import numpy as np
import torch
import yaml
import imageio
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.ppo_agent import PPOAgent


class FixedVisualizationEvaluator:
    """修正された可視化評価器"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path or "config/env_config.yaml"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 出力ディレクトリ
        self.output_dir = Path("data/evaluations/fixed_visualization")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PyBulletの初期化
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 設定の読み込み
        self.config = self._load_config()
        
        # ロボットとエージェントの読み込み
        self.robot_id = None
        self.agent = None
        self._load_robot()
        self._load_agent()
        
        print(f"✅ 修正された可視化評価器を初期化しました")
        print(f"   デバイス: {self.device}")
        print(f"   モデル: {self.model_path}")
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def _load_config(self):
        """設定ファイルの読み込み"""
        config_path = project_root / self.config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_robot(self):
        """ロボットの読み込みと可視化の修正"""
        print("\n=== ロボットの読み込みと可視化修正 ===")
        
        # URDFファイルのパス
        urdf_path = "assets/bittle-urdf/bittle.urdf"
        full_urdf_path = os.path.join(os.getcwd(), urdf_path)
        
        # ロボットの読み込み（フラグなしでシンプルに）
        self.robot_id = p.loadURDF(
            full_urdf_path,
            basePosition=[0, 0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        
        print(f"✅ ロボット読み込み成功: ID={self.robot_id}")
        
        # 可視化の修正
        self._fix_robot_visualization()
        
        # 物理パラメータの設定
        self._setup_physics()
    
    def _fix_robot_visualization(self):
        """ロボットの可視化を修正"""
        print("\n--- 可視化の修正 ---")
        
        # デバッグビジュアライザーの設定
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        
        # ロボットの色設定（各リンクに適切な色を設定）
        self._set_robot_colors()
        
        # レンダリング設定の最適化
        self._optimize_rendering()
    
    def _set_robot_colors(self):
        """ロボットの色を適切に設定"""
        print("  色設定を実行中...")
        
        # ベースリンク（メインボディ）を青色に設定
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0.2, 0.4, 1.0, 1.0])
        
        # 各関節リンクに異なる色を設定
        num_joints = p.getNumJoints(self.robot_id)
        colors = [
            [1.0, 0.3, 0.3, 1.0],  # 赤
            [0.3, 1.0, 0.3, 1.0],  # 緑
            [0.3, 0.3, 1.0, 1.0],  # 青
            [1.0, 1.0, 0.3, 1.0],  # 黄
            [1.0, 0.3, 1.0, 1.0],  # マゼンタ
            [0.3, 1.0, 1.0, 1.0],  # シアン
            [1.0, 0.6, 0.3, 1.0],  # オレンジ
            [0.6, 0.3, 1.0, 1.0],  # 紫
        ]
        
        for i in range(num_joints):
            color = colors[i % len(colors)]
            p.changeVisualShape(self.robot_id, i, rgbaColor=color)
            print(f"    関節 {i}: 色 {color}")
        
        print("  ✅ 色設定完了")
    
    def _optimize_rendering(self):
        """レンダリング設定を最適化"""
        print("  レンダリング設定を最適化中...")
        
        # より良いレンダラーを使用
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        
        # ライティングの設定
        p.setGravity(0, 0, -9.81)
        
        # 地面の追加
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1.0])
        
        print("  ✅ レンダリング設定完了")
    
    def _setup_physics(self):
        """物理パラメータの設定"""
        print("  物理パラメータを設定中...")
        
        # 重力の設定
        p.setGravity(0, 0, -9.81)
        
        # タイムステップの設定
        p.setTimeStep(1/240.0)
        
        print("  ✅ 物理パラメータ設定完了")
    
    def _load_agent(self):
        """エージェントの読み込み"""
        print("\n=== エージェントの読み込み ===")
        
        # エージェントの初期化（正しい次元を使用）
        self.agent = PPOAgent(
            obs_dim=32,  # 実際のモデルの観測次元
            action_dim=9,  # 実際のモデルの行動次元
            device=self.device
        )
        
        # モデルの読み込み
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ エージェント読み込み成功")
        print(f"   観測次元: 32")
        print(f"   行動次元: 9")
    
    def _get_observation(self):
        """観測データの取得（32次元）"""
        # ロボットの位置と姿勢
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 関節角度と速度（8関節）
        joint_states = p.getJointStates(self.robot_id, range(8))
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # 位置と姿勢の情報（6次元）
        # pos: 3次元, orn: 4次元 -> 7次元になるので、6次元に調整
        position_orientation = list(pos) + list(orn)[:3]  # 姿勢の最初の3要素のみ使用
        
        # 目標位置（前方1m）（2次元）
        target_position = [1.0, 0.0]
        
        # 簡易IMUデータ（6次元）
        imu_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # デバッグ情報
        print(f"  デバッグ - 観測データの次元:")
        print(f"    joint_angles: {len(joint_angles)}")
        print(f"    joint_velocities: {len(joint_velocities)}")
        print(f"    imu_data: {len(imu_data)}")
        print(f"    position_orientation: {len(position_orientation)}")
        print(f"    target_position: {len(target_position)}")
        
        # 観測ベクトルの構築（32次元）
        # 8 + 8 + 6 + 6 + 2 + 2 = 32次元
        observation = np.concatenate([
            joint_angles,        # 8次元
            joint_velocities,    # 8次元
            imu_data,           # 6次元
            position_orientation, # 6次元
            target_position,     # 2次元
            [0.0, 0.0]          # 2次元追加
        ])
        
        print(f"    合計: {len(observation)}次元")
        
        # 32次元であることを確認
        assert len(observation) == 32, f"観測次元が正しくありません: {len(observation)}"
        
        return observation.astype(np.float32)
    
    def evaluate(self, num_episodes=2, max_steps=1000):
        """評価の実行"""
        print(f"\n=== 修正された可視化評価の実行 ===")
        print(f"エピソード数: {num_episodes}")
        print(f"最大ステップ数: {max_steps}")
        
        results = []
        
        for episode in range(num_episodes):
            print(f"\n--- エピソード {episode} ---")
            
            # エピソードの初期化
            self._reset_episode()
            
            # 動画記録の準備
            frames = []
            episode_info = {
                'episode': episode,
                'total_reward': 0.0,
                'steps': 0,
                'final_position': [0.0, 0.0, 0.0],
                'success': False
            }
            
            for step in range(max_steps):
                # 観測の取得
                observation = self._get_observation()
                
                # 行動の選択
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(observation, deterministic=True)
                
                # 行動の実行
                self._execute_action(action)
                
                # 物理シミュレーションのステップ
                p.stepSimulation()
                
                # フレームの記録
                frame = self._render_frame()
                if frame is not None:
                    frames.append(frame)
                
                # エピソード情報の更新
                episode_info['steps'] = step + 1
                episode_info['total_reward'] += 0.1  # 簡易報酬
                
                # 終了条件のチェック
                if self._check_episode_done():
                    episode_info['success'] = True
                    break
            
            # 最終位置の記録
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            episode_info['final_position'] = list(pos)
            
            # 動画の保存
            if frames:
                video_path = self._save_episode_video(frames, episode)
                episode_info['video_path'] = str(video_path)
                print(f"  動画保存: {video_path}")
            
            results.append(episode_info)
            print(f"  ステップ数: {episode_info['steps']}")
            print(f"  最終位置: {episode_info['final_position']}")
            print(f"  成功: {episode_info['success']}")
        
        # 結果の保存
        self._save_results(results)
        
        return results
    
    def _reset_episode(self):
        """エピソードのリセット"""
        # ロボットの位置と姿勢をリセット
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 0.1],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # 関節のリセット
        for i in range(8):
            p.resetJointState(self.robot_id, i, 0.0, 0.0)
    
    def _execute_action(self, action):
        """行動の実行（9次元行動）"""
        for i in range(min(8, len(action))):  # 8関節のみ制御
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=10.0
            )
    
    def _render_frame(self):
        """フレームのレンダリング"""
        try:
            # カメラの設定
            camera_pos = [0, -2, 0.5]
            camera_target = [0, 0, 0.1]
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target,
                distance=2.0,
                yaw=0,
                pitch=-20,
                roll=0,
                upAxisIndex=2
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=640/480,
                nearVal=0.1,
                farVal=100.0
            )
            
            # 画像の取得
            width, height = 640, 480
            _, _, rgb_array, depth_array, seg_array = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )
            
            return rgb_array
            
        except Exception as e:
            print(f"  レンダリングエラー: {e}")
            return None
    
    def _check_episode_done(self):
        """エピソード終了条件のチェック"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # 前方に1m以上進んだら成功
        if pos[0] > 1.0:
            return True
        
        # 高さが0.5m以上になったら失敗
        if pos[2] > 0.5:
            return True
        
        return False
    
    def _save_episode_video(self, frames, episode):
        """エピソード動画の保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"episode_{episode}_video_{timestamp}.mp4"
        
        try:
            imageio.mimsave(str(video_path), frames, fps=30)
            return video_path
        except Exception as e:
            print(f"  動画保存エラー: {e}")
            return None
    
    def _save_results(self, results):
        """結果の保存"""
        results_path = self.output_dir / "evaluation_results.yaml"
        
        # 結果の整理
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'total_episodes': len(results),
            'successful_episodes': sum(1 for r in results if r['success']),
            'average_steps': np.mean([r['steps'] for r in results]),
            'episodes': results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n✅ 結果を保存しました: {results_path}")
    
    def close(self):
        """リソースのクリーンアップ"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


def main():
    """メイン関数"""
    # モデルファイルのパス
    model_path = "data/final_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    
    # 評価器の初期化
    evaluator = FixedVisualizationEvaluator(model_path)
    
    try:
        # 評価の実行
        results = evaluator.evaluate(num_episodes=2, max_steps=1000)
        
        # 結果の表示
        print(f"\n=== 評価結果 ===")
        print(f"総エピソード数: {len(results)}")
        print(f"成功エピソード数: {sum(1 for r in results if r['success'])}")
        print(f"平均ステップ数: {np.mean([r['steps'] for r in results]):.1f}")
        
    except Exception as e:
        print(f"❌ 評価中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # リソースのクリーンアップ
        evaluator.close()


if __name__ == "__main__":
    main()
