#!/usr/bin/env python3
"""
PyBullet専用のモデル評価スクリプト
透明度問題を解決し、シンプルで確実な動画記録を実装
"""

import os
import sys
import argparse
import numpy as np
import torch
import pybullet as p
import pybullet_data
import yaml
from pathlib import Path
import time
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.ppo_agent import PPOAgent
from src.environment.bittle_env import BittleWalkingEnv


class PyBulletEvaluator:
    """PyBullet専用の評価器"""
    
    def __init__(self, model_path: str, num_episodes: int = 5, save_video: bool = True):
        """
        評価器の初期化
        
        Args:
            model_path: 学習済みモデルのパス
            num_episodes: 評価エピソード数
            save_video: 動画保存の有無
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.save_video = save_video
        
        # 出力ディレクトリ
        self.output_dir = Path("data/evaluations/pybullet_evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # PyBulletの初期化
        self.physics_client = p.connect(p.DIRECT)  # ヘッドレスモード
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 環境の設定
        self.timestep = 1/240  # 240Hz
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -9.81)
        
        # ロボットの読み込み
        self.robot_id = self._load_robot()
        
        # 環境の作成
        self._create_environment()
        
        # エージェントの読み込み
        self.agent = self._load_agent()
        
        # 動画記録の準備
        if self.save_video:
            self.video_frames = []
            self.video_writer = None
    
    def _load_robot(self):
        """ロボットの読み込みと色設定"""
        urdf_path = "assets/bittle-urdf/bittle.urdf"
        full_urdf_path = os.path.join(os.getcwd(), urdf_path)
        
        if not os.path.exists(full_urdf_path):
            raise FileNotFoundError(f"URDF file not found: {full_urdf_path}")
        
        # ロボットの読み込み
        robot_id = p.loadURDF(
            full_urdf_path,
            basePosition=[0, 0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        
        # 透明度問題を解決するための色設定
        self._set_robot_colors(robot_id)
        
        return robot_id
    
    def _set_robot_colors(self, robot_id):
        """ロボットの色を設定（透明度問題の解決）"""
        # ベースリンクを明るい青色に設定
        p.changeVisualShape(robot_id, -1, rgbaColor=[0.2, 0.4, 1.0, 1.0])
        
        # 全ての関節リンクを明るい赤色に設定
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            p.changeVisualShape(robot_id, i, rgbaColor=[1.0, 0.3, 0.3, 1.0])
        
        print(f"ロボットの色を設定しました（{num_joints}個の関節）")
    
    def _create_environment(self):
        """環境の作成"""
        # 地面の作成
        plane_id = p.loadURDF("plane.urdf")
        
        # 通路の作成（壁）
        wall_height = 0.05
        wall_thickness = 0.01
        corridor_width = 0.15
        
        # 左壁
        left_wall = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness/2, 1.0, wall_height/2]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=left_wall,
            basePosition=[-corridor_width/2, 0, wall_height/2]
        )
        
        # 右壁
        right_wall = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[wall_thickness/2, 1.0, wall_height/2]
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=right_wall,
            basePosition=[corridor_width/2, 0, wall_height/2]
        )
        
        print("環境を作成しました")
    
    def _load_agent(self):
        """エージェントの読み込み"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 設定ファイルの読み込み
        config_path = "config/env_config.yaml"
        bittle_config_path = "config/bittle_config.yaml"
        
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        
        with open(bittle_config_path, 'r') as f:
            bittle_config = yaml.safe_load(f)
        
        # エージェントの初期化
        agent = PPOAgent(
            obs_dim=30,  # 正しい観測次元
            action_dim=8,  # 正しい行動次元
            device=device
        )
        
        # モデルの読み込み
        checkpoint = torch.load(self.model_path, map_location=device)
        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"モデルを読み込みました: {self.model_path}")
        return agent
    
    def _get_observation(self):
        """観測値の取得（30次元）"""
        # ロボットの位置と姿勢
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # 関節角度（8次元）
        joint_states = p.getJointStates(self.robot_id, range(8))  # 最初の8関節のみ
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # IMUデータ（6次元）
        imu_data = [0.0] * 6  # 簡略化
        
        # 位置・姿勢（6次元）
        position_orientation = list(pos[:3]) + list(orn[:3])
        
        # 目標位置（2次元）
        target_position = [1.0, 0.0]  # 1m前方
        
        # 観測値の結合（30次元）
        observation = np.concatenate([
            joint_angles,        # 8次元
            joint_velocities,    # 8次元
            imu_data,           # 6次元
            position_orientation, # 6次元
            target_position     # 2次元
        ])
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, prev_pos, current_pos):
        """報酬の計算"""
        # 前進距離
        forward_distance = current_pos[0] - prev_pos[0]
        
        # 基本的な報酬
        reward = forward_distance * 10.0  # 前進報酬
        
        # 転倒ペナルティ
        if current_pos[2] < 0.05:  # 高さが低すぎる
            reward -= 100.0
        
        return reward
    
    def _render_frame(self):
        """フレームのレンダリング"""
        # カメラ設定
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        camera_target = [robot_pos[0], robot_pos[1], robot_pos[2]]
        
        # カメラの位置と向き
        camera_distance = 2.0
        camera_yaw = 0
        camera_pitch = -30
        
        # ビューマトリックスの計算
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        # プロジェクションマトリックス
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=640/480,
            nearVal=0.1,
            farVal=100.0
        )
        
        # 画像の取得
        width, height = 640, 480
        _, _, rgb_array, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        return rgb_array
    
    def _save_video(self, frames, output_path):
        """動画の保存（PyBulletの内蔵機能を使用）"""
        if not frames:
            print("保存するフレームがありません")
            return
        
        # フレームをnumpy配列に変換
        frame_array = np.array(frames)
        
        # 動画ファイルの保存（imageioを使用）
        try:
            import imageio
            with imageio.get_writer(output_path, fps=60, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"動画を保存しました: {output_path}")
        except ImportError:
            print("imageioが利用できません。フレームを個別に保存します。")
            # フレームを個別の画像として保存
            frame_dir = output_path.parent / f"{output_path.stem}_frames"
            frame_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                frame_path = frame_dir / f"frame_{i:04d}.png"
                import cv2
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"フレームを保存しました: {frame_dir}")
    
    def evaluate(self):
        """評価の実行"""
        print(f"評価を開始します（エピソード数: {self.num_episodes}）")
        
        results = {
            'episodes': [],
            'total_rewards': [],
            'distances': [],
            'successes': [],
            'times': []
        }
        
        for episode in range(self.num_episodes):
            print(f"\nエピソード {episode + 1}/{self.num_episodes}")
            
            # エピソードの初期化
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [0, 0, 0.1],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            
            # 関節の初期化
            for i in range(p.getNumJoints(self.robot_id)):
                p.resetJointState(self.robot_id, i, 0)
            
            # エピソード変数
            total_reward = 0
            step_count = 0
            max_steps = 1000
            start_time = time.time()
            
            # 動画フレームの初期化
            if self.save_video:
                episode_frames = []
            
            # エピソードの実行
            prev_pos = [0, 0, 0.1]
            
            while step_count < max_steps:
                # 観測値の取得
                observation = self._get_observation()
                
                # 行動の選択
                with torch.no_grad():
                    action, _, _ = self.agent.get_action(observation, deterministic=True)
                
                # 行動の実行（8関節のみ）
                for i in range(min(8, len(action))):
                    p.setJointMotorControl2(
                        self.robot_id,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=action[i],
                        force=10.0
                    )
                
                # シミュレーションのステップ
                p.stepSimulation()
                
                # 報酬の計算
                current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                reward = self._calculate_reward(prev_pos, current_pos)
                total_reward += reward
                
                # フレームの記録
                if self.save_video and step_count % 4 == 0:  # 60fps用に4フレームごと
                    frame = self._render_frame()
                    if frame is not None:
                        episode_frames.append(frame)
                
                # 終了条件のチェック
                if current_pos[2] < 0.05:  # 転倒
                    break
                
                if current_pos[0] > 1.0:  # ゴール到達
                    break
                
                prev_pos = current_pos
                step_count += 1
            
            # エピソード結果の記録
            episode_time = time.time() - start_time
            distance = current_pos[0]
            success = distance > 0.9  # 90cm以上で成功
            
            results['episodes'].append(episode + 1)
            results['total_rewards'].append(total_reward)
            results['distances'].append(distance)
            results['successes'].append(success)
            results['times'].append(episode_time)
            
            print(f"  報酬: {total_reward:.2f}")
            print(f"  距離: {distance:.3f}m")
            print(f"  成功: {success}")
            print(f"  時間: {episode_time:.2f}s")
            
            # 動画の保存
            if self.save_video and episode_frames:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_path = self.output_dir / f"episode_{episode}_video_{timestamp}.mp4"
                self._save_video(episode_frames, video_path)
        
        # 結果の表示
        self._print_results(results)
        
        # 結果の保存
        self._save_results(results)
        
        # PyBulletの終了
        p.disconnect(self.physics_client)
        
        return results
    
    def _print_results(self, results):
        """結果の表示"""
        print("\n" + "="*50)
        print("評価結果:")
        print(f"  エピソード数: {len(results['episodes'])}")
        print(f"  成功率: {sum(results['successes'])/len(results['successes'])*100:.1f}%")
        print(f"  平均報酬: {np.mean(results['total_rewards']):.2f} ± {np.std(results['total_rewards']):.2f}")
        print(f"  平均距離: {np.mean(results['distances']):.3f}m ± {np.std(results['distances']):.3f}m")
        print(f"  平均時間: {np.mean(results['times']):.2f}s ± {np.std(results['times']):.2f}s")
        print(f"  成功回数: {sum(results['successes'])}/{len(results['successes'])}")
        print("="*50)
    
    def _save_results(self, results):
        """結果の保存"""
        results_path = self.output_dir / "evaluation_results.yaml"
        
        # 結果をYAML形式で保存
        results_data = {
            'evaluation_info': {
                'model_path': self.model_path,
                'num_episodes': self.num_episodes,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'success_rate': float(sum(results['successes'])/len(results['successes'])),
                'mean_reward': float(np.mean(results['total_rewards'])),
                'std_reward': float(np.std(results['total_rewards'])),
                'mean_distance': float(np.mean(results['distances'])),
                'std_distance': float(np.std(results['distances'])),
                'mean_time': float(np.mean(results['times'])),
                'std_time': float(np.std(results['times'])),
                'success_count': int(sum(results['successes'])),
                'total_episodes': len(results['episodes'])
            },
            'episodes': [
                {
                    'episode': int(ep),
                    'reward': float(reward),
                    'distance': float(dist),
                    'success': bool(success),
                    'time': float(time)
                }
                for ep, reward, dist, success, time in zip(
                    results['episodes'],
                    results['total_rewards'],
                    results['distances'],
                    results['successes'],
                    results['times']
                )
            ]
        }
        
        with open(results_path, 'w') as f:
            yaml.dump(results_data, f, default_flow_style=False)
        
        print(f"結果を保存しました: {results_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PyBullet専用のモデル評価')
    parser.add_argument('--model-path', type=str, required=True,
                        help='学習済みモデルのパス')
    parser.add_argument('--num-episodes', type=int, default=5,
                        help='評価エピソード数')
    parser.add_argument('--save-video', action='store_true',
                        help='動画の保存')
    
    args = parser.parse_args()
    
    # 評価器の作成と実行
    evaluator = PyBulletEvaluator(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        save_video=args.save_video
    )
    
    # 評価の実行
    results = evaluator.evaluate()
    
    print("評価完了!")


if __name__ == "__main__":
    main()
