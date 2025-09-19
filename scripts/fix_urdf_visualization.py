#!/usr/bin/env python3
"""
URDFの可視化問題の根本的修正
色と構造の問題を解決
"""

import os
import sys
import pybullet as p
import pybullet_data
import numpy as np
import imageio
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def fix_urdf_visualization():
    """URDFの可視化問題の根本的修正"""
    print("=== URDFの可視化問題の根本的修正 ===")
    
    # PyBulletの初期化
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 1. ロボットの読み込み
    print("\n1. ロボットの読み込み")
    urdf_path = "assets/bittle-urdf/bittle.urdf"
    full_urdf_path = os.path.join(os.getcwd(), urdf_path)
    
    robot_id = p.loadURDF(
        full_urdf_path,
        basePosition=[0, 0, 0.1],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False
    )
    
    print(f"✅ ロボット読み込み成功: ID={robot_id}")
    
    # 2. 根本的な色設定の修正
    print("\n2. 根本的な色設定の修正")
    
    # ベースリンクの色設定（濃い青色）
    p.changeVisualShape(robot_id, -1, rgbaColor=[0.2, 0.4, 1.0, 1.0])
    print("✅ ベースリンク: 濃い青色に設定")
    
    # 各関節リンクの色設定（明確に区別できる色）
    num_joints = p.getNumJoints(robot_id)
    colors = [
        [1.0, 0.2, 0.2, 1.0],  # 濃い赤
        [0.2, 1.0, 0.2, 1.0],  # 濃い緑
        [0.2, 0.2, 1.0, 1.0],  # 濃い青
        [1.0, 1.0, 0.2, 1.0],  # 濃い黄
        [1.0, 0.2, 1.0, 1.0],  # 濃いマゼンタ
        [0.2, 1.0, 1.0, 1.0],  # 濃いシアン
        [1.0, 0.6, 0.2, 1.0],  # 濃いオレンジ
        [0.6, 0.2, 1.0, 1.0],  # 濃い紫
    ]
    
    for i in range(num_joints):
        color = colors[i % len(colors)]
        p.changeVisualShape(robot_id, i, rgbaColor=color)
        print(f"✅ 関節 {i}: {color} に設定")
    
    # 3. レンダリング設定の最適化
    print("\n3. レンダリング設定の最適化")
    
    # デバッグビジュアライザーの設定
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    
    # ライティングの設定
    p.setGravity(0, 0, -9.81)
    
    # 地面の追加（濃いグレー）
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1.0])
    print("✅ 地面: 濃いグレーに設定")
    
    # 4. 修正後の画像生成
    print("\n4. 修正後の画像生成")
    
    try:
        # カメラ設定
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
        
        # 画像取得
        width, height = 640, 480
        _, _, rgb_array, depth_array, seg_array = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # 画像分析
        print(f"  画像サイズ: {rgb_array.shape}")
        print(f"  色範囲: {rgb_array.min()} - {rgb_array.max()}")
        
        # 色の分布分析
        unique_colors = len(np.unique(rgb_array.reshape(-1, 3), axis=0))
        print(f"  ユニーク色数: {unique_colors}")
        
        # 明度分析
        brightness = np.mean(rgb_array)
        print(f"  平均明度: {brightness:.2f}")
        
        # 修正後の画像の保存
        fixed_image_path = "fixed_urdf_visualization.png"
        imageio.imwrite(fixed_image_path, rgb_array)
        print(f"✅ 修正後の画像を保存: {fixed_image_path}")
        
        # 改善の確認
        if brightness < 200:
            print("✅ 明度が適切に調整されました")
        else:
            print("⚠️  明度がまだ高すぎます")
            
        if unique_colors > 50:
            print("✅ 色の種類が十分に増加しました")
        else:
            print("⚠️  色の種類がまだ少ないです")
            
    except Exception as e:
        print(f"❌ 画像生成失敗: {e}")
    
    # 5. 物理パラメータの確認
    print("\n5. 物理パラメータの確認")
    
    # ロボットの位置と姿勢
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    print(f"  ロボット位置: {pos}")
    print(f"  ロボット姿勢: {orn}")
    
    # 各関節の状態
    print("\n  関節状態:")
    for i in range(num_joints):
        joint_state = p.getJointState(robot_id, i)
        print(f"    関節 {i}: 角度={joint_state[0]:.3f}, 速度={joint_state[1]:.3f}")
    
    # PyBulletの終了
    p.disconnect(physics_client)
    
    print("\n=== URDFの可視化問題の修正完了 ===")
    print("1. 根本的な色設定の修正")
    print("2. レンダリング設定の最適化")
    print("3. 修正後の画像生成")
    print("4. 物理パラメータの確認")


if __name__ == "__main__":
    fix_urdf_visualization()
