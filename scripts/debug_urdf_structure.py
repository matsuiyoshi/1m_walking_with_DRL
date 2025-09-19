#!/usr/bin/env python3
"""
URDFの構造問題の詳細調査
真の透明度問題の原因を特定
"""

import os
import sys
import pybullet as p
import pybullet_data
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def debug_urdf_structure():
    """URDFの構造問題の詳細調査"""
    print("=== URDFの構造問題の詳細調査 ===")
    
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
    
    # 2. 各リンクの詳細分析
    print("\n2. 各リンクの詳細分析")
    num_joints = p.getNumJoints(robot_id)
    
    for i in range(-1, num_joints):
        if i >= 0:
            joint_info = p.getJointInfo(robot_id, i)
            link_name = joint_info[12]
        else:
            link_name = "base_link"
        
        print(f"\n--- リンク {i} ({link_name}) ---")
        
        # 視覚形状データの取得
        visual_data = p.getVisualShapeData(robot_id, i)
        
        if visual_data:
            print(f"  ジオメトリ数: {len(visual_data)}")
            
            for j, visual_item in enumerate(visual_data):
                print(f"    ジオメトリ {j}:")
                print(f"      オブジェクトID: {visual_item[0]}")
                print(f"      リンクID: {visual_item[1]}")
                print(f"      ジオメトリタイプ: {visual_item[2]}")
                print(f"      寸法: {visual_item[3]}")
                print(f"      ファイル名: {visual_item[4]}")
                print(f"      位置: {visual_item[5]}")
                print(f"      向き: {visual_item[6]}")
                print(f"      色 (RGBA): {visual_item[7]}")
                
                # 透明度の詳細分析
                if len(visual_item) >= 8:
                    rgba = visual_item[7]
                    alpha = rgba[3] if len(rgba) > 3 else 1.0
                    print(f"      Alpha値: {alpha}")
                    
                    if alpha < 0.1:
                        print(f"      ⚠️  ほぼ透明")
                    elif alpha < 0.5:
                        print(f"      ⚠️  半透明")
                    elif alpha < 0.9:
                        print(f"      ⚠️  やや透明")
                    else:
                        print(f"      ✅ 不透明")
                        
                    # 色の明度分析
                    brightness = sum(rgba[:3]) / 3.0
                    print(f"      明度: {brightness:.3f}")
                    
                    if brightness < 0.1:
                        print(f"      ⚠️  非常に暗い")
                    elif brightness < 0.3:
                        print(f"      ⚠️  暗い")
                    elif brightness > 0.9:
                        print(f"      ⚠️  非常に明るい")
                    else:
                        print(f"      ✅ 適度な明度")
        else:
            print(f"    ❌ 視覚データなし")
    
    # 3. 色設定のテスト
    print("\n3. 色設定のテスト")
    
    # ベースリンクの色設定テスト
    print("\n--- ベースリンクの色設定テスト ---")
    test_colors = [
        ([1.0, 0.0, 0.0, 1.0], "明るい赤"),
        ([0.0, 1.0, 0.0, 1.0], "明るい緑"),
        ([0.0, 0.0, 1.0, 1.0], "明るい青"),
        ([1.0, 1.0, 0.0, 1.0], "明るい黄"),
        ([0.5, 0.5, 0.5, 1.0], "中程度のグレー"),
        ([0.2, 0.2, 0.2, 1.0], "暗いグレー"),
    ]
    
    for color, name in test_colors:
        print(f"  {name}設定テスト...")
        p.changeVisualShape(robot_id, -1, rgbaColor=color)
        
        # 設定後の確認
        visual_data = p.getVisualShapeData(robot_id, -1)
        if visual_data:
            current_color = visual_data[0][7]
            print(f"    設定後色: {current_color}")
            
            # 色の変化を確認
            color_diff = np.linalg.norm(np.array(current_color[:3]) - np.array(color[:3]))
            if color_diff < 0.1:
                print(f"    ✅ {name}色設定成功")
            else:
                print(f"    ❌ {name}色設定失敗 (差分: {color_diff:.3f})")
    
    # 4. カメラ画像の詳細分析
    print("\n4. カメラ画像の詳細分析")
    
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
        
        # 透明度の推定
        if brightness < 50:
            print(f"  ⚠️  非常に暗い - 透明度問題の可能性")
        elif unique_colors < 20:
            print(f"  ⚠️  色の種類が少ない - 透明度問題の可能性")
        else:
            print(f"  ✅ 画像は正常")
            
        # 画像の保存（デバッグ用）
        import imageio
        debug_image_path = "debug_urdf_image.png"
        imageio.imwrite(debug_image_path, rgb_array)
        print(f"  デバッグ画像を保存: {debug_image_path}")
        
    except Exception as e:
        print(f"  ❌ カメラ画像取得失敗: {e}")
    
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
    
    print("\n=== URDFの構造問題の分析結果 ===")
    print("1. 各リンクの視覚データを詳細分析")
    print("2. 色設定の段階的テスト")
    print("3. カメラ画像の詳細分析")
    print("4. 物理パラメータの確認")


if __name__ == "__main__":
    debug_urdf_structure()
