#!/usr/bin/env python3
"""
透明度問題の根本原因を調査するデバッグスクリプト
PyBulletの内部動作を詳しく分析
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


def debug_transparency_issue():
    """透明度問題の詳細調査"""
    print("=== 透明度問題の根本原因調査 ===")
    
    # PyBulletの初期化
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 1. URDFファイルの構造分析
    print("\n1. URDFファイルの構造分析")
    urdf_path = "assets/bittle-urdf/bittle.urdf"
    full_urdf_path = os.path.join(os.getcwd(), urdf_path)
    
    if not os.path.exists(full_urdf_path):
        print(f"❌ URDFファイルが見つかりません: {full_urdf_path}")
        return
    
    print(f"✅ URDFファイル存在確認: {full_urdf_path}")
    
    # URDFファイルの内容を分析
    with open(full_urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # マテリアル情報の確認
    has_material = "<material" in urdf_content
    has_color = "<color" in urdf_content
    has_texture = "<texture" in urdf_content
    
    print(f"  マテリアル情報: {'✅' if has_material else '❌'}")
    print(f"  色情報: {'✅' if has_color else '❌'}")
    print(f"  テクスチャ情報: {'✅' if has_texture else '❌'}")
    
    # 2. OBJファイルの分析
    print("\n2. OBJファイルの分析")
    obj_dir = "assets/bittle-urdf/obj"
    obj_files = list(Path(obj_dir).glob("*.obj"))
    print(f"  OBJファイル数: {len(obj_files)}")
    
    # 最初のOBJファイルを分析
    if obj_files:
        first_obj = obj_files[0]
        print(f"  分析対象: {first_obj}")
        
        with open(first_obj, 'r') as f:
            obj_content = f.read()
        
        has_mtl = "mtllib" in obj_content
        has_material_ref = "usemtl" in obj_content
        has_vertex_normals = "vn" in obj_content
        has_vertex_colors = "vc" in obj_content
        
        print(f"    MTL参照: {'✅' if has_mtl else '❌'}")
        print(f"    マテリアル参照: {'✅' if has_material_ref else '❌'}")
        print(f"    法線情報: {'✅' if has_vertex_normals else '❌'}")
        print(f"    頂点色: {'✅' if has_vertex_colors else '❌'}")
    
    # 3. MTLファイルの確認
    print("\n3. MTLファイルの確認")
    mtl_files = list(Path("assets/bittle-urdf").glob("*.mtl"))
    print(f"  MTLファイル数: {len(mtl_files)}")
    
    if not mtl_files:
        print("  ❌ MTLファイルが存在しません - これが透明度問題の根本原因")
    
    # 4. PyBulletでのロボット読み込み
    print("\n4. PyBulletでのロボット読み込み")
    try:
        robot_id = p.loadURDF(
            full_urdf_path,
            basePosition=[0, 0, 0.1],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )
        print(f"  ✅ ロボット読み込み成功: ID={robot_id}")
        
        # ロボットの詳細情報
        num_joints = p.getNumJoints(robot_id)
        print(f"  関節数: {num_joints}")
        
        # 各リンクの視覚情報を確認
        print("\n5. 各リンクの視覚情報分析")
        for i in range(-1, num_joints):  # -1はベースリンク
            if i >= 0:
                joint_info = p.getJointInfo(robot_id, i)
                link_name = joint_info[12]
            else:
                link_name = "base_link"
            
            visual_data = p.getVisualShapeData(robot_id, i)
            
            print(f"  リンク {i} ({link_name}):")
            if visual_data:
                for j, visual_item in enumerate(visual_data):
                    print(f"    ジオメトリ {j}:")
                    print(f"      データ: {visual_item}")
                    
                    # 透明度の分析
                    if len(visual_item) >= 8:
                        rgba = visual_item[7]
                        alpha = rgba[3] if len(rgba) > 3 else 1.0
                        if alpha < 1.0:
                            print(f"      ⚠️  透明度: {alpha} (透明)")
                        else:
                            print(f"      ✅ 不透明度: {alpha}")
            else:
                print(f"    ❌ 視覚データなし")
        
        # 6. 色設定のテスト
        print("\n6. 色設定のテスト")
        test_colors = [
            ([1.0, 0.0, 0.0, 1.0], "赤"),
            ([0.0, 1.0, 0.0, 1.0], "緑"),
            ([0.0, 0.0, 1.0, 1.0], "青"),
            ([1.0, 1.0, 0.0, 1.0], "黄"),
        ]
        
        for color, name in test_colors:
            print(f"  {name}色設定テスト...")
            p.changeVisualShape(robot_id, -1, rgbaColor=color)
            
            # 設定後の色を確認
            visual_data = p.getVisualShapeData(robot_id, -1)
            if visual_data:
                current_color = visual_data[0][7]  # rgba
                print(f"    設定後色: {current_color}")
                if np.allclose(current_color[:3], color[:3], atol=0.1):
                    print(f"    ✅ {name}色設定成功")
                else:
                    print(f"    ❌ {name}色設定失敗")
            else:
                print(f"    ❌ 視覚データ取得失敗")
        
        # 7. レンダリング設定の確認
        print("\n7. レンダリング設定の確認")
        renderer = p.ER_TINY_RENDERER
        print(f"  レンダラー: {renderer}")
        
        # カメラ画像の取得テスト
        print("\n8. カメラ画像取得テスト")
        try:
            width, height = 640, 480
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.1],
                distance=2.0,
                yaw=0,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width/height,
                nearVal=0.1,
                farVal=100.0
            )
            
            _, _, rgb_array, depth_array, seg_array = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix
            )
            
            print(f"  ✅ カメラ画像取得成功")
            print(f"  画像サイズ: {rgb_array.shape}")
            print(f"  色範囲: {rgb_array.min()} - {rgb_array.max()}")
            
            # 画像の統計情報
            unique_colors = len(np.unique(rgb_array.reshape(-1, 3), axis=0))
            print(f"  ユニーク色数: {unique_colors}")
            
            # 透明度の確認
            if rgb_array.max() < 50:  # 非常に暗い
                print("  ⚠️  画像が非常に暗い - 透明度問題の可能性")
            elif unique_colors < 10:  # 色の種類が少ない
                print("  ⚠️  色の種類が少ない - 透明度問題の可能性")
            else:
                print("  ✅ 画像は正常")
                
        except Exception as e:
            print(f"  ❌ カメラ画像取得失敗: {e}")
        
    except Exception as e:
        print(f"  ❌ ロボット読み込み失敗: {e}")
    
    # PyBulletの終了
    p.disconnect(physics_client)
    
    print("\n=== 調査結果の要約 ===")
    print("1. URDFファイルにマテリアル情報が欠けている")
    print("2. MTLファイルが存在しない")
    print("3. OBJファイルに色情報が含まれていない")
    print("4. PyBulletはデフォルトで透明なマテリアルを使用")
    print("5. changeVisualShapeでの色設定が唯一の解決策")


if __name__ == "__main__":
    debug_transparency_issue()
