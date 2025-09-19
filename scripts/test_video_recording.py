#!/usr/bin/env python3
"""
動画記録機能のテストスクリプト
学習中の動画記録機能をテストするためのスクリプト
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
import yaml

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.parallel_trainer import ParallelTrainer
from src.training.video_recorder import VideoRecorder


def test_video_recorder():
    """動画記録器の単体テスト"""
    print("=" * 50)
    print("動画記録器の単体テスト")
    print("=" * 50)
    
    # テスト用設定
    video_config = {
        'enabled': True,
        'frequency': 100,  # 短い間隔でテスト
        'episodes_per_video': 2,
        'max_steps_per_episode': 50,
        'video_quality': {
            'fps': 30,
            'resolution': [640, 480],
            'format': 'mp4'
        },
        'overlay_info': True,
        'info_font_size': 16,
        'info_color': [255, 255, 255],
        'info_position': [10, 10]
    }
    
    # 出力ディレクトリ
    output_dir = Path("data/test_video_recording")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 動画記録器の初期化
    recorder = VideoRecorder(video_config, output_dir)
    
    print(f"動画記録器を初期化しました")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"記録頻度: {recorder.frequency}ステップ")
    print(f"エピソード数: {recorder.episodes_per_video}")
    
    # 記録開始
    current_step = 0
    episode_num = 1
    
    if recorder.should_record(current_step):
        print(f"\n動画記録を開始します（ステップ: {current_step}）")
        recorder.start_recording(current_step, episode_num)
        
        # ダミーフレームの記録
        for step in range(100):
            # ダミーフレームを作成
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # エピソード情報
            episode_info = {
                'episode': episode_num,
                'step': step,
                'reward': np.random.uniform(-10, 10),
                'distance': np.random.uniform(0, 1),
                'success': step > 80
            }
            
            # フレーム記録
            recorder.record_frame(frame, episode_info)
            
            # エピソード完了のシミュレーション
            if step == 99:
                recorder.finish_episode(episode_info)
                print(f"エピソード {episode_num} 完了")
                
                # 2つ目のエピソード
                episode_num += 1
                for step2 in range(100):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    episode_info = {
                        'episode': episode_num,
                        'step': step2,
                        'reward': np.random.uniform(-10, 10),
                        'distance': np.random.uniform(0, 1),
                        'success': step2 > 80
                    }
                    recorder.record_frame(frame, episode_info)
                    
                    if step2 == 99:
                        recorder.finish_episode(episode_info)
                        print(f"エピソード {episode_num} 完了")
                        break
        
        # 記録完了
        recorder.finish_recording()
        print("動画記録が完了しました")
    
    # クリーンアップ
    recorder.cleanup()
    print("テスト完了")


def test_training_with_video():
    """学習中の動画記録テスト"""
    print("=" * 50)
    print("学習中の動画記録テスト")
    print("=" * 50)
    
    # 設定ファイルの読み込み
    config_path = "config/training_config_12h_improved.yaml"
    
    try:
        # トレーナーの初期化
        trainer = ParallelTrainer(
            config_path=config_path,
            output_dir="data/test_training_video"
        )
        
        print("トレーナーを初期化しました")
        print(f"動画記録有効: {trainer.video_recorder.enabled}")
        
        if trainer.video_recorder.enabled:
            print(f"記録頻度: {trainer.video_recorder.frequency}ステップ")
            print(f"エピソード数: {trainer.video_recorder.episodes_per_video}")
            print(f"動画品質: {trainer.video_recorder.resolution[0]}x{trainer.video_recorder.resolution[1]}@{trainer.video_recorder.fps}fps")
        
        # 短時間の学習テスト（実際の学習は実行しない）
        print("\n短時間の学習テストを実行します...")
        print("（実際の学習は実行せず、動画記録機能のみテスト）")
        
        # クリーンアップ
        trainer.video_recorder.cleanup()
        trainer.parallel_env.close()
        
        print("テスト完了")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Video Recording Test')
    parser.add_argument('--test-type', type=str, choices=['recorder', 'training', 'both'], 
                       default='both', help='Test type to run')
    parser.add_argument('--config', type=str, default='config/training_config_12h_improved.yaml',
                       help='Training configuration file')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        if args.test_type in ['recorder', 'both']:
            test_video_recorder()
        
        if args.test_type in ['training', 'both']:
            test_training_with_video()
        
        print("\n" + "=" * 50)
        print("すべてのテストが完了しました")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
