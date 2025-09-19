"""
学習進捗動画記録器
学習中のロボットの挙動を動画で記録し、学習進捗を可視化
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import time


class VideoRecorder:
    """学習進捗動画記録器"""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        動画記録器の初期化
        
        Args:
            config: 動画記録設定
            output_dir: 出力ディレクトリ
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 動画記録設定
        self.enabled = config.get('enabled', False)
        self.frequency = config.get('frequency', 180000)  # デフォルト: 5分ごと
        self.episodes_per_video = config.get('episodes_per_video', 3)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1200)
        
        # 動画品質設定
        video_quality = config.get('video_quality', {})
        self.fps = video_quality.get('fps', 60)
        self.resolution = video_quality.get('resolution', [640, 480])
        self.format = video_quality.get('format', 'mp4')
        
        # フレーム記録の制御（シミュレーション240Hz vs 動画60fps = 4:1の比率）
        self.simulation_fps = 240  # PyBulletのシミュレーション周波数
        self.frame_skip = self.simulation_fps // self.fps  # 4フレームごとに1フレーム記録
        self.frame_counter = 0
        
        # オーバーレイ情報設定
        self.overlay_info = config.get('overlay_info', True)
        self.info_font_size = config.get('info_font_size', 16)
        self.info_color = tuple(config.get('info_color', [255, 255, 255]))
        self.info_position = tuple(config.get('info_position', [10, 10]))
        
        # 動画記録状態
        self.video_writer = None
        self.current_video_path = None
        self.episode_frames = []
        self.episode_count = 0
        self.last_recording_step = 0
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"動画記録器を初期化しました（頻度: {self.frequency}ステップ）")
            self.logger.info(f"動画品質: {self.resolution[0]}x{self.resolution[1]}@{self.fps}fps")
            self.logger.info(f"フレームスキップ: {self.frame_skip}（シミュレーション{self.simulation_fps}Hz -> 動画{self.fps}fps）")
    
    def should_record(self, current_step: int) -> bool:
        """
        動画記録のタイミング判定
        
        Args:
            current_step: 現在の学習ステップ数
            
        Returns:
            bool: 記録すべきかどうか
        """
        if not self.enabled:
            return False
        
        return (current_step - self.last_recording_step) >= self.frequency
    
    def start_recording(self, current_step: int, episode_num: int) -> bool:
        """
        動画記録の開始
        
        Args:
            current_step: 現在の学習ステップ数
            episode_num: エピソード番号
            
        Returns:
            bool: 記録開始に成功したかどうか
        """
        if not self.enabled:
            return False
        
        try:
            # 動画ファイル名の生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"training_progress_step{current_step}_ep{episode_num}_{timestamp}.{self.format}"
            self.current_video_path = self.output_dir / video_filename
            
            # 動画ライターの初期化（H.264コーデックで互換性向上）
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                self.fps,
                (self.resolution[0], self.resolution[1])
            )
            
            if not self.video_writer.isOpened():
                self.logger.error(f"動画ファイルの作成に失敗しました: {self.current_video_path}")
                return False
            
            self.episode_frames = []
            self.episode_count = 0
            self.last_recording_step = current_step
            
            self.logger.info(f"動画記録を開始しました: {self.current_video_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"動画記録開始エラー: {e}")
            return False
    
    def record_frame(self, frame: np.ndarray, episode_info: Dict[str, Any]) -> bool:
        """
        フレームの記録（適切なフレームレートで記録）
        
        Args:
            frame: 記録するフレーム
            episode_info: エピソード情報
            
        Returns:
            bool: 記録に成功したかどうか
        """
        if not self.enabled or self.video_writer is None:
            return False
        
        try:
            # フレームスキップの制御（240Hz -> 60fps）
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                return True  # スキップされたフレームは記録しない
            
            # フレームの前処理
            processed_frame = self._preprocess_frame(frame)
            
            # オーバーレイ情報の追加
            if self.overlay_info:
                processed_frame = self._add_overlay_info(processed_frame, episode_info)
            
            # フレームの記録
            self.video_writer.write(processed_frame)
            self.episode_frames.append(processed_frame.copy())
            
            return True
            
        except Exception as e:
            self.logger.error(f"フレーム記録エラー: {e}")
            return False
    
    def finish_episode(self, episode_info: Dict[str, Any]) -> bool:
        """
        エピソードの記録完了
        
        Args:
            episode_info: エピソード情報
            
        Returns:
            bool: 記録完了に成功したかどうか
        """
        if not self.enabled or self.video_writer is None:
            return False
        
        try:
            self.episode_count += 1
            
            # エピソード区切りを追加（黒フレーム）
            separator_frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            if self.overlay_info:
                separator_frame = self._add_episode_separator(separator_frame, episode_info)
            
            # 区切りフレームを追加（1秒分）
            for _ in range(self.fps):
                self.video_writer.write(separator_frame)
            
            # フレームカウンターをリセット
            self.frame_counter = 0
            
            # 指定されたエピソード数に達したら動画を完了
            if self.episode_count >= self.episodes_per_video:
                return self.finish_recording()
            
            return True
            
        except Exception as e:
            self.logger.error(f"エピソード完了エラー: {e}")
            return False
    
    def finish_recording(self) -> bool:
        """
        動画記録の完了
        
        Returns:
            bool: 記録完了に成功したかどうか
        """
        if not self.enabled or self.video_writer is None:
            return False
        
        try:
            # 動画ライターを閉じる
            self.video_writer.release()
            self.video_writer = None
            
            # ファイルサイズの確認
            if self.current_video_path and self.current_video_path.exists():
                file_size = self.current_video_path.stat().st_size
                self.logger.info(f"動画記録完了: {self.current_video_path} ({file_size / 1024 / 1024:.1f}MB)")
                return True
            else:
                self.logger.error("動画ファイルが作成されませんでした")
                return False
                
        except Exception as e:
            self.logger.error(f"動画記録完了エラー: {e}")
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームの前処理
        
        Args:
            frame: 元のフレーム
            
        Returns:
            np.ndarray: 処理済みフレーム
        """
        # フレームの連続メモリレイアウトを保証（OpenCV互換性のため）
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # フレームサイズの調整
        if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
            frame = cv2.resize(frame, (self.resolution[0], self.resolution[1]))
        
        # 色空間の調整（BGRに変換）
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
        else:
            # グレースケールの場合はRGBに変換
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                # ダミーフレームを作成
                frame = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        # 最終的な連続メモリレイアウトを保証
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        return frame
    
    def _add_overlay_info(self, frame: np.ndarray, episode_info: Dict[str, Any]) -> np.ndarray:
        """
        オーバーレイ情報の追加
        
        Args:
            frame: 元のフレーム
            episode_info: エピソード情報
            
        Returns:
            np.ndarray: 情報付きフレーム
        """
        try:
            # フォントの設定
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.info_font_size / 20.0  # フォントサイズの調整
            thickness = 2
            
            # 表示する情報
            info_lines = [
                f"Episode: {episode_info.get('episode', 0)}",
                f"Step: {episode_info.get('step', 0)}",
                f"Reward: {episode_info.get('reward', 0.0):.2f}",
                f"Distance: {episode_info.get('distance', 0.0):.3f}m",
                f"Success: {episode_info.get('success', False)}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]
            
            # 各情報行を描画
            y_offset = self.info_position[1]
            for line in info_lines:
                cv2.putText(
                    frame, line, 
                    (self.info_position[0], y_offset),
                    font, font_scale, self.info_color, thickness
                )
                y_offset += int(self.info_font_size * 1.2)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"オーバーレイ情報追加エラー: {e}")
            return frame
    
    def _add_episode_separator(self, frame: np.ndarray, episode_info: Dict[str, Any]) -> np.ndarray:
        """
        エピソード区切りフレームの作成
        
        Args:
            frame: 元のフレーム
            episode_info: エピソード情報
            
        Returns:
            np.ndarray: 区切りフレーム
        """
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            
            # エピソード完了情報
            text = f"Episode {episode_info.get('episode', 0)} Complete"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # テキストの中央配置
            x = (self.resolution[0] - text_size[0]) // 2
            y = (self.resolution[1] + text_size[1]) // 2
            
            cv2.putText(frame, text, (x, y), font, font_scale, self.info_color, thickness)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"区切りフレーム作成エラー: {e}")
            return frame
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.logger.info("動画記録器のクリーンアップ完了")
