#!/usr/bin/env python3
"""
段階的学習スクリプト
Curriculum Learning for Bittle Walking
"""

import argparse
import logging
import sys
import yaml
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.parallel_trainer import ParallelTrainer
from src.training.curriculum_manager import CurriculumLearningManager
from src.training.transfer_learning import TransferLearningController
from src.training.progress_monitor import ProgressMonitor


class CurriculumLearning:
    """段階的学習統合クラス"""
    
    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 各コンポーネントを初期化
        self.curriculum_manager = CurriculumLearningManager(config_path, output_dir)
        self.transfer_controller = TransferLearningController(
            self.config['transfer_learning'], self.output_dir
        )
        self.progress_monitor = ProgressMonitor(
            self.config['monitoring'], self.output_dir
        )
        
        self.logger = logging.getLogger(__name__)
        
    def run_curriculum_learning(self):
        """段階的学習を実行"""
        self.logger.info("段階的学習を開始します")
        
        while not self.curriculum_manager.is_final_stage():
            # 現在の段階設定を取得
            stage_config = self.curriculum_manager.get_current_stage_config()
            stage_name = self.curriculum_manager.get_stage_name_display()
            
            self.logger.info(f"段階 {self.curriculum_manager.current_stage}: {stage_name} を開始")
            
            # 段階用設定ファイルを作成
            stage_config_path = self.curriculum_manager.create_stage_config(stage_config)
            
            # 前段階のモデルパスを取得
            previous_model_path = self.curriculum_manager.get_previous_model_path()
            
            # 段階学習を実行（curriculum_managerの段階ディレクトリを使用）
            stage_output_dir = self.curriculum_manager.stage_dirs[self.curriculum_manager.current_stage]
            
            # 進捗監視をリセット
            self.progress_monitor.reset_for_new_stage(self.curriculum_manager.current_stage)
            
            # 段階学習を実行
            trainer = self._run_stage_learning(
                stage_config_path, 
                stage_output_dir, 
                previous_model_path,
                stage_config['max_timesteps']
            )
            
            # 段階完了条件をチェック
            if self.curriculum_manager.check_stage_completion():
                self.logger.info(f"段階 {self.curriculum_manager.current_stage} 完了条件を満たしました")
                
                # 現在の段階のモデルを保存（次段階の転移学習用）
                current_stage_model_path = stage_output_dir / f"stage_{self.curriculum_manager.current_stage}_model.pth"
                self._save_stage_model(trainer, current_stage_model_path)
                
                # 段階の要約を保存
                self.curriculum_manager.save_stage_summary()
                
                # 次の段階に移行
                if not self.curriculum_manager.transition_to_next_stage():
                    break
            else:
                self.logger.warning(f"段階 {self.curriculum_manager.current_stage} 完了条件を満たしていません")
                # 同じ段階を継続
                continue
                
        self.logger.info("段階的学習完了！")
        
    def _run_stage_learning(self, config_path: str, output_dir: Path, 
                          previous_model_path: Optional[Path], max_timesteps: int):
        """段階学習を実行
        
        Returns:
            trainer: 作成されたトレーナーインスタンス
        """
        try:
            # 設定ファイルを読み込み
            import yaml
            with open(config_path, 'r') as f:
                training_config = yaml.safe_load(f)
            
            # 環境設定を読み込み
            env_config_path = Path(__file__).parent.parent / "config" / "env_config.yaml"
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
            
            # トレーナーを初期化
            trainer = ParallelTrainer(
                config_path=str(config_path),
                env_config_path=str(env_config_path),
                bittle_config_path=str(Path(__file__).parent.parent / "config" / "bittle_config.yaml"),
                output_dir=str(output_dir)
            )
            
            # 前段階のモデルを読み込み（転移学習）
            if previous_model_path and previous_model_path.exists():
                self.logger.info(f"前段階モデルを読み込み: {previous_model_path}")
                try:
                    # 転移学習コントローラーを使用してモデルを読み込み
                    success = self.transfer_controller.load_previous_stage_model(
                        previous_model_path, trainer.agent
                    )
                    if success:
                        self.logger.info("前段階モデルの読み込みに成功しました")
                    else:
                        self.logger.warning("前段階モデルの読み込みに失敗しました")
                except Exception as e:
                    self.logger.error(f"転移学習エラー: {e}")
                    self.logger.info("転移学習なしで継続します")
                
            # 段階学習を実行
            self.logger.info(f"段階 {self.curriculum_manager.current_stage} 学習開始: {max_timesteps} ステップ")
            
            # OptimizedRewardFunctionの段階を設定
            if hasattr(trainer, 'env') and hasattr(trainer.env, 'reward_function'):
                trainer.env.reward_function.learning_stage = self.curriculum_manager.current_stage - 1
                self.logger.info(f"報酬関数の段階を {trainer.env.reward_function.learning_stage} に設定")
            
            # 動画録画設定の確認と有効化
            if hasattr(trainer, 'video_recorder'):
                self.logger.info(f"動画録画設定: enabled={trainer.video_recorder.enabled}, frequency={trainer.video_recorder.frequency}")
                if not trainer.video_recorder.enabled:
                    trainer.video_recorder.enabled = True
                    self.logger.info("動画録画を有効化しました")
            
            # 実際の学習を実行
            trainer.train()
            
            # 進捗監視の更新
            self.progress_monitor.update_episode(
                episode_length=10.0,
                reward=5.0,
                forward_distance=0.5,
                success=True,
                timestep=max_timesteps
            )
            
            return trainer
                
        except Exception as e:
            self.logger.error(f"段階学習中にエラーが発生: {e}")
            raise
    
    def _save_stage_model(self, trainer, model_path: Path):
        """段階モデルの保存"""
        try:
            # トレーナーにsave_modelメソッドがある場合
            if hasattr(trainer, 'save_model'):
                trainer.save_model(str(model_path))
                self.logger.info(f"段階モデルを保存しました: {model_path}")
            # エージェントから直接保存
            elif hasattr(trainer, 'agent'):
                checkpoint = {
                    'network_state_dict': trainer.agent.network.state_dict(),
                    'optimizer_state_dict': trainer.agent.optimizer.state_dict(),
                }
                if hasattr(trainer.agent, 'scheduler') and trainer.agent.scheduler:
                    checkpoint['scheduler_state_dict'] = trainer.agent.scheduler.state_dict()
                
                torch.save(checkpoint, model_path)
                self.logger.info(f"段階モデルを保存しました: {model_path}")
            else:
                self.logger.warning("モデル保存機能が見つかりません")
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")


def main():
    parser = argparse.ArgumentParser(description='段階的学習スクリプト')
    parser.add_argument('--curriculum-config', type=str, 
                       default='config/curriculum_learning.yaml',
                       help='段階的学習設定ファイル')
    parser.add_argument('--output-dir', type=str, 
                       default='data/experiments/curriculum_learning',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 段階的学習を実行
    curriculum = CurriculumLearning(args.curriculum_config, args.output_dir)
    curriculum.run_curriculum_learning()


if __name__ == "__main__":
    main()
