#!/usr/bin/env python3
"""
Training Script for Bittle Walking DRL
Bittle四足歩行ロボットの深層強化学習スクリプト
"""

import os
import sys
import argparse
import logging
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training import Trainer
from src.training.parallel_trainer import ParallelTrainer


def start_tensorboard_auto(log_dir: str, port: int = 6006):
    """
    自動でTensorBoardを起動し、ブラウザで開く
    
    Args:
        log_dir: TensorBoardのログディレクトリ
        port: TensorBoardのポート番号
    """
    def run_tensorboard():
        try:
            # TensorBoardの起動
            cmd = [
                'tensorboard',
                '--logdir', log_dir,
                '--port', str(port),
                '--host', '0.0.0.0',
                '--reload_interval', '5',
                '--max_reload_threads', '4'
            ]
            
            print(f"TensorBoardを起動しています...")
            print(f"ログディレクトリ: {log_dir}")
            print(f"URL: http://localhost:{port}")
            print(f"コマンド: {' '.join(cmd)}")
            
            # TensorBoardプロセスを開始
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 少し待ってからブラウザで開く
            time.sleep(3)
            
            # ブラウザでTensorBoardを開く
            try:
                webbrowser.open(f'http://localhost:{port}')
                print(f"ブラウザでTensorBoardを開いています: http://localhost:{port}")
            except Exception as e:
                print(f"ブラウザの自動起動に失敗しました: {e}")
                print(f"手動で http://localhost:{port} にアクセスしてください")
            
            # プロセスが終了するまで待機
            process.wait()
            
        except Exception as e:
            print(f"TensorBoardの起動に失敗しました: {e}")
    
    # 別スレッドでTensorBoardを起動
    tensorboard_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tensorboard_thread.start()
    
    return tensorboard_thread


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Bittle Walking DRL Training')
    
    # 設定ファイルのパス
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training configuration file path')
    parser.add_argument('--env-config', type=str, default='config/env_config.yaml',
                       help='Environment configuration file path')
    parser.add_argument('--bittle-config', type=str, default='config/bittle_config.yaml',
                       help='Bittle configuration file path')
    
    # 出力設定
    parser.add_argument('--output-dir', type=str, default='data/experiments',
                       help='Output directory for experiments')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    
    # 学習設定
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='Total training timesteps (overrides config)')
    parser.add_argument('--eval-freq', type=int, default=None,
                       help='Evaluation frequency (overrides config)')
    parser.add_argument('--save-freq', type=int, default=None,
                       help='Model save frequency (overrides config)')
    
    # デバッグ設定
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training')
    parser.add_argument('--no-eval', action='store_true',
                       help='Disable evaluation during training')
    
    # TensorBoard設定
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Enable automatic TensorBoard startup (default: True)')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable automatic TensorBoard startup')
    parser.add_argument('--tensorboard-port', type=int, default=6006,
                       help='TensorBoard port (default: 6006)')
    
    # 段階的学習設定
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning mode')
    parser.add_argument('--curriculum-config', type=str, default='config/curriculum_learning.yaml',
                       help='Curriculum learning configuration file path')
    
    args = parser.parse_args()
    
    # ログレベルの設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 設定ファイルの存在確認
    config_files = [args.config, args.env_config, args.bittle_config]
    for config_file in config_files:
        if not os.path.exists(config_file):
            logger.error(f"設定ファイルが見つかりません: {config_file}")
            sys.exit(1)
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 段階的学習モードの確認
        if args.curriculum:
            logger.info("段階的学習モードを有効化します")
            # 段階的学習の実行
            from scripts.curriculum_train import CurriculumLearning
            curriculum = CurriculumLearning(args.curriculum_config, str(output_dir))
            curriculum.run_curriculum_learning()
            return
        
        # 学習器の作成
        logger.info("学習器を初期化しています...")
        
        # 設定ファイルを読み込んで並列学習の設定を確認
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 並列学習が有効な場合はParallelTrainerを使用
        num_envs = config.get('training', {}).get('env', {}).get('num_envs', 1)
        if num_envs > 1:
            logger.info(f"並列学習を有効化: {num_envs}環境")
            trainer = ParallelTrainer(
                config_path=args.config,
                env_config_path=args.env_config,
                bittle_config_path=args.bittle_config,
                output_dir=str(output_dir),
                experiment_name=args.experiment_name
            )
        else:
            logger.info("単一環境学習を使用")
            trainer = Trainer(
                config_path=args.config,
                env_config_path=args.env_config,
                bittle_config_path=args.bittle_config,
                output_dir=str(output_dir),
                experiment_name=args.experiment_name
            )
        
        # TensorBoardの自動起動
        tensorboard_thread = None
        if args.tensorboard and not args.no_tensorboard:
            # TensorBoardのログディレクトリを取得
            tensorboard_log_dir = trainer.output_dir / "tensorboard"
            if hasattr(trainer, 'tensorboard_dir'):
                tensorboard_log_dir = trainer.tensorboard_dir
            
            logger.info("TensorBoardを自動起動しています...")
            tensorboard_thread = start_tensorboard_auto(
                str(tensorboard_log_dir), 
                args.tensorboard_port
            )
            logger.info(f"TensorBoardが起動しました: http://localhost:{args.tensorboard_port}")
            logger.info("リアルタイムで学習進捗を監視できます")
        
        # 設定の上書き
        if args.total_timesteps is not None:
            trainer.config['training']['total_timesteps'] = args.total_timesteps
        if args.eval_freq is not None:
            trainer.config['training']['eval_freq'] = args.eval_freq
        if args.save_freq is not None:
            trainer.config['training']['save_freq'] = args.save_freq
        if args.no_eval:
            trainer.config['training']['eval_freq'] = 0  # 評価を無効化
        
        # 学習の実行
        logger.info("学習を開始します...")
        results = trainer.train()
        
        # 結果の表示
        logger.info("学習完了!")
        logger.info(f"最終成功率: {results.get('success_rate', 0):.2%}")
        logger.info(f"平均報酬: {results.get('avg_reward', 0):.2f}")
        logger.info(f"平均距離: {results.get('avg_distance', 0):.3f}m")
        logger.info(f"平均時間: {results.get('avg_time', 0):.2f}s")
        
    except KeyboardInterrupt:
        logger.info("学習が中断されました")
    except Exception as e:
        logger.error(f"学習中にエラーが発生しました: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # リソースの解放
        if 'trainer' in locals():
            trainer.close()
        
        # TensorBoardプロセスの終了
        if 'tensorboard_thread' in locals() and tensorboard_thread:
            logger.info("TensorBoardを終了しています...")
            # デーモンスレッドなので自動的に終了します


if __name__ == "__main__":
    main()
