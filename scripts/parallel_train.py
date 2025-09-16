#!/usr/bin/env python3
"""
並列学習スクリプト
GPU並列処理を活用した高速学習
"""

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.parallel_trainer import ParallelTrainer


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/logs/parallel_training.log')
        ]
    )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='並列学習スクリプト')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='学習設定ファイルのパス')
    parser.add_argument('--env-config', type=str, default='config/env_config.yaml',
                       help='環境設定ファイルのパス')
    parser.add_argument('--bittle-config', type=str, default='config/bittle_config.yaml',
                       help='Bittle設定ファイルのパス')
    parser.add_argument('--output-dir', type=str, default='data/experiments/parallel_training',
                       help='出力ディレクトリ')
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='総学習タイムステップ数（設定ファイルを上書き）')
    parser.add_argument('--eval-freq', type=int, default=None,
                       help='評価頻度（設定ファイルを上書き）')
    parser.add_argument('--save-freq', type=int, default=None,
                       help='モデル保存頻度（設定ファイルを上書き）')
    parser.add_argument('--num-envs', type=int, default=None,
                       help='並列環境数（設定ファイルを上書き）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='バッチサイズ（設定ファイルを上書き）')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("並列学習を開始します...")
    logger.info(f"設定ファイル: {args.config}")
    logger.info(f"出力ディレクトリ: {output_dir}")
    
    try:
        # 並列トレーナーの初期化
        trainer = ParallelTrainer(
            config_path=args.config,
            env_config_path=args.env_config,
            bittle_config_path=args.bittle_config,
            output_dir=str(output_dir)
        )
        
        # コマンドライン引数で設定を上書き
        if args.total_timesteps is not None:
            trainer.config['training']['total_timesteps'] = args.total_timesteps
        if args.eval_freq is not None:
            trainer.config['training']['eval_freq'] = args.eval_freq
        if args.save_freq is not None:
            trainer.config['training']['save_freq'] = args.save_freq
        if args.num_envs is not None:
            trainer.config['training']['env']['num_envs'] = args.num_envs
            trainer.num_envs = args.num_envs
        if args.batch_size is not None:
            if 'hyperparameters' not in trainer.config:
                trainer.config['hyperparameters'] = {}
            trainer.config['hyperparameters']['batch_size'] = args.batch_size
        
        logger.info(f"並列環境数: {trainer.num_envs}")
        batch_size = trainer.config.get('hyperparameters', {}).get('batch_size', 256)
        logger.info(f"バッチサイズ: {batch_size}")
        logger.info(f"総タイムステップ数: {trainer.config['training']['total_timesteps']}")
        
        # 学習実行
        results = trainer.train()
        
        logger.info("並列学習完了!")
        logger.info("=" * 50)
        logger.info("学習結果:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"並列学習中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
