#!/usr/bin/env python3
"""
Training Script for Bittle Walking DRL
Bittle四足歩行ロボットの深層強化学習スクリプト
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training import Trainer


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
        # 学習器の作成
        logger.info("学習器を初期化しています...")
        trainer = Trainer(
            config_path=args.config,
            env_config_path=args.env_config,
            bittle_config_path=args.bittle_config,
            output_dir=str(output_dir),
            experiment_name=args.experiment_name
        )
        
        # 設定の上書き
        if args.total_timesteps is not None:
            trainer.config['training']['total_timesteps'] = args.total_timesteps
        if args.eval_freq is not None:
            trainer.config['training']['eval_freq'] = args.eval_freq
        if args.save_freq is not None:
            trainer.config['training']['save_freq'] = args.save_freq
        
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


if __name__ == "__main__":
    main()
