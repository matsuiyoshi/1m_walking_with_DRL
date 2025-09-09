#!/usr/bin/env python3
"""
Evaluation Script for Bittle Walking DRL
Bittle四足歩行ロボットの評価スクリプト
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training import Evaluator
from src.models import PPOAgent


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Bittle Walking DRL Evaluation')
    
    # モデルファイル
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained model file')
    
    # 設定ファイルのパス
    parser.add_argument('--env-config', type=str, default='config/env_config.yaml',
                       help='Environment configuration file path')
    parser.add_argument('--bittle-config', type=str, default='config/bittle_config.yaml',
                       help='Bittle configuration file path')
    
    # 評価設定
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during evaluation')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic actions')
    parser.add_argument('--save-trajectories', action='store_true',
                       help='Save trajectory data')
    
    # 出力設定
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results')
    
    # デバッグ設定
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # ログレベルの設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # ファイルの存在確認
    if not os.path.exists(args.model_path):
        logger.error(f"モデルファイルが見つかりません: {args.model_path}")
        sys.exit(1)
    
    config_files = [args.env_config, args.bittle_config]
    for config_file in config_files:
        if not os.path.exists(config_file):
            logger.error(f"設定ファイルが見つかりません: {config_file}")
            sys.exit(1)
    
    # 出力ディレクトリの設定
    if args.output_dir is None:
        model_name = Path(args.model_path).stem
        args.output_dir = f"data/evaluations/{model_name}"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # エージェントの作成
        logger.info("エージェントを初期化しています...")
        agent = PPOAgent(
            obs_dim=32,  # 観測次元（設定から取得する方が良い）
            action_dim=9,  # 行動次元（設定から取得する方が良い）
            config_path="config/training_config.yaml"
        )
        
        # モデルの読み込み
        logger.info(f"モデルを読み込み中: {args.model_path}")
        agent.load(args.model_path)
        
        # 評価器の作成
        logger.info("評価器を初期化しています...")
        evaluator = Evaluator(
            env_config_path=args.env_config,
            bittle_config_path=args.bittle_config,
            output_dir=output_dir
        )
        
        # 評価の実行
        logger.info(f"評価を開始します（エピソード数: {args.num_episodes}）...")
        results = evaluator.evaluate(
            agent=agent,
            num_episodes=args.num_episodes,
            render=args.render,
            deterministic=args.deterministic,
            save_trajectories=args.save_trajectories
        )
        
        # 結果の表示
        logger.info("評価完了!")
        logger.info("=" * 50)
        logger.info("評価結果:")
        logger.info(f"  エピソード数: {results['num_episodes']}")
        logger.info(f"  成功率: {results['success_rate']:.2%}")
        logger.info(f"  平均報酬: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"  平均距離: {results['avg_distance']:.3f}m ± {results['std_distance']:.3f}m")
        logger.info(f"  平均時間: {results['avg_time']:.2f}s ± {results['std_time']:.2f}s")
        logger.info(f"  成功回数: {results['success_count']}/{results['num_episodes']}")
        logger.info(f"  効率性: {results.get('efficiency', 0):.3f}")
        logger.info("=" * 50)
        
        # 結果の保存
        logger.info(f"評価結果を保存しました: {output_dir}")
        
    except Exception as e:
        logger.error(f"評価中にエラーが発生しました: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # リソースの解放
        if 'evaluator' in locals():
            evaluator.close()


if __name__ == "__main__":
    main()
