#!/usr/bin/env python3
"""
TensorBoard起動スクリプト
並列学習の監視用
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import logging

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def start_tensorboard(log_dir: str, port: int = 6006, host: str = "0.0.0.0", 
                     auto_open: bool = True, reload_interval: int = 5):
    """
    TensorBoardを起動する
    
    Args:
        log_dir: TensorBoardログディレクトリ
        port: ポート番号
        host: ホスト名
        auto_open: 自動でブラウザを開くかどうか
        reload_interval: リロード間隔（秒）
    """
    logger = logging.getLogger(__name__)
    
    # ログディレクトリの存在確認
    log_path = Path(log_dir)
    if not log_path.exists():
        logger.error(f"ログディレクトリが存在しません: {log_dir}")
        return False
    
    # TensorBoardコマンドの構築
    cmd = [
        "tensorboard",
        "--logdir", str(log_path),
        "--port", str(port),
        "--host", host,
        "--reload_interval", str(reload_interval),
        "--max_reload_threads", "4"
    ]
    
    logger.info(f"TensorBoardを起動しています...")
    logger.info(f"ログディレクトリ: {log_dir}")
    logger.info(f"URL: http://{host}:{port}")
    logger.info(f"コマンド: {' '.join(cmd)}")
    
    try:
        # TensorBoardプロセスを起動
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 少し待ってからブラウザを開く
        if auto_open:
            time.sleep(3)
            url = f"http://{host}:{port}"
            logger.info(f"ブラウザでTensorBoardを開いています: {url}")
            webbrowser.open(url)
        
        logger.info("TensorBoardが起動しました。Ctrl+Cで終了します。")
        
        # プロセスの監視
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("TensorBoardを終了しています...")
            process.terminate()
            process.wait()
            logger.info("TensorBoardが終了しました。")
            
    except FileNotFoundError:
        logger.error("TensorBoardが見つかりません。インストールしてください:")
        logger.error("pip install tensorboard")
        return False
    except Exception as e:
        logger.error(f"TensorBoardの起動中にエラーが発生しました: {e}")
        return False
    
    return True


def find_latest_experiment(experiments_dir: str = "data/experiments"):
    """
    最新の実験ディレクトリを検索する
    
    Args:
        experiments_dir: 実験ディレクトリのパス
        
    Returns:
        最新のTensorBoardログディレクトリのパス
    """
    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        return None
    
    # 実験ディレクトリを検索
    experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
    if not experiment_dirs:
        return None
    
    # 最新のディレクトリを取得
    latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    
    # TensorBoardディレクトリを確認
    tensorboard_dir = latest_dir / "tensorboard"
    if tensorboard_dir.exists():
        return str(tensorboard_dir)
    
    return None


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='TensorBoard起動スクリプト')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='TensorBoardログディレクトリ（指定しない場合は最新の実験を自動検索）')
    parser.add_argument('--port', type=int, default=6006,
                       help='ポート番号（デフォルト: 6006）')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='ホスト名（デフォルト: 0.0.0.0）')
    parser.add_argument('--no-browser', action='store_true',
                       help='ブラウザを自動で開かない')
    parser.add_argument('--reload-interval', type=int, default=5,
                       help='リロード間隔（秒）（デフォルト: 5）')
    parser.add_argument('--experiments-dir', type=str, default="data/experiments",
                       help='実験ディレクトリ（デフォルト: data/experiments）')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # ログディレクトリの決定
    if args.log_dir is None:
        logger.info("最新の実験ディレクトリを検索しています...")
        log_dir = find_latest_experiment(args.experiments_dir)
        if log_dir is None:
            logger.error(f"実験ディレクトリが見つかりません: {args.experiments_dir}")
            logger.info("利用可能な実験ディレクトリ:")
            experiments_path = Path(args.experiments_dir)
            if experiments_path.exists():
                for exp_dir in experiments_path.iterdir():
                    if exp_dir.is_dir():
                        logger.info(f"  - {exp_dir}")
            return 1
        logger.info(f"最新の実験ディレクトリを使用: {log_dir}")
    else:
        log_dir = args.log_dir
    
    # TensorBoardを起動
    success = start_tensorboard(
        log_dir=log_dir,
        port=args.port,
        host=args.host,
        auto_open=not args.no_browser,
        reload_interval=args.reload_interval
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
