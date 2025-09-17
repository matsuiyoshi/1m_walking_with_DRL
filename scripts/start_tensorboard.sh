#!/bin/bash
# TensorBoard起動スクリプト（シェル版）

# デフォルト値
PORT=6006
HOST="localhost"
LOG_DIR=""
EXPERIMENTS_DIR="data/experiments"
NO_BROWSER=false
RELOAD_INTERVAL=5

# ヘルプ表示
show_help() {
    echo "TensorBoard起動スクリプト"
    echo ""
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -d, --log-dir DIR        TensorBoardログディレクトリ"
    echo "  -p, --port PORT          ポート番号 (デフォルト: 6006)"
    echo "  -h, --host HOST          ホスト名 (デフォルト: localhost)"
    echo "  -e, --experiments-dir DIR 実験ディレクトリ (デフォルト: data/experiments)"
    echo "  -n, --no-browser         ブラウザを自動で開かない"
    echo "  -r, --reload-interval SEC リロード間隔 (デフォルト: 5秒)"
    echo "  --help                   このヘルプを表示"
    echo ""
    echo "例:"
    echo "  $0                                    # 最新の実験を自動検索"
    echo "  $0 -d data/experiments/my_exp/tensorboard  # 特定のディレクトリを指定"
    echo "  $0 -p 6007 -h 0.0.0.0                # ポート6007、全ホストからアクセス可能"
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -e|--experiments-dir)
            EXPERIMENTS_DIR="$2"
            shift 2
            ;;
        -n|--no-browser)
            NO_BROWSER=true
            shift
            ;;
        -r|--reload-interval)
            RELOAD_INTERVAL="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_help
            exit 1
            ;;
    esac
done

# プロジェクトルートに移動
cd "$(dirname "$0")/.."

# TensorBoardがインストールされているかチェック
if ! command -v tensorboard &> /dev/null; then
    echo "エラー: TensorBoardがインストールされていません"
    echo "インストールしてください: pip install tensorboard"
    exit 1
fi

# ログディレクトリの決定
if [ -z "$LOG_DIR" ]; then
    echo "最新の実験ディレクトリを検索しています..."
    
    if [ ! -d "$EXPERIMENTS_DIR" ]; then
        echo "エラー: 実験ディレクトリが存在しません: $EXPERIMENTS_DIR"
        exit 1
    fi
    
    # 最新の実験ディレクトリを検索
    LATEST_EXP=$(find "$EXPERIMENTS_DIR" -maxdepth 1 -type d -name "*" | grep -v "^$EXPERIMENTS_DIR$" | sort | tail -1)
    
    if [ -z "$LATEST_EXP" ]; then
        echo "エラー: 実験ディレクトリが見つかりません: $EXPERIMENTS_DIR"
        echo "利用可能な実験ディレクトリ:"
        ls -la "$EXPERIMENTS_DIR"
        exit 1
    fi
    
    # TensorBoardディレクトリを確認
    if [ -d "$LATEST_EXP/tensorboard" ]; then
        LOG_DIR="$LATEST_EXP/tensorboard"
        echo "最新の実験ディレクトリを使用: $LOG_DIR"
    else
        echo "エラー: TensorBoardログディレクトリが見つかりません: $LATEST_EXP/tensorboard"
        exit 1
    fi
fi

# ログディレクトリの存在確認
if [ ! -d "$LOG_DIR" ]; then
    echo "エラー: ログディレクトリが存在しません: $LOG_DIR"
    exit 1
fi

# TensorBoardコマンドの構築
TB_CMD="tensorboard --logdir=\"$LOG_DIR\" --port=$PORT --host=$HOST --reload_interval=$RELOAD_INTERVAL --max_reload_threads=4"

echo "TensorBoardを起動しています..."
echo "ログディレクトリ: $LOG_DIR"
echo "URL: http://$HOST:$PORT"
echo "コマンド: $TB_CMD"
echo ""

# TensorBoardを起動
eval $TB_CMD
