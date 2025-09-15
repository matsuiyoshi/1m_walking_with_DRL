#!/bin/bash

# Bittle DRL Project Docker Entrypoint Script

set -e

# 環境変数の確認
echo "=== Bittle DRL Environment Setup ==="
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'CUDA not available')"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not installed')"
echo "PyBullet Available: $(python -c 'import pybullet; print("Yes")' 2>/dev/null || echo 'No')"
echo "CUDA Available in PyTorch: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo "======================================"

# ディレクトリの作成
mkdir -p /app/data/models
mkdir -p /app/data/logs
mkdir -p /app/data/experiments
mkdir -p /app/logs/tensorboard

# 権限の設定
chmod -R 755 /app/scripts/

# Xvfbの起動（PyBulletのGUI用）
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting Xvfb for PyBullet GUI..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    export DISPLAY=:99
fi

# GPU情報の表示
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi
    echo "======================="
fi

# 渡されたコマンドを実行
exec "$@"
