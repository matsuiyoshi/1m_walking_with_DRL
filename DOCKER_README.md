# Docker環境セットアップガイド

Bittle四足歩行ロボット深層強化学習プロジェクトのDocker環境構築と使用方法を説明します。

## 📋 前提条件

### システム要件
- **OS**: Linux (Ubuntu 20.04以降推奨) / WSL2
- **GPU**: NVIDIA GPU (CUDA対応)
- **メモリ**: 8GB以上 (16GB推奨)
- **ストレージ**: 10GB以上の空き容量

### 必要なソフトウェア
1. **Docker Engine** (20.10以降)
2. **Docker Compose** (v2.0以降)
3. **NVIDIA Container Toolkit**

## 🚀 セットアップ手順

### 1. NVIDIA Container Toolkitのインストール

```bash
# リポジトリの追加
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# パッケージのインストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Dockerの再起動
sudo systemctl restart docker
```

### 2. GPUアクセスの確認

```bash
# NVIDIA Container Toolkitのテスト
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

### 3. プロジェクトのクローン・移動

```bash
cd /path/to/1m_walking_with_DRL
```

## 🛠️ Docker環境の使用方法

### 基本的な起動

```bash
# 開発環境の起動
docker-compose up -d

# コンテナに接続
docker exec -it bittle-drl-container bash
```

### サービス別起動

```bash
# メイン開発環境のみ
docker-compose up bittle-drl

# TensorBoardも含めて起動
docker-compose --profile tensorboard up -d

# Jupyter Labも含めて起動
docker-compose --profile jupyter up -d

# 全サービス起動
docker-compose --profile tensorboard --profile jupyter up -d
```

## 🔧 主要な使用方法

### 1. 学習の実行

```bash
# コンテナ内で学習を開始
docker exec -it bittle-drl-container bash
cd /app
python scripts/train.py
```

### 2. Jupyter Notebook/Lab

```bash
# メインコンテナでJupyterを起動
docker exec -it bittle-drl-container jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# または専用サービスを使用
docker-compose --profile jupyter up -d
```

アクセス: http://localhost:8888 または http://localhost:8889

### 3. TensorBoard

```bash
# メインコンテナでTensorBoardを起動
docker exec -it bittle-drl-container tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# または専用サービスを使用
docker-compose --profile tensorboard up -d
```

アクセス: http://localhost:6006 または http://localhost:6007

### 4. 評価とテスト

```bash
# モデル評価
docker exec -it bittle-drl-container python scripts/evaluate.py

# テスト実行
docker exec -it bittle-drl-container pytest tests/
```

## 📁 ボリュームマウント

| ホストパス | コンテナパス | 説明 |
|-----------|-------------|------|
| `.` | `/app` | プロジェクトルート |
| `./data` | `/app/data` | データファイル |
| `./logs` | `/app/logs` | ログファイル |
| `bittle-models` | `/app/models` | 学習済みモデル |
| `bittle-experiments` | `/app/experiments` | 実験結果 |

## 🌐 ポート設定

| ポート | サービス | 説明 |
|--------|----------|------|
| 8888 | Jupyter Notebook | メインコンテナ |
| 8889 | Jupyter Lab | 専用サービス |
| 6006 | TensorBoard | メインコンテナ |
| 6007 | TensorBoard | 専用サービス |
| 8050 | Plotly Dash | 可視化ダッシュボード |
| 5000 | Flask API | 推論サーバー |

## 🐛 トラブルシューティング

### GPU が認識されない場合

```bash
# NVIDIA Container Toolkitの状態確認
sudo systemctl status nvidia-container-toolkit

# GPUの認識確認
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

### メモリ不足エラー

```bash
# Docker のメモリ設定を確認・変更
docker system info | grep "Total Memory"

# 不要なコンテナ・イメージの削除
docker system prune -a
```

### PyBullet GUIの表示問題

```bash
# X11フォワーディングの確認
echo $DISPLAY

# Xvfbプロセスの確認
docker exec -it bittle-drl-container pgrep Xvfb
```

## 🔄 開発ワークフロー

### 1. 新機能開発

```bash
# 新しいブランチで開発
git checkout -b feature/new-algorithm

# Docker環境で開発・テスト
docker-compose up -d
docker exec -it bittle-drl-container bash

# 開発完了後
git add .
git commit -m "Add new algorithm"
git push origin feature/new-algorithm
```

### 2. 実験管理

```bash
# 実験の実行
docker exec -it bittle-drl-container python scripts/train.py --config=experiments/exp1.yaml

# 結果の確認
docker exec -it bittle-drl-container tensorboard --logdir=./logs
```

### 3. モデルの保存・読み込み

```bash
# モデルの保存（永続化ボリュームに保存）
# /app/models/ に保存されたファイルは永続化されます

# モデルの共有
docker cp bittle-drl-container:/app/models/best_model.pth ./models/
```

## 📊 パフォーマンス最適化

### リソース使用量の監視

```bash
# コンテナのリソース使用量
docker stats bittle-drl-container

# GPUの使用量
docker exec -it bittle-drl-container nvidia-smi -l 1
```

### メモリ最適化

```bash
# PyTorchのメモリキャッシュクリア
docker exec -it bittle-drl-container python -c "import torch; torch.cuda.empty_cache()"
```

## 🔧 カスタマイズ

### 環境変数の変更

`docker-compose.yml`の`environment`セクションを編集：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用するGPUを指定
  - PYTHONPATH=/app/src:/app
  - WANDB_PROJECT=bittle-drl   # W&Bプロジェクト名
```

### 追加パッケージのインストール

```bash
# 一時的なインストール
docker exec -it bittle-drl-container pip install package_name

# 永続的なインストール（requirements.txtに追加）
echo "package_name>=version" >> requirements.txt
docker-compose build --no-cache
```

## 📚 参考資料

- [Docker公式ドキュメント](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [PyBullet公式ドキュメント](https://pybullet.org/wordpress/)
- [Stable-Baselines3ドキュメント](https://stable-baselines3.readthedocs.io/)

## 💬 サポート

問題が発生した場合は、以下の情報を含めてプロジェクトチームに連絡してください：

1. エラーメッセージ
2. Docker環境情報 (`docker version`, `docker-compose version`)
3. GPU情報 (`nvidia-smi`)
4. 実行したコマンド

---

**作成日**: 2025年1月  
**更新日**: 2025年1月  
**バージョン**: 1.0
