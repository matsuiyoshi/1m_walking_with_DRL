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
docker compose up -d

# コンテナに接続
docker compose exec bittle-drl bash
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
# 基本学習
docker compose exec bittle-drl python scripts/train.py

# 並列学習（GPU最適化）
docker compose exec bittle-drl python scripts/parallel_train.py --total-timesteps 10000 --num-envs 4 --batch-size 128

# モデル評価
docker compose exec bittle-drl python scripts/evaluate.py --model-path data/models/final_model.pth --render
```

### 2. TensorBoard監視

```bash
# TensorBoardを起動（別ターミナル）
docker compose exec bittle-drl tensorboard --logdir=data/experiments --port=6006 --host=0.0.0.0

# ブラウザでアクセス
# http://localhost:6006
```

**監視できる指標:**
- **Training**: 平均報酬、エピソード数、学習速度
- **Loss**: ポリシー損失、価値関数損失、エントロピー損失
- **Performance**: ステップ/秒、バッファ使用率

### 3. Jupyter Notebook/Lab

```bash
# メインコンテナでJupyterを起動
docker compose exec bittle-drl jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# または専用サービスを使用
docker compose --profile jupyter up -d
```

アクセス: http://localhost:8888 または http://localhost:8889

### 4. 評価とテスト

```bash
# モデル評価
docker compose exec bittle-drl python scripts/evaluate.py

# テスト実行
docker compose exec bittle-drl pytest tests/
```

### 5. GPU使用状況の確認

```bash
# GPU認識確認
docker compose exec bittle-drl nvidia-smi

# PyTorch CUDA確認
docker compose exec bittle-drl python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU使用率の監視
docker compose exec bittle-drl nvidia-smi -l 1
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

# Docker daemon.jsonの確認
cat /etc/docker/daemon.json

# 必要に応じて設定を追加
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
EOF

# Dockerの再起動
sudo systemctl restart docker
```

### メモリ不足エラー

```bash
# Docker のメモリ設定を確認・変更
docker system info | grep "Total Memory"

# 不要なコンテナ・イメージの削除
docker system prune -a

# コンテナのリソース使用量確認
docker stats bittle-drl

# PyTorchのメモリキャッシュクリア
docker compose exec bittle-drl python -c "import torch; torch.cuda.empty_cache()"
```

### 並列学習でのエラー

```bash
# CUDA tensorのnumpy変換エラー
# 解決策: 環境設定でCUDA tensorをCPUに移動してからnumpyに変換

# テンソル形状エラー
# 解決策: バッチサイズと並列環境数の調整
docker compose exec bittle-drl python scripts/parallel_train.py --num-envs 2 --batch-size 64
```

### PyBullet GUIの表示問題

```bash
# X11フォワーディングの確認
echo $DISPLAY

# Xvfbプロセスの確認
docker compose exec bittle-drl pgrep Xvfb

# ヘッドレスモードでの実行
docker compose exec bittle-drl python scripts/train.py --headless
```

## 🔄 開発ワークフロー

### 1. 新機能開発

```bash
# 新しいブランチで開発
git checkout -b feature/new-algorithm

# Docker環境で開発・テスト
docker compose up -d
docker compose exec bittle-drl bash

# 開発完了後
git add .
git commit -m "Add new algorithm"
git push origin feature/new-algorithm
```

### 2. 実験管理

```bash
# 実験の実行
docker compose exec bittle-drl python scripts/train.py --config=experiments/exp1.yaml

# 並列学習実験
docker compose exec bittle-drl python scripts/parallel_train.py --total-timesteps 50000 --num-envs 4

# 結果の確認
docker compose exec bittle-drl tensorboard --logdir=data/experiments
```

### 3. モデルの保存・読み込み

```bash
# モデルの保存（永続化ボリュームに保存）
# /app/data/models/ に保存されたファイルは永続化されます

# モデルの共有
docker cp bittle-drl:/app/data/models/best_model.pth ./data/models/

# 評価用モデルの読み込み
docker compose exec bittle-drl python scripts/evaluate.py --model-path data/models/final_model.pth
```

## 📊 パフォーマンス最適化

### リソース使用量の監視

```bash
# コンテナのリソース使用量
docker stats bittle-drl

# GPUの使用量
docker compose exec bittle-drl nvidia-smi -l 1

# リアルタイム監視
watch -n 1 'docker stats bittle-drl --no-stream'
```

### メモリ最適化

```bash
# PyTorchのメモリキャッシュクリア
docker compose exec bittle-drl python -c "import torch; torch.cuda.empty_cache()"

# 並列学習の最適化設定
# 環境数: 2-8環境（GPU メモリに応じて調整）
# バッチサイズ: 64-256（並列環境数に応じて調整）
# 学習率: 0.001（並列学習用に最適化）
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
docker compose exec bittle-drl pip install package_name

# 永続的なインストール（requirements.txtに追加）
echo "package_name>=version" >> requirements.txt
docker compose build --no-cache
```

### 並列学習の設定調整

#### 設定ファイルでの並列学習
```bash
# 設定ファイルの編集
docker compose exec bittle-drl vim config/training_config.yaml

# 並列学習パラメータの調整例
# num_envs: 12  # 並列環境数（CPU コア数に合わせて調整）
# batch_size: 256  # バッチサイズ（num_envs × 32程度）
# buffer_size: 1536  # バッファサイズ（num_envs × 128程度）
# learning_rate: 0.0004  # 学習率（並列学習用に最適化）
```

#### 並列学習の実行と監視
```bash
# 1時間学習の実行
docker compose exec bittle-drl python scripts/parallel_train.py --config config/training_config_1h.yaml

# 並列学習の監視（別ターミナル）
docker compose exec bittle-drl python scripts/start_tensorboard.py --log-dir data/experiments/current_experiment/tensorboard

# 学習プロセスの確認
docker compose exec bittle-drl ps aux | grep parallel_train

# GPU使用状況の確認
docker compose exec bittle-drl nvidia-smi
```

#### 詳細な並列学習監視方法

##### 1. リアルタイム進捗監視
```bash
# 学習ログのリアルタイム確認
docker compose exec bittle-drl tail -f /app/data/logs/parallel_training.log

# 5分ごとの進捗確認
watch -n 300 "docker compose exec bittle-drl tail -1 /app/data/logs/parallel_training.log"

# 学習プロセス数の確認
docker compose exec bittle-drl ps aux | grep -c parallel_train
```

##### 2. TensorBoardでの残り時間確認
```bash
# TensorBoardを起動（親ディレクトリ指定で全実験監視）
docker compose exec bittle-drl tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0

# ブラウザで http://localhost:6006 にアクセス
# SCALARSタブで以下を確認：
# - global_step: 現在のステップ数
# - steps_per_second: 学習速度
# 残り時間 = (総ステップ数 - 現在ステップ数) ÷ 学習速度
```

##### 3. リソース使用量監視
```bash
# コンテナのリソース使用量
docker stats bittle-drl --no-stream

# GPU使用率の監視
docker compose exec bittle-drl nvidia-smi -l 1

# メモリ使用量の確認
docker compose exec bittle-drl free -h
```

##### 4. 学習の完了確認
```bash
# 学習プロセスの確認
docker compose exec bittle-drl ps aux | grep parallel_train

# 最終ログの確認
docker compose exec bittle-drl tail -5 /app/data/logs/parallel_training.log

# 保存されたモデルの確認
docker compose exec bittle-drl ls -la /app/data/experiments/*/final_model.pth
```

#### 並列学習の最適化ガイドライン
- **CPU コア数との関係**: 並列環境数はCPU コア数以下に設定
- **メモリ使用量**: 環境数が多いほどメモリ使用量が増加
- **学習効率**: 適切な並列数で学習速度が向上
- **安定性**: 過度な並列化は学習の不安定化を招く可能性

#### 並列学習のトラブルシューティング
```bash
# メモリ不足の場合
docker compose exec bittle-drl python scripts/parallel_train.py --num-envs 4 --batch-size 64

# テンソル形状エラーの場合
# バッファサイズとバッチサイズの整合性を確認

# 学習が進まない場合
# ログ間隔を小さくしてリアルタイム監視
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
**更新日**: 2025年9月  
**バージョン**: 2.0
