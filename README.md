# Bittle四足歩行ロボット深層強化学習プロジェクト

## 概要
Petoi社のBittle四足歩行ロボットを深層強化学習（DRL）で制御し、幅15cmの通路を1m直進する歩行を実現するプロジェクトです。

## 🚀 クイックスタート

### Docker環境での開発（推奨）

#### 1. 前提条件
- Docker Engine 20.10以上
- NVIDIA Container Toolkit（GPU使用時）
- NVIDIA Driver 470以上（GPU使用時）

#### 2. セットアップ
```bash
# リポジトリをクローン
git clone <repository-url>
cd 1m_walking_with_DRL

# Docker環境を起動
docker compose up -d

# コンテナに接続
docker compose exec bittle-drl bash
```

#### 3. 学習の実行
```bash
# 基本学習
python scripts/train.py

# 並列学習（GPU最適化）
python scripts/parallel_train.py --total-timesteps 10000 --num-envs 4 --batch-size 128

# モデル評価
python scripts/evaluate.py --model-path data/models/final_model.pth --render
```

#### 4. TensorBoard監視
```bash
# TensorBoardを起動（別ターミナル）
docker compose exec bittle-drl tensorboard --logdir=data/experiments --port=6006 --host=0.0.0.0

# ブラウザでアクセス
# http://localhost:6006
```

詳細な手順は [DOCKER_README.md](./DOCKER_README.md) をご参照ください。

## 🐳 Docker環境の詳細

### GPU対応の確認
```bash
# GPU認識確認
docker compose exec bittle-drl nvidia-smi

# PyTorch CUDA確認
docker compose exec bittle-drl python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### よく使用するDockerコマンド
```bash
# コンテナの状態確認
docker compose ps

# ログの確認
docker compose logs bittle-drl

# コンテナの再起動
docker compose restart bittle-drl

# コンテナの停止
docker compose down

# イメージの再ビルド
docker compose build --no-cache
```

### トラブルシューティング
- **GPU認識されない場合**: NVIDIA Container Toolkitのインストールを確認
- **メモリ不足**: `docker-compose.yml`のメモリ制限を調整
- **ポート競合**: `docker-compose.yml`のポート設定を変更

## 📁 プロジェクト構成
- `src/` - メインソースコード
  - `environment/` - 環境実装（BittleWalkingEnv, ParallelEnv）
  - `models/` - ニューラルネットワーク（PPO Agent）
  - `training/` - 学習器（Trainer, ParallelTrainer）
- `config/` - 設定ファイル
  - `training_config.yaml` - 学習パラメータ
  - `env_config.yaml` - 環境設定
  - `bittle_config.yaml` - ロボット設定
- `scripts/` - 実行スクリプト
  - `train.py` - 基本学習
  - `parallel_train.py` - 並列学習
  - `evaluate.py` - モデル評価
- `assets/` - URDFファイル等のアセット
- `data/` - データ保存用ディレクトリ
  - `experiments/` - 学習実験結果
  - `models/` - 保存されたモデル
  - `evaluations/` - 評価結果と動画
- `logs/` - ログファイル

## 📊 学習監視

### TensorBoardで監視できる指標
- **Training**: 平均報酬、エピソード数、学習速度
- **Loss**: ポリシー損失、価値関数損失、エントロピー損失
- **Performance**: ステップ/秒、バッファ使用率

### 並列学習の最適化

#### 基本的な並列学習実行
```bash
# 設定ファイルを使用（推奨）
python scripts/parallel_train.py --config config/training_config_1h.yaml

# コマンドライン引数で指定
python scripts/parallel_train.py \
  --total-timesteps 300000 \
  --num-envs 12 \
  --batch-size 256 \
  --learning-rate 0.0004
```

#### 並列学習のパラメータ調整
- **環境数**: 2-16環境（CPU コア数に応じて調整）
  - 12コアCPU: 推奨12環境
  - 8コアCPU: 推奨8環境
  - 4コアCPU: 推奨4環境
- **バッチサイズ**: 64-512（並列環境数に応じて調整）
  - 推奨: `num_envs × 32`
- **バッファサイズ**: 1024-2048（並列環境数に応じて調整）
  - 推奨: `num_envs × 128`
- **学習率**: 0.0004-0.001（並列学習用に最適化）

#### 並列学習の監視
```bash
# TensorBoardでリアルタイム監視
python scripts/start_tensorboard.py --log-dir data/experiments/current_experiment/tensorboard

# 学習プロセスの確認
docker exec bittle-drl-container ps aux | grep parallel_train

# GPU使用状況の確認
nvidia-smi
```

#### TensorBoardでの残り時間確認
TensorBoardのGUI上で学習の残り時間を確認する方法：

1. **SCALARSタブ**で以下の指標を確認：
   - `global_step`: 現在のステップ数
   - `steps_per_second`: 1秒あたりのステップ数

2. **残り時間の計算**：
   ```
   残り時間（秒） = (総ステップ数 - 現在のステップ数) ÷ 1秒あたりのステップ数
   ```

3. **例**：
   - 総ステップ数: 1,800,000
   - 現在のステップ数: 595,968
   - 1秒あたりのステップ数: 736.2
   - 残り時間: (1,800,000 - 595,968) ÷ 736.2 ≈ 1,635秒 ≈ 27分

#### 並列学習のトラブルシューティング
- **メモリ不足**: 環境数やバッチサイズを削減
- **テンソル形状エラー**: バッファサイズとバッチサイズの整合性確認
- **学習が進まない**: ログ間隔を小さくしてリアルタイム監視

## 📖 ドキュメント
- [プロジェクト仕様書](./PROJECT_SPECIFICATION.md)
- [Docker環境セットアップ](./DOCKER_README.md)
- [TensorBoard監視ガイド](./TENSORBOARD_GUIDE.md)
- [トラブルシューティングガイド](./TROUBLESHOOTING.md)
