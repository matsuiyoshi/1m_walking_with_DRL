# トラブルシューティングガイド

このガイドでは、Bittle四足歩行ロボット深層強化学習プロジェクトで発生する可能性のある問題とその解決方法を説明します。

## 目次

1. [並列学習関連の問題](#並列学習関連の問題)
2. [TensorBoard関連の問題](#tensorboard関連の問題)
3. [Docker環境関連の問題](#docker環境関連の問題)
4. [GPU関連の問題](#gpu関連の問題)
5. [学習関連の問題](#学習関連の問題)
6. [評価・推論関連の問題](#評価推論関連の問題)
7. [システムリソース関連の問題](#システムリソース関連の問題)

## 並列学習関連の問題

### 1. テンソル形状エラー

#### 症状
```
RuntimeError: The size of tensor a (8) must match the size of tensor b (32) at non-singleton dimension 1
```

#### 原因
- 並列環境数とバッチサイズの不整合
- テンソルの次元が期待される形状と異なる

#### 解決方法
```bash
# 1. 設定ファイルの確認
cat config/training_config_1h.yaml | grep -E "(num_envs|batch_size|buffer_size)"

# 2. 推奨設定の適用
# num_envs: 12 (CPU コア数に合わせて)
# batch_size: 512 (num_envs × 32程度)
# buffer_size: 1536 (num_envs × 128程度)

# 3. デバッグ用設定でテスト
docker exec bittle-drl-container python scripts/parallel_train.py --config config/training_config_debug.yaml
```

### 2. メモリ不足エラー

#### 症状
```
RuntimeError: CUDA out of memory
```

#### 原因
- 並列環境数が多すぎる
- バッチサイズが大きすぎる
- GPU メモリの不足

#### 解決方法
```bash
# 1. 並列環境数を削減
# config/training_config_1h.yaml で num_envs を 8 または 4 に変更

# 2. バッチサイズを削減
# batch_size を 256 または 128 に変更

# 3. GPU メモリの確認
docker exec bittle-drl-container nvidia-smi

# 4. メモリキャッシュのクリア
docker exec bittle-drl-container python -c "import torch; torch.cuda.empty_cache()"
```

### 3. 学習が進まない

#### 症状
- 報酬が上昇しない
- エピソードが短いまま
- ロボットが動かない

#### 原因
- 報酬関数の設計問題
- 学習率が不適切
- 環境設定の問題

#### 解決方法
```bash
# 1. ログ間隔を小さくしてリアルタイム監視
# config/training_config_1h.yaml で log_interval: 100 に設定

# 2. TensorBoardで学習状況を確認
docker exec bittle-drl-container tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0

# 3. 学習率の調整
# learning_rate を 0.0001 または 0.001 に変更

# 4. デバッグ用設定でテスト
docker exec bittle-drl-container python scripts/parallel_train.py --config config/training_config_debug.yaml
```

## TensorBoard関連の問題

### 1. TensorBoardが起動しない

#### 症状
```
ModuleNotFoundError: No module named 'tensorboard'
```

#### 解決方法
```bash
# 1. TensorBoardのインストール確認
docker exec bittle-drl-container pip list | grep tensorboard

# 2. 再インストール
docker exec bittle-drl-container pip install --upgrade tensorboard

# 3. 直接起動
docker exec bittle-drl-container tensorboard --logdir=data/experiments --port=6006 --host=0.0.0.0
```

### 2. TensorBoardがデータを表示しない

#### 症状
```
No dashboards are active for the current data set
```

#### 原因
- ログディレクトリが間違っている
- ログ間隔が大きすぎる
- 学習がまだ開始されていない

#### 解決方法
```bash
# 1. ログディレクトリの確認
docker exec bittle-drl-container find data/experiments -name "tensorboard" -type d

# 2. 正しいログディレクトリを指定
docker exec bittle-drl-container tensorboard --logdir data/experiments/parallel_training/bittle_walking_XXXXXX/tensorboard --port 6006 --host 0.0.0.0

# 3. ログ間隔の調整
# config/training_config_1h.yaml で log_interval: 100 に設定

# 4. 学習プロセスの確認
docker exec bittle-drl-container ps aux | grep parallel_train
```

### 3. TensorBoardにアクセスできない

#### 症状
```
このサイトにアクセスできません
```

#### 原因
- ホスト設定が間違っている
- ポートが使用中
- ファイアウォールの設定

#### 解決方法
```bash
# 1. ホスト設定の確認
docker exec bittle-drl-container ps aux | grep tensorboard

# 2. 正しいホスト設定で再起動
docker exec bittle-drl-container pkill -f tensorboard
docker exec bittle-drl-container tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0

# 3. ポートの確認
netstat -tulpn | grep 6006

# 4. 別のポートを使用
docker exec bittle-drl-container tensorboard --logdir data/experiments/ --port 6007 --host 0.0.0.0
```

## Docker環境関連の問題

### 1. コンテナが起動しない

#### 症状
```
docker: Error response from daemon: failed to start container
```

#### 解決方法
```bash
# 1. Docker の状態確認
docker --version
docker-compose --version

# 2. コンテナの停止・削除
docker-compose down
docker system prune -a

# 3. イメージの再ビルド
docker-compose build --no-cache

# 4. 再起動
docker-compose up -d
```

### 2. ボリュームマウントエラー

#### 症状
```
Permission denied
```

#### 解決方法
```bash
# 1. ディレクトリの権限確認
ls -la data/

# 2. 権限の修正
chmod -R 755 data/
chown -R $USER:$USER data/

# 3. Docker の再起動
sudo systemctl restart docker
docker-compose up -d
```

### 3. コンテナ内でコマンドが実行できない

#### 症状
```
bash: command not found
```

#### 解決方法
```bash
# 1. コンテナの状態確認
docker-compose ps

# 2. コンテナに接続
docker-compose exec bittle-drl bash

# 3. 必要なパッケージのインストール
pip install package_name

# 4. コンテナの再ビルド
docker-compose build --no-cache
```

## GPU関連の問題

### 1. GPUが認識されない

#### 症状
```
CUDA available: False
```

#### 解決方法
```bash
# 1. NVIDIA Container Toolkitの確認
sudo systemctl status nvidia-container-toolkit

# 2. GPU の認識確認
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi

# 3. Docker daemon.jsonの確認
cat /etc/docker/daemon.json

# 4. 設定の追加（必要に応じて）
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

# 5. Docker の再起動
sudo systemctl restart docker
```

### 2. GPU メモリ不足

#### 症状
```
RuntimeError: CUDA out of memory
```

#### 解決方法
```bash
# 1. GPU メモリの確認
docker exec bittle-drl-container nvidia-smi

# 2. メモリキャッシュのクリア
docker exec bittle-drl-container python -c "import torch; torch.cuda.empty_cache()"

# 3. 並列環境数の削減
# config/training_config_1h.yaml で num_envs を 4 に変更

# 4. バッチサイズの削減
# batch_size を 128 に変更
```

### 3. GPU 使用率が低い

#### 症状
- GPU 使用率が 50% 以下
- 学習速度が遅い

#### 解決方法
```bash
# 1. 並列環境数の増加
# config/training_config_1h.yaml で num_envs を 12 に変更

# 2. バッチサイズの増加
# batch_size を 512 に変更

# 3. 学習プロセスの確認
docker exec bittle-drl-container ps aux | grep parallel_train

# 4. システムリソースの確認
docker exec bittle-drl-container htop
```

## 学習関連の問題

### 1. 学習が収束しない

#### 症状
- 報酬が上昇しない
- 損失が振動する
- 学習が不安定

#### 解決方法
```bash
# 1. 学習率の調整
# config/training_config_1h.yaml で learning_rate を 0.0001 に変更

# 2. バッチサイズの調整
# batch_size を 256 に変更

# 3. エポック数の調整
# n_epochs を 5 に変更

# 4. 勾配クリッピングの確認
# max_grad_norm を 0.5 に設定
```

### 2. 学習速度が遅い

#### 症状
- ステップ/秒が低い
- 学習に時間がかかる

#### 解決方法
```bash
# 1. 並列環境数の確認
# config/training_config_1h.yaml で num_envs を 12 に設定

# 2. バッチサイズの最適化
# batch_size を 512 に設定

# 3. ログ間隔の調整
# log_interval を 1000 に設定

# 4. システムリソースの確認
docker exec bittle-drl-container htop
```

### 3. 学習が途中で停止する

#### 症状
- 学習プロセスが終了する
- エラーメッセージが表示される

#### 解決方法
```bash
# 1. ログの確認
docker exec bittle-drl-container tail -50 /app/data/logs/parallel_training.log

# 2. システムリソースの確認
docker exec bittle-drl-container free -h
docker exec bittle-drl-container df -h

# 3. 学習プロセスの確認
docker exec bittle-drl-container ps aux | grep parallel_train

# 4. 設定の調整
# 並列環境数やバッチサイズを削減
```

## 評価・推論関連の問題

### 1. モデル評価が失敗する

#### 症状
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/models/final_model.pth'
```

#### 解決方法
```bash
# 1. モデルファイルの確認
docker exec bittle-drl-container find data/ -name "*.pth" -type f

# 2. 正しいパスを指定
docker exec bittle-drl-container python scripts/evaluate.py --model-path data/experiments/XXXXXX/final_model.pth

# 3. モデルの保存確認
docker exec bittle-drl-container ls -la data/experiments/*/final_model.pth
```

### 2. 評価動画が生成されない

#### 症状
- 動画ファイルが作成されない
- 評価は成功するが動画がない

#### 解決方法
```bash
# 1. 動画保存オプションの確認
docker exec bittle-drl-container python scripts/evaluate.py --model-path data/models/final_model.pth --save-video

# 2. 出力ディレクトリの確認
docker exec bittle-drl-container ls -la data/evaluations/

# 3. 権限の確認
docker exec bittle-drl-container ls -la data/evaluations/
```

### 3. 推論速度が遅い

#### 症状
- 推論に時間がかかる
- リアルタイム制御ができない

#### 解決方法
```bash
# 1. モデルの軽量化
# ネットワークサイズを削減

# 2. バッチ推論の使用
# 複数の推論を同時実行

# 3. CPU 最適化
# モデルをCPU用に最適化
```

## システムリソース関連の問題

### 1. メモリ不足

#### 症状
```
Out of memory
```

#### 解決方法
```bash
# 1. メモリ使用量の確認
docker exec bittle-drl-container free -h

# 2. 不要なプロセスの停止
docker exec bittle-drl-container pkill -f python

# 3. 並列環境数の削減
# config/training_config_1h.yaml で num_envs を 4 に変更

# 4. バッチサイズの削減
# batch_size を 128 に変更
```

### 2. ディスク容量不足

#### 症状
```
No space left on device
```

#### 解決方法
```bash
# 1. ディスク使用量の確認
docker exec bittle-drl-container df -h

# 2. 不要なファイルの削除
docker exec bittle-drl-container find data/ -name "*.log" -mtime +7 -delete

# 3. Docker のクリーンアップ
docker system prune -a

# 4. 古い実験データの削除
docker exec bittle-drl-container find data/experiments/ -mtime +30 -type d -exec rm -rf {} +
```

### 3. CPU 使用率が高い

#### 症状
- システムが重い
- 他のプロセスが遅い

#### 解決方法
```bash
# 1. CPU 使用率の確認
docker exec bittle-drl-container htop

# 2. 並列環境数の調整
# config/training_config_1h.yaml で num_envs を 8 に変更

# 3. プロセスの優先度調整
# nice コマンドで優先度を下げる

# 4. システムリソースの監視
watch -n 1 'docker exec bittle-drl-container htop'
```

## 緊急時の対応

### 学習の緊急停止

```bash
# 1. 学習プロセスの停止
docker exec bittle-drl-container pkill -f parallel_train

# 2. 全Pythonプロセスの停止
docker exec bittle-drl-container pkill -f python

# 3. コンテナの再起動
docker-compose restart bittle-drl
```

### システムのリセット

```bash
# 1. 全コンテナの停止
docker-compose down

# 2. システムのクリーンアップ
docker system prune -a

# 3. 再起動
docker-compose up -d
```

### データのバックアップ

```bash
# 1. 重要なデータのバックアップ
cp -r data/experiments/ backup/experiments_$(date +%Y%m%d_%H%M%S)/

# 2. モデルのバックアップ
cp data/models/*.pth backup/models_$(date +%Y%m%d_%H%M%S)/

# 3. 設定ファイルのバックアップ
cp config/*.yaml backup/config_$(date +%Y%m%d_%H%M%S)/
```

## 予防策

### 定期的なメンテナンス

```bash
# 1. 週次メンテナンス
docker system prune -a
docker exec bittle-drl-container find data/ -name "*.log" -mtime +7 -delete

# 2. 月次メンテナンス
docker exec bittle-drl-container find data/experiments/ -mtime +30 -type d -exec rm -rf {} +

# 3. システムリソースの監視
watch -n 300 'docker exec bittle-drl-container free -h && docker exec bittle-drl-container df -h'
```

### 設定の最適化

```bash
# 1. 並列環境数の最適化
# CPU コア数に合わせて num_envs を設定

# 2. バッチサイズの最適化
# GPU メモリに合わせて batch_size を設定

# 3. ログ間隔の最適化
# 監視頻度に合わせて log_interval を設定
```

## サポート

問題が解決しない場合は、以下の情報を含めてプロジェクトチームに連絡してください：

1. **エラーメッセージ**: 完全なエラーメッセージ
2. **システム情報**: OS、Docker バージョン、GPU 情報
3. **実行コマンド**: 実行したコマンド
4. **ログファイル**: 関連するログファイル
5. **設定ファイル**: 使用した設定ファイル

---

**作成日**: 2025年9月  
**更新日**: 2025年9月  
**バージョン**: 1.0
