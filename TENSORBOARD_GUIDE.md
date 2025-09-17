# TensorBoard監視付き並列学習ガイド

このガイドでは、TensorBoardを使用して並列学習を監視する方法を説明します。

## 概要

並列学習システムには以下のTensorBoard監視機能が統合されています：

- **学習メトリクス**: 報酬、エピソード長、損失関数
- **パフォーマンス指標**: ステップ/秒、収集時間、バッファ使用率
- **学習パラメータ**: 学習率、勾配ノルム
- **分布情報**: 報酬分布のヒストグラム

## セットアップ

### 1. 必要なパッケージのインストール

```bash
pip install tensorboard
```

### 2. 並列学習の実行

```bash
# 基本的な並列学習
python scripts/parallel_train.py

# カスタム設定での並列学習
python scripts/parallel_train.py \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --batch-size 512 \
    --output-dir data/experiments/my_experiment
```

### 3. TensorBoardの起動

#### 方法1: Pythonスクリプトを使用

```bash
# 最新の実験を自動検索して起動
python scripts/start_tensorboard.py

# 特定の実験ディレクトリを指定
python scripts/start_tensorboard.py --log-dir data/experiments/my_experiment/tensorboard

# カスタムポートで起動
python scripts/start_tensorboard.py --port 6007 --host 0.0.0.0
```

#### 方法2: シェルスクリプトを使用

```bash
# 最新の実験を自動検索して起動
./scripts/start_tensorboard.sh

# 特定の実験ディレクトリを指定
./scripts/start_tensorboard.sh -d data/experiments/my_experiment/tensorboard

# カスタム設定で起動
./scripts/start_tensorboard.sh -p 6007 -h 0.0.0.0 -r 10
```

#### 方法3: 直接TensorBoardコマンドを使用

```bash
tensorboard --logdir=data/experiments/my_experiment/tensorboard --port=6006
```

## 監視可能なメトリクス

### Training タブ

#### 報酬関連
- **Average_Reward**: 平均報酬
- **Max_Reward**: 最大報酬
- **Min_Reward**: 最小報酬
- **Reward_Std**: 報酬の標準偏差
- **Reward_Distribution**: 報酬分布のヒストグラム

#### エピソード関連
- **Average_Length**: 平均エピソード長
- **Max_Length**: 最大エピソード長
- **Min_Length**: 最小エピソード長
- **Episode_Count**: エピソード数

#### パフォーマンス関連
- **Steps_Per_Second**: 1秒あたりのステップ数
- **Collection_Time**: 経験収集時間
- **Total_Steps**: 総ステップ数
- **Parallel_Envs**: 並列環境数

#### 学習関連
- **Policy_Loss**: ポリシー損失
- **Value_Loss**: 価値関数損失
- **Entropy_Loss**: エントロピー損失
- **Total_Loss**: 総損失
- **Learning_Rate**: 学習率
- **Gradient_Norm**: 勾配ノルム

#### システム関連
- **Buffer_Index**: バッファインデックス

## 実用的な使用方法

### 1. 学習開始前の準備

```bash
# ターミナル1: 並列学習を開始
python scripts/parallel_train.py --output-dir data/experiments/monitored_training

# ターミナル2: TensorBoardを起動
python scripts/start_tensorboard.py --log-dir data/experiments/monitored_training/tensorboard
```

### 2. リアルタイム監視

1. ブラウザで `http://localhost:6006` を開く
2. **SCALARS** タブで学習メトリクスを確認
3. **HISTOGRAMS** タブで報酬分布を確認
4. 必要に応じてグラフの表示範囲を調整

### 3. 学習の調整

TensorBoardで以下の指標を監視して学習を調整：

- **Average_Reward** が上昇しているか
- **Policy_Loss** と **Value_Loss** が適切に収束しているか
- **Gradient_Norm** が爆発していないか
- **Steps_Per_Second** でパフォーマンスを確認

### 4. 複数実験の比較

```bash
# 複数の実験ディレクトリを指定
tensorboard --logdir=data/experiments --port=6006
```

## トラブルシューティング

### TensorBoardが起動しない

```bash
# TensorBoardのインストール確認
pip list | grep tensorboard

# 再インストール
pip install --upgrade tensorboard
```

### ログディレクトリが見つからない

```bash
# 実験ディレクトリの確認
ls -la data/experiments/

# TensorBoardログディレクトリの確認
find data/experiments -name "tensorboard" -type d
```

### ポートが使用中

```bash
# 別のポートを使用
python scripts/start_tensorboard.py --port 6007
```

### リモートアクセス

```bash
# 全ホストからアクセス可能にする
python scripts/start_tensorboard.py --host 0.0.0.0 --port 6006
```

## 高度な設定

### カスタムメトリクスの追加

`parallel_trainer.py` の `_collect_experience` メソッドで追加のメトリクスをログできます：

```python
# カスタムメトリクスの例
self.writer.add_scalar('Custom/Metric_Name', metric_value, self.tensorboard_step)
```

### 画像のログ

```python
# 画像のログ例（必要に応じて）
self.writer.add_image('Environment/State', state_image, self.tensorboard_step)
```

## 実践的なノウハウ

### ログディレクトリの管理

#### 複数の実験を同時監視

```bash
# 方法1: 親ディレクトリを指定（全実験を自動検出）
tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0

# 方法2: 複数の特定ディレクトリを同時監視
tensorboard --logdir_spec=exp1:data/experiments/experiment1/tensorboard,exp2:data/experiments/experiment2/tensorboard --port 6006
```

#### 新しい学習を監視する際の注意点

**問題**: 新しい学習を開始したが、TensorBoardが古い学習のログを表示している

**解決方法**:
1. **停止→再起動**（シンプルで確実）
   ```bash
   # 古いTensorBoardを停止
   pkill -f tensorboard
   
   # 新しい学習用TensorBoardを起動
   python scripts/start_tensorboard.py --log-dir data/experiments/new_experiment/tensorboard
   ```

2. **親ディレクトリ指定**（推奨）
   ```bash
   # 全ての実験を監視（新しい実験も自動検出）
   tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0
   ```

3. **シンボリックリンク使用**
   ```bash
   # 新しいログを既存の監視ディレクトリにリンク
   ln -s data/experiments/new_experiment/tensorboard data/experiments/current_training
   ```

### `--no-browser`オプションの理解

```bash
# --no-browserオプションの効果
python scripts/start_tensorboard.py --log-dir path/to/logs --no-browser
```

**効果**:
- ✅ **TensorBoard自体は正常に起動**
- ✅ **ログの監視は正常に動作**
- ❌ **ブラウザが自動で開かない**

**注意点**:
- `--no-browser`は**ブラウザの自動起動を防ぐだけ**
- **ログディレクトリの指定には影響しない**
- 手動で`http://localhost:6006`にアクセスする必要がある

### Docker環境でのTensorBoard

#### 外部アクセス設定

```bash
# コンテナ内でTensorBoardを起動（外部アクセス可能）
docker exec bittle-drl-container python scripts/start_tensorboard.py \
  --log-dir data/experiments/current_experiment/tensorboard \
  --host 0.0.0.0 --port 6006
```

**重要なポイント**:
- `--host 0.0.0.0`で外部アクセスを許可
- `--host localhost`では外部からアクセス不可
- ポートマッピング（`-p 6006:6006`）がDocker Composeで設定されている必要がある

#### コンテナ内プロセスの確認

```bash
# TensorBoardプロセスの確認
docker exec bittle-drl-container ps aux | grep tensorboard

# 学習プロセスの確認
docker exec bittle-drl-container ps aux | grep parallel_train

# 全Pythonプロセスの確認
docker exec bittle-drl-container ps aux | grep python
```

### 学習中のリアルタイム監視

#### ログ間隔の調整

```yaml
# training_config.yaml
training:
  log_interval: 100  # 100ステップごとにログ出力（リアルタイム監視用）
  # log_interval: 1000  # 1000ステップごと（通常の監視用）
```

**推奨設定**:
- **リアルタイム監視**: `log_interval: 100`
- **通常監視**: `log_interval: 1000-5000`
- **長期学習**: `log_interval: 10000`

#### 並列環境でのログ出力

**注意**: 並列環境では`total_timesteps`が`num_envs`倍の速度で増加するため、ログ条件を調整する必要がある

```python
# 並列環境対応のログ条件
if self.total_timesteps // log_interval > (self.total_timesteps - collection_stats['total_steps']) // log_interval:
    # ログ出力
```

### トラブルシューティング（実践編）

#### TensorBoardがデータを表示しない

**症状**: "No dashboards are active for the current data set"

**原因と解決**:
1. **ログディレクトリが間違っている**
   ```bash
   # 正しいログディレクトリを確認
   find data/experiments -name "tensorboard" -type d
   ```

2. **ログ間隔が大きすぎる**
   ```yaml
   # 設定ファイルで調整
   log_interval: 100  # より小さな値に設定
   ```

3. **学習がまだ開始されていない**
   ```bash
   # 学習プロセスの確認
   docker exec bittle-drl-container ps aux | grep parallel_train
   ```

#### 複数のTensorBoardプロセスが競合

```bash
# 全てのTensorBoardプロセスを停止
docker exec bittle-drl-container pkill -f tensorboard

# 新しいTensorBoardを起動
docker exec bittle-drl-container python scripts/start_tensorboard.py --log-dir correct/path
```

#### ポート競合の解決

```bash
# 使用中のポートを確認
netstat -tulpn | grep 6006

# 別のポートを使用
python scripts/start_tensorboard.py --port 6007
```

### 効率的な監視ワークフロー

#### 1. 学習開始前の準備

```bash
# ターミナル1: 学習開始
docker exec bittle-drl-container python scripts/parallel_train.py --config config/training_config.yaml

# ターミナル2: TensorBoard起動（親ディレクトリ指定で全実験監視）
docker exec bittle-drl-container tensorboard --logdir data/experiments/ --port 6006 --host 0.0.0.0
```

#### 2. 学習中の監視ポイント

**重要なメトリクス**:
- **Episode Reward**: 学習の進捗
- **Policy Loss / Value Loss**: 学習の安定性
- **Steps/sec**: パフォーマンス
- **Gradient Norm**: 勾配爆発の検出

#### 3. 複数実験の比較

```bash
# 複数の実験を同時監視
tensorboard --logdir_spec=exp1:data/experiments/experiment1/tensorboard,exp2:data/experiments/experiment2/tensorboard
```

**比較ポイント**:
- 学習曲線の収束速度
- 最終的な報酬の到達値
- 学習の安定性（損失の振動）

## 学習の残り時間確認方法

### TensorBoardでの残り時間推定

TensorBoard自体には残り時間を直接表示する機能はありませんが、以下の手順で推定できます：

#### 1. 必要な指標の確認

**SCALARSタブ**で以下の指標を確認：
- **`global_step`** または **`steps`**: 現在のステップ数
- **`steps_per_second`**: 1秒あたりのステップ数

#### 2. 残り時間の計算

```
残り時間（秒） = (総ステップ数 - 現在のステップ数) ÷ 1秒あたりのステップ数
```

#### 3. 計算例

**設定例**:
- 総ステップ数: 1,800,000
- 現在のステップ数: 595,968
- 1秒あたりのステップ数: 736.2

**計算**:
```
残りステップ数 = 1,800,000 - 595,968 = 1,204,032
残り時間 = 1,204,032 ÷ 736.2 ≈ 1,635秒 ≈ 27分
```

#### 4. 実用的な監視方法

##### リアルタイム進捗確認
```bash
# 学習ログの確認
docker exec bittle-drl-container tail -1 /app/data/logs/parallel_training.log

# 5分ごとの進捗確認
watch -n 300 "docker exec bittle-drl-container tail -1 /app/data/logs/parallel_training.log"
```

##### TensorBoardでの視覚的確認
1. **SCALARSタブ**で`global_step`のグラフを確認
2. **傾き**で学習速度の変化を把握
3. **X軸**で現在の進捗率を確認

#### 5. 注意点

- **学習速度の変動**: 学習速度は時間とともに変動する可能性があるため、定期的に再計算することを推奨
- **並列環境の影響**: 並列環境数が多いほど学習速度が向上する
- **GPU使用率**: GPU使用率が100%に近い場合、学習速度が最適化されている

## 参考情報

- [TensorBoard公式ドキュメント](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard統合](https://pytorch.org/docs/stable/tensorboard.html)
- [並列学習のベストプラクティス](https://spinningup.openai.com/en/latest/user/algorithms.html#ppo)
