# Bittle四足歩行ロボット深層強化学習プロジェクト

## 概要
Petoi社のBittle四足歩行ロボットを深層強化学習（DRL）で制御し、幅15cmの通路を1m直進する歩行を実現するプロジェクトです。

## 🚀 クイックスタート

### Docker環境での開発（推奨）
```bash
# リポジトリをクローン
git clone <repository-url>
cd 1m_walking_with_DRL

# Docker環境を起動
docker-compose up -d

# コンテナに接続
docker exec -it bittle-drl-container bash

# 学習を開始
python scripts/train.py
```

詳細な手順は [DOCKER_README.md](./DOCKER_README.md) をご参照ください。

## 📁 プロジェクト構成
- `src/` - メインソースコード
- `config/` - 設定ファイル
- `scripts/` - 実行スクリプト
- `assets/` - URDFファイル等のアセット
- `data/` - データ保存用ディレクトリ
- `logs/` - ログファイル

## 📖 ドキュメント
- [プロジェクト仕様書](./PROJECT_SPECIFICATION.md)
- [Docker環境セットアップ](./DOCKER_README.md)
