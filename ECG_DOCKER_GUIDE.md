# 12誘導心電図生成モデル - Docker環境セットアップガイド

このガイドでは、Docker環境でEDMフレームワークを使用して12誘導心電図の生成モデルを学習する手順を説明します。

## 📋 前提条件

- **Docker**: 20.10以降
- **NVIDIA Docker Runtime**: GPUサポート用
- **NVIDIA Driver**: r520以降
- **GPU**: NVIDIA GPU 1台以上（学習には8台以上推奨）

### NVIDIA Docker Runtimeの確認

```bash
# Dockerでのアクセスを確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 クイックスタート（Docker版）

### ステップ1: Dockerイメージのビルド

```bash
# リポジトリのルートディレクトリで実行
cd /home/user/edm

# Dockerイメージをビルド（初回のみ、数分かかります）
docker build --tag edm-ecg:latest .
```

### ステップ2: テストデータセットの作成

```bash
# データセット用ディレクトリを作成
mkdir -p datasets/ecg_test

# Dockerコンテナでダミーデータを生成
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py create_dummy \
        --output_dir datasets/ecg_test \
        --num_samples 100

# データセットを検証
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py verify \
        --dataset_dir datasets/ecg_test
```

### ステップ3: テスト学習の実行

```bash
# 単一GPUでテスト学習（数分で完了）
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python train.py \
        --outdir=training-runs \
        --data=datasets/ecg_test \
        --cond=0 \
        --arch=ddpmpp \
        --duration=1 \
        --batch=32 \
        --batch-gpu=32 \
        --tick=1 \
        --snap=5 \
        --seed=42
```

### ステップ4: 心電図の生成

```bash
# 学習したモデルから心電図を生成
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python generate_ecg.py \
        --network training-runs/00000-ecg_test-uncond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-000005.pkl \
        --outdir generated_ecg \
        --num_samples 5 \
        --steps 18
```

生成された心電図は`generated_ecg/`ディレクトリに保存されます。

---

## 📊 実データでの学習（Docker版）

### 1. データの準備

ホストマシン上でデータを準備します：

```bash
# データディレクトリを作成
mkdir -p raw_data

# あなたの心電図データをコピー
# 各ファイルは shape=(12, 1000) のnumpy配列
cp /path/to/your/ecg/*.npy raw_data/
```

### 2. データの前処理

```bash
# Dockerコンテナでデータを前処理
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py preprocess \
        --input_dir raw_data \
        --output_dir datasets/ecg_real \
        --normalize minmax

# 検証
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py verify \
        --dataset_dir datasets/ecg_real
```

### 3. 本格的な学習（複数GPU）

```bash
# 8GPUでの本格学習
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=8g \
    edm-ecg:latest \
    torchrun --standalone --nproc_per_node=8 train.py \
        --outdir=training-runs \
        --data=datasets/ecg_real \
        --cond=0 \
        --arch=ddpmpp \
        --precond=edm \
        --duration=200 \
        --batch=512 \
        --batch-gpu=64 \
        --lr=0.0001 \
        --ema=0.5 \
        --dropout=0.1 \
        --augment=0.0 \
        --tick=50 \
        --snap=50 \
        --seed=42
```

**重要なDockerオプション:**
- `--gpus all`: すべてのGPUを使用
- `-v $(pwd):/workspace`: 現在のディレクトリをコンテナにマウント
- `--shm-size=8g`: 共有メモリサイズを増やす（複数GPU時に必要）

---

## 🔧 便利なDockerコマンド

### インタラクティブモードで起動

```bash
# コンテナ内でシェルを起動
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=8g \
    edm-ecg:latest \
    bash

# コンテナ内で直接コマンドを実行
python preprocess_ecg_data.py --help
python train.py --help
python generate_ecg.py --help
```

### バックグラウンドで学習を実行

```bash
# デタッチモードで学習を実行
docker run -d --name ecg-training \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=8g \
    edm-ecg:latest \
    python train.py \
        --outdir=training-runs \
        --data=datasets/ecg_real \
        --cond=0 \
        --arch=ddpmpp \
        --duration=200 \
        --batch=512

# ログを確認
docker logs -f ecg-training

# 学習を停止
docker stop ecg-training

# コンテナを削除
docker rm ecg-training
```

### 学習の進捗を監視

```bash
# ログファイルをリアルタイムで監視（ホスト側から）
tail -f training-runs/00000-*/log.txt

# または、コンテナ内から
docker exec ecg-training tail -f /workspace/training-runs/00000-*/log.txt
```

---

## 🛠️ Docker Composeの使用（推奨）

より便利な管理のため、Docker Composeを使用できます。

`docker-compose.yml`を作成：

```yaml
version: '3.8'

services:
  edm-ecg:
    build: .
    image: edm-ecg:latest
    container_name: ecg-training
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
    working_dir: /workspace
    shm_size: 8gb
    stdin_open: true
    tty: true
```

使用方法：

```bash
# イメージをビルド
docker-compose build

# インタラクティブモードで起動
docker-compose run --rm edm-ecg bash

# 学習を実行
docker-compose run --rm edm-ecg python train.py \
    --outdir=training-runs \
    --data=datasets/ecg_test \
    --cond=0 \
    --arch=ddpmpp \
    --duration=1 \
    --batch=32

# 心電図を生成
docker-compose run --rm edm-ecg python generate_ecg.py \
    --network training-runs/00000-*/network-snapshot-*.pkl \
    --outdir generated_ecg
```

---

## 📁 ディレクトリ構成（Docker版）

```
edm/  (ホストマシン)
├── Dockerfile                  # Dockerイメージ定義
├── docker-compose.yml          # Docker Compose設定（オプション）
├── ECG_DOCKER_GUIDE.md         # このファイル
├── ECG_TRAINING_GUIDE.md       # 詳細ガイド
├── preprocess_ecg_data.py      # データ前処理
├── generate_ecg.py             # 心電図生成
├── training/
│   └── ecg_dataset.py          # ECGデータセット
│
├── raw_data/                   # 生データ（前処理前）
│   ├── sample_001.npy
│   └── ...
│
├── datasets/                   # 前処理済みデータ
│   ├── ecg_test/              # テスト用
│   └── ecg_real/              # 実データ
│
├── training-runs/              # 学習結果
│   └── 00000-*/
│       ├── network-snapshot-*.pkl
│       ├── log.txt
│       └── stats.jsonl
│
└── generated_ecg/              # 生成された心電図
    ├── ecg_0000.npy
    ├── ecg_0000.png
    └── ...
```

**重要**: すべてのディレクトリはホストマシンとコンテナ間で共有されます（`-v $(pwd):/workspace`）。

---

## 🐛 トラブルシューティング（Docker版）

### 1. GPU が認識されない

```bash
# NVIDIA Dockerランタイムの確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# エラーが出る場合はNVIDIA Container Runtimeをインストール
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 2. 権限エラー（Permission denied）

```bash
# ユーザーIDを指定して実行
docker run --rm --gpus all \
    --user $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py create_dummy --output_dir datasets/ecg_test --num_samples 100
```

### 3. 共有メモリ不足（Multi-GPU学習時）

```bash
# --shm-size を増やす
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    --shm-size=16g \  # 8g → 16gに増やす
    edm-ecg:latest \
    ...
```

### 4. ディスク容量不足

```bash
# 不要なDockerイメージ・コンテナを削除
docker system prune -a

# ビルドキャッシュをクリア
docker builder prune
```

### 5. OOM (Out of Memory) エラー

```bash
# batch-gpu を減らす
--batch-gpu=32  # 64 → 32に減らす

# または fp16 を有効化
--fp16=True
```

---

## 📊 パフォーマンス最適化（Docker版）

### マルチノード学習（複数マシン）

複数のマシンで分散学習を行う場合：

**マスターノード:**
```bash
docker run --rm --gpus all \
    --network host \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=16g \
    edm-ecg:latest \
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=192.168.1.100 \
        --master_port=29500 \
        train.py --outdir=training-runs --data=datasets/ecg_real --batch=1024
```

**ワーカーノード:**
```bash
docker run --rm --gpus all \
    --network host \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=16g \
    edm-ecg:latest \
    torchrun \
        --nproc_per_node=8 \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr=192.168.1.100 \
        --master_port=29500 \
        train.py --outdir=training-runs --data=datasets/ecg_real --batch=1024
```

---

## ✅ 動作確認チェックリスト

- [ ] NVIDIA Docker Runtimeのインストール完了
- [ ] Dockerイメージのビルド成功
- [ ] テストデータセットの作成成功
- [ ] テスト学習の実行成功
- [ ] 心電図生成の成功
- [ ] 実データの準備完了
- [ ] 実データでの学習開始

---

## 🎯 推奨ワークフロー（Docker版）

### 開発・テスト時

```bash
# インタラクティブモードで作業
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    bash
```

### 本番学習時

```bash
# バックグラウンドで長時間実行
docker run -d --name ecg-training \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=8g \
    --restart unless-stopped \
    edm-ecg:latest \
    torchrun --standalone --nproc_per_node=8 train.py \
        --outdir=training-runs \
        --data=datasets/ecg_real \
        --duration=200 \
        --batch=512
```

---

## 📚 関連ドキュメント

- **詳細ガイド**: [ECG_TRAINING_GUIDE.md](./ECG_TRAINING_GUIDE.md)
- **クイックスタート**: [ECG_QUICKSTART.md](./ECG_QUICKSTART.md)
- **元のREADME**: [README.md](./README.md)
- **Docker公式ドキュメント**: https://docs.docker.com/

---

## 💡 ヒント

1. **データの永続化**: `-v`オプションで必ずホストディレクトリをマウントしてください
2. **GPU選択**: 特定のGPUのみ使用する場合は`--gpus '"device=0,1"'`のように指定
3. **メモリ管理**: 大規模学習時は`--shm-size`を適切に設定
4. **ログ監視**: `docker logs -f`でリアルタイムにログを確認
5. **定期的なスナップショット保存**: `--snap`オプションを適切に設定

---

Good luck with your ECG generation model training! 🚀🏥
