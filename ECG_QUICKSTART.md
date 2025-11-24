# 12誘導心電図生成モデル - クイックスタートガイド（Docker版）

このガイドでは、Docker環境で最短手順で心電図生成モデルの学習を始める方法を説明します。

## ⚡ 5分で始める

### ステップ1: 環境セットアップ

```bash
# Dockerイメージをビルド
docker build --tag edm-ecg:latest .

# または、ヘルパースクリプトを使用
chmod +x docker-helper.sh
./docker-helper.sh build
```

### ステップ2: テストデータセットの作成

```bash
# ダミーデータセットを作成（100サンプル）
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py create_dummy \
        --output_dir datasets/ecg_test \
        --num_samples 100

# または、ヘルパースクリプトを使用
./docker-helper.sh create-dummy 100

# データセットの検証
./docker-helper.sh verify datasets/ecg_test
```

### ステップ3: 学習の開始（テスト実行）

```bash
# 単一GPUで小規模テスト学習
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

# または、ヘルパースクリプトを使用
./docker-helper.sh train datasets/ecg_test 1 32 1
```

### ステップ4: 心電図の生成

```bash
# 学習したモデルから心電図を生成
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python generate_ecg.py \
        --network training-runs/00000-*/network-snapshot-000005.pkl \
        --outdir generated_ecg \
        --num_samples 5 \
        --steps 18

# または、ヘルパースクリプトを使用
./docker-helper.sh generate training-runs/00000-*/network-snapshot-000005.pkl 5 18
```

生成された心電図は`generated_ecg/`ディレクトリに保存されます。

## 📊 実データでの学習

### 1. データの準備

あなたのデータを`(12, 1000)`のnumpy配列形式で準備します：

```python
import numpy as np

# 例: shape=(12, 1000) の心電図データ
ecg_data = your_ecg_data  # 12誘導、1000サンプル、100Hz、10秒

# 保存
np.save('raw_data/sample_001.npy', ecg_data)
```

### 2. データの前処理

```bash
# 実データを前処理
docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    python preprocess_ecg_data.py preprocess \
        --input_dir raw_data \
        --output_dir datasets/ecg_real \
        --normalize minmax

# 検証
./docker-helper.sh verify datasets/ecg_real
```

### 3. 本格的な学習

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

# または、ヘルパースクリプトを使用
./docker-helper.sh train datasets/ecg_real 200 512 8
```

## 📁 ファイル構成

```
edm/
├── Dockerfile                  # Dockerイメージ定義
├── docker-compose.yml          # Docker Compose設定
├── docker-helper.sh            # Dockerヘルパースクリプト ⭐
├── ECG_DOCKER_GUIDE.md         # Docker詳細ガイド
├── ECG_TRAINING_GUIDE.md       # 詳細ガイド
├── ECG_QUICKSTART.md           # このファイル
├── preprocess_ecg_data.py      # データ前処理スクリプト
├── generate_ecg.py             # 心電図生成スクリプト
├── training/
│   └── ecg_dataset.py          # ECGデータセットクラス
├── datasets/
│   ├── ecg_test/               # テストデータセット
│   └── ecg_real/               # 実データセット
└── training-runs/              # 学習結果
    └── 00000-*/
        ├── network-snapshot-*.pkl    # モデル
        ├── log.txt                   # ログ
        └── stats.jsonl               # 統計
```

## 🔍 トラブルシューティング

### GPU メモリ不足

```bash
# batch-gpuを減らす
--batch-gpu=32  # または16、8
```

### データ形状エラー

```bash
# データを検証
./docker-helper.sh verify datasets/ecg_test
```

### GPU利用不可

```bash
# GPUの確認
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### インタラクティブモードで作業

```bash
# Dockerコンテナ内でシェルを起動
./docker-helper.sh shell

# または
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    edm-ecg:latest \
    bash
```

### バックグラウンドで学習を実行

```bash
# デタッチモードで実行
docker run -d --name ecg-training \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --shm-size=8g \
    edm-ecg:latest \
    python train.py --outdir=training-runs --data=datasets/ecg_real --duration=200 --batch=512

# ログを確認
docker logs -f ecg-training

# または、ホスト側から
tail -f training-runs/00000-*/log.txt
```

## 📚 詳細情報

詳しい説明は以下を参照してください：
- **Docker詳細ガイド**: [ECG_DOCKER_GUIDE.md](./ECG_DOCKER_GUIDE.md) ⭐
- **詳細ガイド**: [ECG_TRAINING_GUIDE.md](./ECG_TRAINING_GUIDE.md)
- **元のREADME**: [README.md](./README.md)

## 💡 推奨設定

| データセットサイズ | 推奨GPU数 | batch | duration | 学習時間（目安） |
|------------------|----------|-------|----------|----------------|
| < 1,000 | 1-2 | 64-128 | 50-100 | 数時間 |
| 1,000-10,000 | 4-8 | 256-512 | 100-200 | 1-2日 |
| > 10,000 | 8+ | 512-1024 | 200-500 | 2-4日 |

## ✅ チェックリスト

- [ ] Dockerイメージのビルド完了
- [ ] GPUの動作確認完了
- [ ] テストデータセットの作成・検証完了
- [ ] テスト学習の実行成功
- [ ] テスト心電図の生成成功
- [ ] 実データの準備完了
- [ ] 実データの前処理完了
- [ ] 本格学習の開始

## 🎯 次のステップ

1. 小規模データでテスト学習を実行
2. 生成された心電図の品質を確認
3. ハイパーパラメータを調整
4. 大規模データセットで本格学習
5. 評価指標を計算して品質を評価

## 🛠️ ヘルパースクリプトのコマンド一覧

`docker-helper.sh`を使うとより簡単に操作できます：

```bash
./docker-helper.sh build            # Dockerイメージをビルド
./docker-helper.sh test             # テスト実行（データ作成→学習）
./docker-helper.sh shell            # インタラクティブシェル
./docker-helper.sh create-dummy 100 # ダミーデータ作成
./docker-helper.sh verify <dir>     # データ検証
./docker-helper.sh train <data> <duration> <batch> <gpus>  # 学習
./docker-helper.sh generate <model> <num> <steps>          # 生成
./docker-helper.sh logs             # ログ表示
./docker-helper.sh clean            # クリーンアップ
```

Good luck! 🚀🏥
