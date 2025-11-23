# 12誘導心電図生成モデル - クイックスタートガイド

このガイドでは、最短手順で心電図生成モデルの学習を始める方法を説明します。

## ⚡ 5分で始める

### ステップ1: 環境セットアップ

```bash
# Conda環境を作成・有効化
conda env create -f environment.yml -n edm
conda activate edm
```

### ステップ2: テストデータセットの作成

```bash
# ダミーデータセットを作成（100サンプル）
python preprocess_ecg_data.py create_dummy \
    --output_dir datasets/ecg_test \
    --num_samples 100

# データセットの検証
python preprocess_ecg_data.py verify \
    --dataset_dir datasets/ecg_test
```

### ステップ3: 学習の開始（テスト実行）

```bash
# 単一GPUで小規模テスト学習
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
python generate_ecg.py \
    --network training-runs/00000-*/network-snapshot-000005.pkl \
    --outdir generated_ecg \
    --num_samples 5 \
    --steps 18
```

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
python preprocess_ecg_data.py preprocess \
    --input_dir raw_data \
    --output_dir datasets/ecg_real \
    --normalize minmax

# 検証
python preprocess_ecg_data.py verify \
    --dataset_dir datasets/ecg_real
```

### 3. 本格的な学習

```bash
# 8GPUでの本格学習
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

## 📁 ファイル構成

```
edm/
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
python preprocess_ecg_data.py verify --dataset_dir datasets/ecg_test
```

### CUDA利用不可

```bash
# CUDAの確認
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 詳細情報

詳しい説明は以下を参照してください：
- **詳細ガイド**: [ECG_TRAINING_GUIDE.md](./ECG_TRAINING_GUIDE.md)
- **元のREADME**: [README.md](./README.md)

## 💡 推奨設定

| データセットサイズ | 推奨GPU数 | batch | duration | 学習時間（目安） |
|------------------|----------|-------|----------|----------------|
| < 1,000 | 1-2 | 64-128 | 50-100 | 数時間 |
| 1,000-10,000 | 4-8 | 256-512 | 100-200 | 1-2日 |
| > 10,000 | 8+ | 512-1024 | 200-500 | 2-4日 |

## ✅ チェックリスト

- [ ] Conda環境のセットアップ完了
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

Good luck! 🚀
