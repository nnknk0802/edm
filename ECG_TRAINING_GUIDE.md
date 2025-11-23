# 12誘導心電図生成モデル 学習ガイド

このガイドでは、NVIDIA EDM（Elucidating the Design Space of Diffusion-Based Generative Models）フレームワークを使用して、12誘導心電図データの生成モデルを学習する手順を説明します。

## 目次
1. [環境セットアップ](#1-環境セットアップ)
2. [データ仕様](#2-データ仕様)
3. [データセット準備](#3-データセット準備)
4. [コード修正](#4-コード修正)
5. [学習実行](#5-学習実行)
6. [モデル評価と生成](#6-モデル評価と生成)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 環境セットアップ

### 1.1 必要な環境
- **OS**: Linux推奨（Windowsも可）
- **GPU**: NVIDIA GPU 1台以上（学習には8台以上推奨）
  - V100、A100などの高性能GPUを推奨
- **Python**: 3.8以上
- **PyTorch**: 1.12.0以上
- **CUDA**: 対応バージョン

### 1.2 Conda環境の構築

```bash
# リポジトリのルートディレクトリで実行
cd /home/user/edm

# Conda環境を作成
conda env create -f environment.yml -n edm

# 環境を有効化
conda activate edm

# 動作確認
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 1.3 Dockerを使用する場合（オプション）

```bash
# Dockerイメージをビルド
docker build --tag edm:latest .

# コンテナを起動
docker run --gpus all -it --rm --user $(id -u):$(id -g) \
    -v `pwd`:/scratch --workdir /scratch -e HOME=/scratch \
    edm:latest bash
```

---

## 2. データ仕様

### 2.1 入力データの形式

12誘導心電図データの仕様：

| パラメータ | 値 | 説明 |
|----------|-----|------|
| **誘導数** | 12 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| **サンプル長** | 1000 | 時系列のサンプル数 |
| **サンプリング周波数** | 100 Hz | 1秒あたり100サンプル |
| **記録時間** | 10秒 | 100 Hz × 10秒 = 1000サンプル |
| **データ型** | float32 | 推奨 |
| **正規化範囲** | [-1, 1] | モデル学習時に使用 |

### 2.2 データ形状

```python
# 1サンプルのデータ形状
shape = (num_leads, seqlen)
      = (12, 1000)

# バッチデータの形状
batch_shape = (batch_size, 12, 1000)
```

### 2.3 EDMフレームワークへの適応方法

EDMは元々2D画像データ（C×H×W）用に設計されているため、1D時系列データである心電図を扱うには以下の選択肢があります：

#### **オプション1: 1D → 2D変換（推奨）**
```python
# 元の形状: (12, 1000)
# 変換後: (12, 1, 1024)  # 高さ1、幅1024にパディング
# または
# 変換後: (1, 12, 1024)  # チャンネル1、12×1024の2D画像として扱う
```

#### **オプション2: ネットワークアーキテクチャの1D化**
Conv2d → Conv1dに変更（より大規模な修正が必要）

このガイドでは**オプション1**を採用し、`(1, 12, 1024)`の形状に変換します。

---

## 3. データセット準備

### 3.1 データセット構造

心電図データを以下のディレクトリ構造で準備します：

```
ecg_dataset/
├── 00000.npy          # サンプル0: shape=(12, 1000)
├── 00001.npy          # サンプル1: shape=(12, 1000)
├── 00002.npy          # サンプル2: shape=(12, 1000)
├── ...
└── dataset_info.json  # メタデータ（オプション）
```

### 3.2 データの前処理スクリプト例

データを準備するPythonスクリプトの例：

```python
# preprocess_ecg_data.py
import numpy as np
import os
from pathlib import Path

def normalize_ecg(ecg_data, target_min=-1, target_max=1):
    """
    心電図データを[-1, 1]の範囲に正規化

    Args:
        ecg_data: shape=(12, 1000) の心電図データ
    Returns:
        正規化されたデータ
    """
    # 各誘導ごとに正規化（オプション: 全誘導で統一も可能）
    min_val = ecg_data.min(axis=1, keepdims=True)
    max_val = ecg_data.max(axis=1, keepdims=True)

    # 正規化
    normalized = (ecg_data - min_val) / (max_val - min_val + 1e-8)
    normalized = normalized * (target_max - target_min) + target_min

    return normalized

def prepare_ecg_dataset(input_dir, output_dir):
    """
    心電図データセットを準備

    Args:
        input_dir: 元データのディレクトリ（例: CSVファイルなど）
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # ここでは例として、input_dirにnumpyファイルがあると仮定
    input_files = sorted(Path(input_dir).glob('*.npy'))

    for idx, input_file in enumerate(input_files):
        # データ読み込み
        ecg_data = np.load(input_file)  # shape: (12, 1000)

        # データ検証
        assert ecg_data.shape == (12, 1000), f"Invalid shape: {ecg_data.shape}"

        # 正規化
        ecg_normalized = normalize_ecg(ecg_data)

        # float32に変換
        ecg_normalized = ecg_normalized.astype(np.float32)

        # 保存
        output_path = os.path.join(output_dir, f'{idx:05d}.npy')
        np.save(output_path, ecg_normalized)

        if (idx + 1) % 100 == 0:
            print(f'処理済み: {idx + 1} / {len(input_files)}')

    print(f'完了: {len(input_files)}サンプルを処理しました')

# 実行例
if __name__ == '__main__':
    prepare_ecg_dataset(
        input_dir='path/to/raw/ecg/data',
        output_dir='datasets/ecg_dataset'
    )
```

### 3.3 データの検証

準備したデータを検証します：

```python
# verify_dataset.py
import numpy as np
from pathlib import Path

def verify_ecg_dataset(dataset_dir):
    """データセットの整合性を確認"""
    files = sorted(Path(dataset_dir).glob('*.npy'))

    print(f'総サンプル数: {len(files)}')

    for idx, file in enumerate(files[:5]):  # 最初の5サンプルを確認
        data = np.load(file)
        print(f'\nサンプル {idx}:')
        print(f'  ファイル: {file.name}')
        print(f'  形状: {data.shape}')
        print(f'  データ型: {data.dtype}')
        print(f'  最小値: {data.min():.4f}')
        print(f'  最大値: {data.max():.4f}')
        print(f'  平均値: {data.mean():.4f}')
        print(f'  標準偏差: {data.std():.4f}')

if __name__ == '__main__':
    verify_ecg_dataset('datasets/ecg_dataset')
```

---

## 4. コード修正

EDMフレームワークを心電図データに対応させるため、以下のファイルを作成・修正します。

### 4.1 カスタムデータセットクラスの作成

`training/ecg_dataset.py`を新規作成します：

```python
# training/ecg_dataset.py
"""12誘導心電図データセットローダー"""

import os
import numpy as np
import torch
from pathlib import Path
from training.dataset import Dataset

class ECGDataset(Dataset):
    """
    12誘導心電図データセット

    データ形式:
        - 入力: (12, 1000) の numpy配列（12誘導、1000サンプル）
        - 出力: (1, 12, 1024) のテンソル（EDMの2D画像形式に変換）
    """

    def __init__(self,
        path,                   # データセットディレクトリのパス
        resolution=None,        # 使用しない（互換性のため残す）
        use_labels=False,       # ラベル使用フラグ
        max_size=None,          # データセットサイズ制限
        xflip=False,            # 使用しない（心電図には不適切）
        random_seed=0,          # ランダムシード
        cache=True,             # キャッシュ使用フラグ
    ):
        self._path = path
        self._use_labels = use_labels
        self._cache = cache
        self._cached_data = {}

        # データファイルのリストを取得
        if not os.path.isdir(self._path):
            raise IOError(f'Path must point to a directory: {self._path}')

        self._all_files = sorted(Path(self._path).glob('*.npy'))

        if len(self._all_files) == 0:
            raise IOError(f'No .npy files found in {self._path}')

        # 最初のサンプルをロードして形状を確認
        first_sample = np.load(self._all_files[0])
        if first_sample.shape != (12, 1000):
            raise ValueError(
                f'Expected ECG shape (12, 1000), got {first_sample.shape}. '
                f'File: {self._all_files[0]}'
            )

        # データセット名と形状を設定
        name = os.path.basename(self._path)

        # EDM形式の画像形状: (チャンネル, 高さ, 幅)
        # ECGを (1, 12, 1024) の2D画像として扱う
        raw_shape = [len(self._all_files), 1, 12, 1024]

        super().__init__(
            name=name,
            raw_shape=raw_shape,
            max_size=max_size,
            use_labels=use_labels,
            xflip=False,  # 心電図にxflipは不適切なので常にFalse
            random_seed=random_seed,
            cache=cache
        )

        print(f'ECGDataset initialized:')
        print(f'  Path: {self._path}')
        print(f'  Samples: {len(self._all_files)}')
        print(f'  Shape: {raw_shape}')

    def _load_raw_image(self, raw_idx):
        """
        心電図データをロードしてEDM形式に変換

        Args:
            raw_idx: データインデックス

        Returns:
            shape=(1, 12, 1024), dtype=uint8 のnumpy配列
        """
        # ファイルから読み込み
        ecg_data = np.load(self._all_files[raw_idx])  # shape: (12, 1000)

        # データ範囲の確認（[-1, 1]を想定）
        assert ecg_data.shape == (12, 1000), f'Invalid shape: {ecg_data.shape}'

        # (12, 1000) → (12, 1024) にゼロパディング
        padded = np.zeros((12, 1024), dtype=np.float32)
        padded[:, :1000] = ecg_data

        # (12, 1024) → (1, 12, 1024) にリシェイプ
        reshaped = padded[np.newaxis, :, :]  # shape: (1, 12, 1024)

        # [-1, 1] → [0, 255] に変換（EDMは内部でuint8を期待）
        # 学習ループで再度[-1, 1]に正規化されます
        normalized = ((reshaped + 1) * 127.5).clip(0, 255)

        # uint8に変換
        image = normalized.astype(np.uint8)

        return image

    def _load_raw_labels(self):
        """
        ラベルをロード（オプション）

        クラス条件付き生成を行う場合、ここでラベルを実装します。
        例: 不整脈の種類、患者の年齢グループなど
        """
        if not self._use_labels:
            return None

        # ラベルファイルが存在する場合の処理例
        label_file = os.path.join(self._path, 'labels.npy')
        if os.path.exists(label_file):
            labels = np.load(label_file)
            return labels.astype(np.int64)

        return None

    @property
    def resolution(self):
        """画像の解像度（幅）を返す"""
        # EDMは正方形画像を期待しますが、心電図は非正方形です
        # ここでは幅（1024）を返します
        return 1024
```

### 4.2 学習スクリプトの修正

`train.py`を修正して、ECGデータセットを使用できるようにします。

**重要な変更点:**
- `c.dataset_kwargs`でECGDatasetを使用
- 解像度チェックを無効化または調整

```python
# train.pyの97行目付近を以下のように修正:

# 元のコード:
# c.dataset_kwargs = dnnlib.EasyDict(
#     class_name='training.dataset.ImageFolderDataset',
#     path=opts.data,
#     use_labels=opts.cond,
#     xflip=opts.xflip,
#     cache=opts.cache
# )

# 修正後: ECGデータセット用
c.dataset_kwargs = dnnlib.EasyDict(
    class_name='training.ecg_dataset.ECGDataset',
    path=opts.data,
    use_labels=opts.cond,
    cache=opts.cache
)
```

### 4.3 ネットワーク設定の調整

心電図データは`(1, 12, 1024)`という非正方形の画像として扱われます。EDMのネットワークは正方形画像を前提としているため、以下の調整が必要です。

`train.py`のネットワーク設定部分（116-124行目付近）を確認し、必要に応じてチャンネル設定を調整します：

```python
# ネットワークアーキテクチャの例（ddpmpp）
if opts.arch == 'ddpmpp':
    c.network_kwargs.update(
        model_type='SongUNet',
        embedding_type='positional',
        encoder_type='standard',
        decoder_type='standard',
        channel_mult_noise=1,
        resample_filter=[1,1],
        model_channels=128,      # ベースチャンネル数
        channel_mult=[2,2,2]     # 解像度ごとのチャンネル倍率
    )
```

**心電図用の推奨設定:**
```python
# より小さいモデルから始める（12×1024は画像より情報量が少ない可能性）
--arch=ddpmpp
--cbase=64              # model_channelsを64に設定
--cres=1,2,2            # channel_multを調整
--dropout=0.1           # ドロップアウトを調整
--augment=0.0           # データ拡張は心電図には不適切
```

---

## 5. 学習実行

### 5.1 基本的な学習コマンド

```bash
# 単一GPU での学習（テスト用）
python train.py \
    --outdir=training-runs \
    --data=datasets/ecg_dataset \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=200 \
    --batch=64 \
    --batch-gpu=64 \
    --lr=0.0001 \
    --dropout=0.1 \
    --augment=0.0 \
    --xflip=False \
    --tick=10 \
    --snap=50 \
    --seed=42
```

### 5.2 複数GPU での学習（推奨）

```bash
# 8GPU での学習
torchrun --standalone --nproc_per_node=8 train.py \
    --outdir=training-runs \
    --data=datasets/ecg_dataset \
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
    --fp16=False \
    --tick=50 \
    --snap=50 \
    --seed=42
```

### 5.3 主要なハイパーパラメータの説明

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `--duration` | 200 | 学習期間（メガ画像単位）<br>200 = 200M画像 = データセット全体を何度も学習 |
| `--batch` | 512 | 総バッチサイズ（全GPU合計） |
| `--batch-gpu` | 64 | GPU1台あたりのバッチサイズ |
| `--lr` | 0.0001 | 学習率（Adam） |
| `--ema` | 0.5 | EMA半減期（メガ画像単位） |
| `--dropout` | 0.1 | ドロップアウト確率 |
| `--augment` | 0.0 | データ拡張確率（心電図には不適切なため0） |
| `--arch` | ddpmpp | ネットワークアーキテクチャ<br>ddpmpp / ncsnpp / adm |
| `--precond` | edm | 事前条件付け方式<br>edm / vp / ve |
| `--cond` | 0 | クラス条件付き学習<br>0=無条件、1=条件付き |

### 5.4 学習の監視

学習中、以下のファイルが出力されます：

```
training-runs/00000-ecg_dataset-uncond-ddpmpp-edm-gpus8-batch512-fp32/
├── training_options.json      # 学習設定
├── log.txt                    # ログファイル
├── stats.jsonl                # 統計情報（損失など）
├── network-snapshot-000000.pkl  # ネットワークスナップショット
├── network-snapshot-000050.pkl
├── network-snapshot-000100.pkl
└── training-state-000500.pt    # 学習状態（再開用）
```

**重要な監視項目:**

1. **学習損失（Loss）**: `stats.jsonl`の`"Loss/loss"`を確認
   ```bash
   # 損失をリアルタイムで確認
   tail -f training-runs/00000-*/log.txt
   ```

2. **GPU メモリ使用量**: OOM エラーが発生する場合は`--batch-gpu`を減らす

3. **学習時間**: 進捗状況を確認

### 5.5 学習の再開

学習が中断した場合、以下のコマンドで再開できます：

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --outdir=training-runs \
    --resume=training-runs/00000-*/training-state-XXXXXX.pt \
    # その他のパラメータは自動的に復元されます
```

---

## 6. モデル評価と生成

### 6.1 心電図生成スクリプトの作成

学習したモデルから心電図を生成するスクリプトを作成します：

```python
# generate_ecg.py
"""学習済みモデルから心電図を生成"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def load_model(network_pkl):
    """学習済みモデルをロード"""
    print(f'モデルをロード中: {network_pkl}')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to('cuda')
    return net

def generate_ecg_samples(net, num_samples=10, num_steps=18, seed=42):
    """
    心電図サンプルを生成

    Args:
        net: 学習済みネットワーク
        num_samples: 生成サンプル数
        num_steps: 拡散ステップ数（少ないほど高速、多いほど高品質）
        seed: ランダムシード

    Returns:
        生成された心電図データ: shape=(num_samples, 12, 1000)
    """
    torch.manual_seed(seed)

    # 潜在変数をサンプリング（ノイズから開始）
    latents = torch.randn(num_samples, 1, 12, 1024, device='cuda')

    # クラスラベル（無条件生成の場合はNone）
    class_labels = None

    # EDMサンプラーの設定
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_churn = 0
    S_min = 0
    S_max = float('inf')
    S_noise = 1

    # 時間ステップの設定
    step_indices = torch.arange(num_steps, device='cuda')
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # 拡散プロセスの実行
    x_next = latents * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # ノイズ追加（確率的サンプリング）
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        if gamma > 0:
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        else:
            x_hat = x_cur

        # デノイジング
        denoised = net(x_hat, t_hat, class_labels).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2次補正（Heun法）
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    # 生成結果を取得
    generated = x_next.cpu().numpy()  # shape: (num_samples, 1, 12, 1024)

    # (num_samples, 1, 12, 1024) → (num_samples, 12, 1000) に変換
    generated = generated[:, 0, :, :1000]

    # [0, 255] → [-1, 1] に逆正規化
    generated = (generated / 127.5) - 1

    return generated

def plot_ecg(ecg_data, save_path=None, title='Generated 12-Lead ECG'):
    """
    12誘導心電図をプロット

    Args:
        ecg_data: shape=(12, 1000) の心電図データ
        save_path: 保存先パス
        title: グラフタイトル
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, axes = plt.subplots(12, 1, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    time = np.arange(1000) / 100  # 100Hzのため、時間軸は0-10秒

    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, ecg_data[i], linewidth=0.5, color='black')
        ax.set_ylabel(lead_name, rotation=0, labelpad=20, fontsize=10)
        ax.set_xlim(0, 10)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        if i == len(axes) - 1:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'保存しました: {save_path}')
    else:
        plt.show()

    plt.close()

def main():
    """メイン処理"""
    # 設定
    network_pkl = 'training-runs/00000-ecg_dataset-uncond-ddpmpp-edm-gpus8-batch512-fp32/network-snapshot-010000.pkl'
    output_dir = 'generated_ecg'
    num_samples = 10
    num_steps = 18  # ステップ数（18: 高速、40: 中速高品質、256: 低速最高品質）

    os.makedirs(output_dir, exist_ok=True)

    # モデルをロード
    net = load_model(network_pkl)

    # 心電図を生成
    print(f'{num_samples}個の心電図を生成中...')
    ecg_samples = generate_ecg_samples(net, num_samples=num_samples, num_steps=num_steps)

    # 各サンプルを保存・可視化
    for i, ecg in enumerate(ecg_samples):
        # numpy配列として保存
        np.save(os.path.join(output_dir, f'ecg_{i:04d}.npy'), ecg)

        # プロットを保存
        plot_ecg(
            ecg,
            save_path=os.path.join(output_dir, f'ecg_{i:04d}.png'),
            title=f'Generated 12-Lead ECG #{i}'
        )

    print(f'完了: {num_samples}個の心電図を生成しました')
    print(f'出力ディレクトリ: {output_dir}')

if __name__ == '__main__':
    main()
```

### 6.2 生成の実行

```bash
# 心電図を生成
python generate_ecg.py
```

### 6.3 評価指標

心電図生成モデルの評価には、以下の指標を使用できます：

#### **定量的評価**

1. **Fréchet Distance (FD)**
   - 実データと生成データの分布間距離
   - 画像のFID（Fréchet Inception Distance）の心電図版

2. **統計的特徴の比較**
   ```python
   # 実データと生成データの統計量を比較
   def compute_statistics(ecg_data):
       """心電図の統計量を計算"""
       stats = {
           'mean': np.mean(ecg_data),
           'std': np.std(ecg_data),
           'min': np.min(ecg_data),
           'max': np.max(ecg_data),
           'rms': np.sqrt(np.mean(ecg_data ** 2))
           # QRS幅、PR間隔、QT間隔なども計算可能
       }
       return stats
   ```

3. **自己相関関数**
   - 心電図の周期性を評価

#### **定性的評価**

1. **視覚的検査**
   - 医師や専門家による評価
   - P波、QRS波、T波の形状確認

2. **臨床的妥当性**
   - 生理学的に正常な波形か
   - 病的パターンの再現性（条件付き生成の場合）

---

## 7. トラブルシューティング

### 7.1 よくある問題と解決策

#### **問題1: GPU メモリ不足（OOM）**

```
RuntimeError: CUDA out of memory
```

**解決策:**
```bash
# batch-gpuを減らす（勾配累積を使用）
--batch-gpu=32  # または16、8など

# またはfp16混合精度を使用
--fp16=True
```

#### **問題2: データセットの形状エラー**

```
AssertionError: Invalid shape: (12, 999)
```

**解決策:**
- データが正確に`(12, 1000)`の形状であることを確認
- `verify_dataset.py`でデータを検証

#### **問題3: 学習が進まない（損失が下がらない）**

**考えられる原因と対策:**

1. **学習率が不適切**
   ```bash
   # 学習率を調整
   --lr=0.0002  # 大きくする
   --lr=0.00005 # 小さくする
   ```

2. **正規化の問題**
   - データが適切に[-1, 1]に正規化されているか確認

3. **アーキテクチャの問題**
   - より小さいモデルから始める
   ```bash
   --cbase=64
   --cres=1,2
   ```

#### **問題4: 生成される心電図の品質が低い**

**対策:**

1. **サンプリングステップ数を増やす**
   ```python
   num_steps = 40  # 18 → 40に増やす
   ```

2. **学習期間を延長**
   ```bash
   --duration=500  # 200 → 500に増やす
   ```

3. **EMA半減期を調整**
   ```bash
   --ema=1.0  # 0.5 → 1.0に増やす
   ```

### 7.2 デバッグモード

学習前に設定を確認するドライラン：

```bash
python train.py \
    --outdir=training-runs \
    --data=datasets/ecg_dataset \
    --cond=0 \
    --arch=ddpmpp \
    --dry-run  # 実際には学習せず、設定のみ表示
```

### 7.3 ログの確認

```bash
# 学習ログをリアルタイムで確認
tail -f training-runs/00000-*/log.txt

# 統計情報を解析
python -c "
import json
with open('training-runs/00000-*/stats.jsonl') as f:
    for line in f:
        stats = json.loads(line)
        print(f\"Kimg: {stats['Progress/kimg']}, Loss: {stats['Loss/loss']}\")
"
```

---

## 付録

### A. ハイパーパラメータチューニングガイド

| データセットサイズ | 推奨batch | 推奨duration | 推奨lr |
|------------------|-----------|--------------|--------|
| < 1,000 | 64-128 | 50-100 | 0.0002 |
| 1,000 - 10,000 | 256-512 | 100-200 | 0.0001 |
| > 10,000 | 512-1024 | 200-500 | 0.0001 |

### B. クラス条件付き生成（オプション）

心電図の種類（正常、不整脈など）でラベル付けされたデータがある場合：

1. **ラベルファイルの準備**
   ```python
   # datasets/ecg_dataset/labels.npy
   labels = np.array([0, 1, 0, 2, ...])  # クラスID
   np.save('datasets/ecg_dataset/labels.npy', labels)
   ```

2. **条件付き学習**
   ```bash
   --cond=1  # クラス条件付きを有効化
   ```

3. **条件付き生成**
   ```python
   # generate_ecg.py内で
   class_labels = torch.tensor([0, 1, 2, ...], device='cuda')  # クラス指定
   ```

### C. 参考文献

- **EDM論文**: [Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022](https://arxiv.org/abs/2206.00364)
- **NVIDIA EDM GitHub**: https://github.com/NVlabs/edm

---

## まとめ

このガイドでは、NVIDIA EDMフレームワークを使用して12誘導心電図の生成モデルを学習する手順を説明しました。

**主要ステップ:**
1. ✅ Conda環境のセットアップ
2. ✅ ECGデータの準備と前処理（12×1000）
3. ✅ カスタムデータセットクラス（`ECGDataset`）の作成
4. ✅ 学習の実行と監視
5. ✅ 心電図の生成と評価

**次のステップ:**
- 小規模データセットでテスト学習を実行
- ハイパーパラメータをチューニング
- 生成品質を評価・改善
- 必要に応じてアーキテクチャを調整

質問や問題が発生した場合は、トラブルシューティングセクションを参照してください。
