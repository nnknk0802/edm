# Copyright (c) 2024. All rights reserved.
# 12-lead ECG dataset loader for EDM framework

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
