#!/usr/bin/env python3
"""
心電図データの前処理スクリプト

このスクリプトは、生の心電図データをEDM学習用の形式に変換します。

使用方法:
    python preprocess_ecg_data.py --input_dir <入力ディレクトリ> --output_dir <出力ディレクトリ>
"""

import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def normalize_ecg(ecg_data, method='minmax', target_min=-1, target_max=1):
    """
    心電図データを正規化

    Args:
        ecg_data: shape=(12, 1000) の心電図データ
        method: 正規化方法 ('minmax', 'zscore', 'global_minmax')
        target_min: 正規化後の最小値（minmaxの場合）
        target_max: 正規化後の最大値（minmaxの場合）

    Returns:
        正規化されたデータ
    """
    if method == 'minmax':
        # 各誘導ごとに正規化
        min_val = ecg_data.min(axis=1, keepdims=True)
        max_val = ecg_data.max(axis=1, keepdims=True)
        normalized = (ecg_data - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized * (target_max - target_min) + target_min

    elif method == 'global_minmax':
        # 全誘導で統一的に正規化
        min_val = ecg_data.min()
        max_val = ecg_data.max()
        normalized = (ecg_data - min_val) / (max_val - min_val + 1e-8)
        normalized = normalized * (target_max - target_min) + target_min

    elif method == 'zscore':
        # Z-score正規化（平均0、標準偏差1）
        mean = ecg_data.mean(axis=1, keepdims=True)
        std = ecg_data.std(axis=1, keepdims=True)
        normalized = (ecg_data - mean) / (std + 1e-8)
        # [-1, 1]の範囲にクリップ
        normalized = np.clip(normalized, -3, 3) / 3

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def validate_ecg_shape(ecg_data, expected_shape=(12, 1000)):
    """
    心電図データの形状を検証

    Args:
        ecg_data: 心電図データ
        expected_shape: 期待される形状

    Returns:
        bool: 形状が正しければTrue

    Raises:
        ValueError: 形状が不正な場合
    """
    if ecg_data.shape != expected_shape:
        raise ValueError(
            f"Invalid ECG shape: expected {expected_shape}, got {ecg_data.shape}"
        )
    return True


def prepare_ecg_dataset(
    input_dir,
    output_dir,
    file_pattern='*.npy',
    normalize_method='minmax',
    verify=True
):
    """
    心電図データセットを準備

    Args:
        input_dir: 元データのディレクトリ
        output_dir: 出力ディレクトリ
        file_pattern: 入力ファイルのパターン
        normalize_method: 正規化方法
        verify: データ検証を行うかどうか
    """
    os.makedirs(output_dir, exist_ok=True)

    # 入力ファイルのリストを取得
    input_path = Path(input_dir)
    input_files = sorted(input_path.glob(file_pattern))

    if len(input_files) == 0:
        raise FileNotFoundError(
            f"No files found matching pattern '{file_pattern}' in {input_dir}"
        )

    print(f"入力ファイル数: {len(input_files)}")
    print(f"正規化方法: {normalize_method}")
    print(f"出力ディレクトリ: {output_dir}")

    # 統計情報の収集用
    all_stats = {
        'min': [],
        'max': [],
        'mean': [],
        'std': []
    }

    # 各ファイルを処理
    for idx, input_file in enumerate(tqdm(input_files, desc="処理中")):
        try:
            # データ読み込み
            ecg_data = np.load(input_file)  # shape: (12, 1000)

            # データ検証
            if verify:
                validate_ecg_shape(ecg_data)

            # 統計情報を収集
            all_stats['min'].append(ecg_data.min())
            all_stats['max'].append(ecg_data.max())
            all_stats['mean'].append(ecg_data.mean())
            all_stats['std'].append(ecg_data.std())

            # 正規化
            ecg_normalized = normalize_ecg(ecg_data, method=normalize_method)

            # float32に変換
            ecg_normalized = ecg_normalized.astype(np.float32)

            # 保存
            output_path = os.path.join(output_dir, f'{idx:05d}.npy')
            np.save(output_path, ecg_normalized)

        except Exception as e:
            print(f"\n警告: {input_file} の処理中にエラーが発生しました: {e}")
            continue

    # 統計情報を表示
    print("\n=== データセット統計情報（正規化前） ===")
    print(f"最小値: {np.min(all_stats['min']):.4f}")
    print(f"最大値: {np.max(all_stats['max']):.4f}")
    print(f"平均値: {np.mean(all_stats['mean']):.4f}")
    print(f"標準偏差: {np.mean(all_stats['std']):.4f}")

    print(f"\n完了: {len(input_files)}サンプルを処理しました")
    print(f"出力: {output_dir}")


def create_dummy_dataset(output_dir, num_samples=100, seed=42):
    """
    テスト用のダミー心電図データセットを作成

    Args:
        output_dir: 出力ディレクトリ
        num_samples: 作成するサンプル数
        seed: ランダムシード
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"ダミーデータセットを作成中: {num_samples}サンプル")

    for i in range(num_samples):
        # ダミーの心電図データを生成（単純な正弦波 + ノイズ）
        t = np.linspace(0, 10, 1000)
        ecg_data = np.zeros((12, 1000), dtype=np.float32)

        for lead in range(12):
            # 各誘導で異なる周波数とノイズレベル
            freq = 1.0 + lead * 0.1  # 心拍数のバリエーション
            amplitude = 0.5 + np.random.rand() * 0.5
            noise_level = 0.05

            # 正弦波 + ノイズ
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            noise = np.random.randn(1000) * noise_level
            ecg_data[lead] = signal + noise

        # [-1, 1]に正規化
        ecg_data = normalize_ecg(ecg_data, method='minmax')

        # 保存
        output_path = os.path.join(output_dir, f'{i:05d}.npy')
        np.save(output_path, ecg_data)

    print(f"完了: {num_samples}サンプルを作成しました")
    print(f"出力: {output_dir}")


def verify_dataset(dataset_dir, num_samples=5):
    """
    データセットの整合性を確認

    Args:
        dataset_dir: データセットディレクトリ
        num_samples: 確認するサンプル数
    """
    files = sorted(Path(dataset_dir).glob('*.npy'))

    if len(files) == 0:
        print(f"エラー: {dataset_dir} にnpyファイルが見つかりません")
        return

    print(f'\n=== データセット検証 ===')
    print(f'総サンプル数: {len(files)}')

    for idx, file in enumerate(files[:num_samples]):
        data = np.load(file)
        print(f'\nサンプル {idx}:')
        print(f'  ファイル: {file.name}')
        print(f'  形状: {data.shape}')
        print(f'  データ型: {data.dtype}')
        print(f'  最小値: {data.min():.4f}')
        print(f'  最大値: {data.max():.4f}')
        print(f'  平均値: {data.mean():.4f}')
        print(f'  標準偏差: {data.std():.4f}')

        # 形状チェック
        if data.shape != (12, 1000):
            print(f'  ⚠️  警告: 形状が正しくありません！')


def main():
    parser = argparse.ArgumentParser(
        description='心電図データの前処理スクリプト'
    )

    subparsers = parser.add_subparsers(dest='command', help='サブコマンド')

    # preprocess コマンド
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='心電図データを前処理'
    )
    preprocess_parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='入力データのディレクトリ'
    )
    preprocess_parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='出力ディレクトリ'
    )
    preprocess_parser.add_argument(
        '--pattern',
        type=str,
        default='*.npy',
        help='入力ファイルのパターン（デフォルト: *.npy）'
    )
    preprocess_parser.add_argument(
        '--normalize',
        type=str,
        default='minmax',
        choices=['minmax', 'global_minmax', 'zscore'],
        help='正規化方法'
    )

    # create_dummy コマンド
    dummy_parser = subparsers.add_parser(
        'create_dummy',
        help='テスト用のダミーデータセットを作成'
    )
    dummy_parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='出力ディレクトリ'
    )
    dummy_parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='作成するサンプル数（デフォルト: 100）'
    )

    # verify コマンド
    verify_parser = subparsers.add_parser(
        'verify',
        help='データセットの整合性を確認'
    )
    verify_parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='データセットディレクトリ'
    )
    verify_parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='確認するサンプル数（デフォルト: 5）'
    )

    args = parser.parse_args()

    if args.command == 'preprocess':
        prepare_ecg_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            normalize_method=args.normalize
        )
    elif args.command == 'create_dummy':
        create_dummy_dataset(
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    elif args.command == 'verify':
        verify_dataset(
            dataset_dir=args.dataset_dir,
            num_samples=args.num_samples
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
