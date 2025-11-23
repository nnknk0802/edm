#!/usr/bin/env python3
"""
学習済みモデルから心電図を生成

使用方法:
    python generate_ecg.py --network <モデルのパス> --outdir <出力ディレクトリ> --num_samples <サンプル数>
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_model(network_pkl):
    """学習済みモデルをロード"""
    print(f'モデルをロード中: {network_pkl}')

    if not os.path.exists(network_pkl):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {network_pkl}")

    with open(network_pkl, 'rb') as f:
        data = pickle.load(f)

    # EMAモデルを取得
    if 'ema' in data:
        net = data['ema'].to('cuda')
    elif 'model' in data:
        net = data['model'].to('cuda')
    else:
        net = data.to('cuda')

    net.eval()
    print('モデルのロードが完了しました')

    return net


def generate_ecg_samples(
    net,
    num_samples=10,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float('inf'),
    S_noise=1,
    class_labels=None,
    seed=42
):
    """
    心電図サンプルを生成（EDMサンプラー）

    Args:
        net: 学習済みネットワーク
        num_samples: 生成サンプル数
        num_steps: 拡散ステップ数（少ないほど高速、多いほど高品質）
            - 18: 高速（NFE=35）
            - 40: 中速高品質（NFE=79）
            - 256: 低速最高品質（NFE=511）
        sigma_min: 最小ノイズレベル
        sigma_max: 最大ノイズレベル
        rho: ノイズスケジュールのパラメータ
        S_churn: 確率的サンプリングの強度
        S_min: 確率的サンプリングの最小シグマ
        S_max: 確率的サンプリングの最大シグマ
        S_noise: ノイズの倍率
        class_labels: クラスラベル（条件付き生成の場合）
        seed: ランダムシード

    Returns:
        生成された心電図データ: shape=(num_samples, 12, 1000)
    """
    torch.manual_seed(seed)
    device = 'cuda'

    print(f'{num_samples}個の心電図を生成中...')
    print(f'  ステップ数: {num_steps}')
    print(f'  シグマ範囲: [{sigma_min}, {sigma_max}]')

    # 潜在変数をサンプリング（ノイズから開始）
    # shape: (num_samples, 1, 12, 1024)
    latents = torch.randn(num_samples, 1, 12, 1024, device=device)

    # 時間ステップの計算
    step_indices = torch.arange(num_steps, device=device)
    t_steps = (
        sigma_max ** (1 / rho) +
        step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # 拡散プロセスの実行（Heunサンプラー）
    x_next = latents * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # ノイズ追加（確率的サンプリング）
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur

        if gamma > 0:
            noise = torch.randn_like(x_cur) * S_noise
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * noise
        else:
            x_hat = x_cur

        # デノイジング
        with torch.no_grad():
            denoised = net(x_hat, t_hat, class_labels).to(torch.float32)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2次補正（Heun法）
        if i < num_steps - 1:
            with torch.no_grad():
                denoised = net(x_next, t_next, class_labels).to(torch.float32)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # 進捗表示
        if (i + 1) % max(1, num_steps // 10) == 0:
            print(f'  ステップ {i+1}/{num_steps}')

    # 生成結果を取得
    generated = x_next.cpu().numpy()  # shape: (num_samples, 1, 12, 1024)

    # (num_samples, 1, 12, 1024) → (num_samples, 12, 1000) に変換
    generated = generated[:, 0, :, :1000]

    # [0, 255] → [-1, 1] に逆正規化
    generated = (generated / 127.5) - 1

    print('生成完了')

    return generated


def plot_ecg(ecg_data, save_path=None, title='Generated 12-Lead ECG', dpi=300):
    """
    12誘導心電図をプロット

    Args:
        ecg_data: shape=(12, 1000) の心電図データ
        save_path: 保存先パス
        title: グラフタイトル
        dpi: 解像度
    """
    lead_names = [
        'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    ]

    fig, axes = plt.subplots(12, 1, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    time = np.arange(1000) / 100  # 100Hzのため、時間軸は0-10秒

    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, ecg_data[i], linewidth=0.5, color='black')
        ax.set_ylabel(lead_name, rotation=0, labelpad=20, fontsize=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        if i == len(axes) - 1:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def save_ecg_data(ecg_samples, output_dir, save_plots=True, save_numpy=True):
    """
    生成された心電図を保存

    Args:
        ecg_samples: 心電図サンプル array (num_samples, 12, 1000)
        output_dir: 出力ディレクトリ
        save_plots: プロットを保存するか
        save_numpy: numpy配列を保存するか
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, ecg in enumerate(ecg_samples):
        # numpy配列として保存
        if save_numpy:
            npy_path = os.path.join(output_dir, f'ecg_{i:04d}.npy')
            np.save(npy_path, ecg)

        # プロットを保存
        if save_plots:
            plot_path = os.path.join(output_dir, f'ecg_{i:04d}.png')
            plot_ecg(
                ecg,
                save_path=plot_path,
                title=f'Generated 12-Lead ECG #{i}'
            )

    print(f'保存完了: {len(ecg_samples)}個のサンプルを {output_dir} に保存しました')


def main():
    parser = argparse.ArgumentParser(
        description='学習済みモデルから心電図を生成'
    )

    parser.add_argument(
        '--network',
        type=str,
        required=True,
        help='学習済みモデルのパス（.pklファイル）'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='generated_ecg',
        help='出力ディレクトリ（デフォルト: generated_ecg）'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='生成するサンプル数（デフォルト: 10）'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=18,
        help='拡散ステップ数（デフォルト: 18）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='ランダムシード（デフォルト: 42）'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        default=True,
        help='プロットを保存'
    )
    parser.add_argument(
        '--save_numpy',
        action='store_true',
        default=True,
        help='numpy配列を保存'
    )

    args = parser.parse_args()

    # デバイスの確認
    if not torch.cuda.is_available():
        print('警告: CUDAが利用できません。CPUで実行します（非常に遅い可能性があります）')

    # モデルをロード
    net = load_model(args.network)

    # 心電図を生成
    ecg_samples = generate_ecg_samples(
        net,
        num_samples=args.num_samples,
        num_steps=args.steps,
        seed=args.seed
    )

    # 結果を保存
    save_ecg_data(
        ecg_samples,
        output_dir=args.outdir,
        save_plots=args.save_plots,
        save_numpy=args.save_numpy
    )

    print('\n完了!')


if __name__ == '__main__':
    main()
