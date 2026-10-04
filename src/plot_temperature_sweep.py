"""
Plots the temperature sweep results produced by temperature_sweep.py.
Reads temperature_sweep_summary.csv and produces one figure with all five
metrics plotted against T_kld, for inclusion in the final report.

Usage:
    python src/plot_temperature_sweep.py \
        --summary_csv /kaggle/working/results/temp_sweep/temperature_sweep_summary.csv \
        --output_path /kaggle/working/results/figures/temperature_sweep.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--summary_csv', type=str, required=True)
    p.add_argument('--output_path', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.summary_csv).sort_values('T_kld')

    metrics = [
        ('CAPE_AD', 'AD (lower is better)', 'tab:red'),
        ('CAPE_IC', 'IC (higher is better)', 'tab:green'),
        ('CAPE_ADD', 'ADD (higher is better)', 'tab:blue'),
        ('CAPE_ADCC', 'ADCC (higher is better)', 'tab:orange'),
        ('CAPE_mIoU', 'mIoU (lower is better)', 'tab:purple'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for ax, (col, title, color) in zip(axes, metrics):
        ax.plot(df['T_kld'], df[col], marker='o', color=color, linewidth=2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(df['T_kld'])
        ax.set_xticklabels([str(t) for t in df['T_kld']])
        ax.set_xlabel('T_kld')
        ax.set_title(title)
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5, label='Paper default (T=2)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle('CAPE (PF) explanation quality vs. distillation temperature (T_kld)', y=1.05)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {args.output_path}')

    print('\n--- Summary table ---')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
