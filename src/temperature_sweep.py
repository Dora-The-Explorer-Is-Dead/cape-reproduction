"""
Temperature (T_kld) sensitivity sweep — the Milestone 3/4 experiment.

Per the paper: CAPE's weighted_contribution computes a single softmax over
every voxel at once (H*W*|C| = 14*14*200 = 39,200 entries, Eq. 9), which
dilutes probability mass and makes per-voxel confidence inherently low —
an accuracy-vs-explainability tradeoff built into the CAPE formulation
itself. Bootstrap/distillation training (controlled by T_kld) was
introduced specifically to make PF training converge despite this dilution,
not to directly control final prediction softness. This sweep tests whether
the choice of T_kld changes how well training compensates for that
dilution, and whether that shows up in explanation quality (AD/IC/ADD/ADCC)
and classification accuracy — something the paper states as a design
rationale but never quantifies.

Sweeps T in {0.5, 1, 2, 4, 8}, training a fresh PF model at each value
(full 30 epochs, matching the paper's own PF training length) and
evaluating each on the full 5,794-image test set (matching the paper's
evaluation protocol exactly — no subset compromise, since compute time is
not constrained for this run).

This script runs the whole sweep end-to-end and skips any T value whose
results already exist, so it can be safely re-run after an interruption
(e.g. a Kaggle session reset) without redoing finished work.

Usage:
    python src/temperature_sweep.py \
        --processed_dir /kaggle/working/data/processed \
        --output_dir /kaggle/working/results/temp_sweep \
        --temperatures 0.5 1 2 4 8
"""

import os
import sys
import json
import argparse
import subprocess

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--temperatures', type=float, nargs='+', default=[0.5, 1, 2, 4, 8])
    p.add_argument('--num_epochs', type=int, default=30,
                    help='Matches the paper\'s own PF training length — no reason to cut this '
                         'down given no compute/time constraint.')
    p.add_argument('--metrics_num_images', type=int, default=5794,
                    help='Full CUB-200-2011 test set, matching the paper\'s evaluation protocol '
                         'exactly. Selected via a seeded random choice (--seed) without '
                         'replacement in metrics.py — with n=5794 this is simply every test '
                         'image, so the seed only matters if you later reduce this number.')
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--repo_dir', type=str, default='.')
    return p.parse_args()


def run_cmd(cmd):
    print(f'\n>>> {" ".join(cmd)}\n')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {result.returncode}: {" ".join(cmd)}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'temperature_sweep_summary.csv')

    rows = []
    if os.path.isfile(summary_path):
        existing = pd.read_csv(summary_path)
        rows = existing.to_dict('records')
        done_temps = set(existing['T_kld'].tolist())
        print(f'Found existing summary with T values already done: {sorted(done_temps)}')
    else:
        done_temps = set()

    # Status overview before doing any work, so it's always clear what's
    # protected by resume logic and what still needs to run.
    print('\n=== Sweep status ===')
    for T in args.temperatures:
        run_name = f'temp_sweep_T{T}'.replace('.', 'p')
        ckpt_path = os.path.join(args.output_dir, 'checkpoints', run_name, 'best.pth')
        latest_path = os.path.join(args.output_dir, 'checkpoints', run_name, 'latest.pth')
        metrics_csv = os.path.join(args.output_dir, 'metrics', run_name, 'metrics_results.csv')
        metrics_ckpt = os.path.join(args.output_dir, 'metrics', run_name, 'metrics_checkpoint.pkl')

        if T in done_temps and os.path.isfile(metrics_csv):
            status = 'DONE (training + metrics complete)'
        elif os.path.isfile(ckpt_path) and os.path.isfile(metrics_ckpt):
            status = 'training done, metrics in progress (will resume)'
        elif os.path.isfile(ckpt_path):
            status = 'training done, metrics not started'
        elif os.path.isfile(latest_path):
            status = 'training in progress (will resume from last epoch)'
        else:
            status = 'not started'
        print(f'  T={T}: {status}')
    print()

    for T in args.temperatures:
        run_name = f'temp_sweep_T{T}'.replace('.', 'p')
        ckpt_path = os.path.join(args.output_dir, 'checkpoints', run_name, 'best.pth')
        metrics_dir = os.path.join(args.output_dir, 'metrics', run_name)
        metrics_csv = os.path.join(metrics_dir, 'metrics_results.csv')

        if T in done_temps and os.path.isfile(metrics_csv):
            print(f'\n=== T={T}: already done, skipping ===')
            continue

        print(f'\n=== T={T}: training ===')
        if not os.path.isfile(ckpt_path):
            run_cmd([
                sys.executable, os.path.join(args.repo_dir, 'src', 'train_pf.py'),
                '--subset_fraction', '1.0',
                '--num_epochs', str(args.num_epochs),
                '--run_name', run_name,
                '--T_kld', str(T),
                '--learning_rate', str(args.learning_rate),
                '--processed_dir', args.processed_dir,
                '--output_dir', args.output_dir,
                '--seed', str(args.seed),
            ])
        else:
            print(f'Checkpoint already exists at {ckpt_path}, skipping training for T={T}.')

        print(f'\n=== T={T}: evaluating ===')
        run_cmd([
            sys.executable, os.path.join(args.repo_dir, 'src', 'metrics.py'),
            '--checkpoint', ckpt_path,
            '--processed_dir', args.processed_dir,
            '--output_dir', metrics_dir,
            '--num_images', str(args.metrics_num_images),
            '--seed', str(args.seed),
        ])

        df = pd.read_csv(metrics_csv)
        cape_row = df[df['Method'] == 'CAPE'].iloc[0]
        row = {
            'T_kld': T,
            'CAPE_AD': cape_row['AD'],
            'CAPE_IC': cape_row['IC'],
            'CAPE_ADD': cape_row['ADD'],
            'CAPE_ADCC': cape_row['ADCC'],
            'CAPE_mIoU': cape_row['mIoU'],
        }
        rows.append(row)

        pd.DataFrame(rows).sort_values('T_kld').to_csv(summary_path, index=False)
        print(f'Saved running summary to {summary_path}')

    print('\n=== Sweep complete ===')
    final = pd.DataFrame(rows).sort_values('T_kld')
    print(final.to_string(index=False))
    print(f'\nFull summary: {summary_path}')


if __name__ == '__main__':
    main()
