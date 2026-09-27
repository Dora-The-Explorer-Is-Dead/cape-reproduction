"""
Reads metrics.py's checkpoint file (saved automatically when a run is
interrupted, e.g. via Kaggle's stop button) and computes the final results
table from whatever images were actually evaluated before stopping.

Use this when you deliberately stop metrics.py partway through (time
constraints) rather than letting it run to completion on the full test set.

Usage:
    python src/finalize_partial_metrics.py \
        --checkpoint_dir /kaggle/working/results/tables \
        --output_dir /kaggle/working/results/tables
"""

import os
import pickle
import argparse
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from metrics import borda_count  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Folder containing metrics_checkpoint.pkl (the --output_dir you '
                         'originally passed to metrics.py)')
    p.add_argument('--output_dir', type=str, default=None,
                    help='Where to save the results CSV. Defaults to --checkpoint_dir.')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.checkpoint_dir
    ckpt_path = os.path.join(args.checkpoint_dir, 'metrics_checkpoint.pkl')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'No checkpoint found at {ckpt_path}. This script is for finalizing an '
            f'INTERRUPTED metrics.py run. If your run finished normally, the results '
            f'CSV was already written directly by metrics.py — check '
            f'{os.path.join(args.checkpoint_dir, "metrics_results.csv")}.'
        )

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    running = ckpt['running']
    all_maps_by_method = ckpt['all_maps_by_method']
    n_done = ckpt['count']
    methods = list(running.keys())

    print(f'Checkpoint has {n_done} images evaluated. Computing results from partial data.')

    results = {}
    for m in methods:
        results[m] = {
            'AD': float(np.mean(running[m]['AD'])) * 100,
            'IC': float(np.mean(running[m]['IC'])) * 100,
            'ADD': float(np.mean(running[m]['ADD'])) * 100,
            'ADCC': float(np.mean(running[m]['ADCC'])) * 100,
            'mIoU': float(np.mean(all_maps_by_method[m])) * 100,
        }

    bc = borda_count(results)
    for m in methods:
        results[m]['BC'] = bc[m]

    print(f'\n--- Partial Results (n={n_done} images, run stopped early) ---')
    print(f'{"Method":12s} {"AD":>6s} {"IC":>6s} {"ADD":>6s} {"ADCC":>6s} {"mIoU":>6s} {"BC":>4s}')
    for m in methods:
        r = results[m]
        print(f'{m:12s} {r["AD"]:6.1f} {r["IC"]:6.1f} {r["ADD"]:6.1f} {r["ADCC"]:6.1f} {r["mIoU"]:6.1f} {r["BC"]:4d}')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'metrics_results_partial_n{n_done}.csv')
    with open(out_path, 'w') as f:
        f.write(f'# PARTIAL RESULTS — n={n_done} images (run stopped early, not the full test set)\n')
        f.write('Method,AD,IC,ADD,ADCC,mIoU,BC\n')
        for m in methods:
            r = results[m]
            f.write(f'{m},{r["AD"]:.2f},{r["IC"]:.2f},{r["ADD"]:.2f},{r["ADCC"]:.2f},{r["mIoU"]:.2f},{r["BC"]}\n')

    print(f'\nSaved to {out_path}')
    print(f'IMPORTANT: this is a partial result (n={n_done}, not the full 5,794-image test '
          f'set) — state this sample size explicitly wherever you report these numbers.')


if __name__ == '__main__':
    main()
