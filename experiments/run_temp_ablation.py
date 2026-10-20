"""
Stage 4 Experiment: Distillation Temperature (T_kld) Ablation Study.
Tests whether softening the distillation target during Post-Fitting (PF)
reduces CAPE's extreme spatial sparsity and improves faithfulness metrics
(AD, IC, ADD, ADCC) without destroying class-discriminativeness (mIoU).
"""

import argparse
import os
import subprocess
import sys


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument(
      '--T_kld',
      type=float,
      required=True,
      help='Distillation temperature (e.g., 5.0 or 10.0)',
  )
  p.add_argument('--processed_dir', type=str, required=True)
  p.add_argument(
      '--output_dir', type=str, default='/kaggle/working/results_exp'
  )
  p.add_argument('--num_epochs', type=int, default=30)
  p.add_argument(
      '--eval_images',
      type=int,
      default=500,
      help='Number of test images for metrics.py',
  )
  return p.parse_args()


def run_cmd(cmd):
  print(f'\n>>> Running: {" ".join(cmd)}')
  subprocess.run(cmd, check=True)


def main():
  args = parse_args()
  run_name = f'pf_Tkld_{args.T_kld:g}'
  ckpt_path = os.path.join(args.output_dir, 'checkpoints', run_name, 'best.pth')
  tables_dir = os.path.join(args.output_dir, 'tables', run_name)
  figures_dir = os.path.join(args.output_dir, 'figures', run_name)

  # 1. Run PF Training with specified T_kld
  run_cmd([
      sys.executable,
      'src/train_pf.py',
      '--processed_dir',
      args.processed_dir,
      '--output_dir',
      args.output_dir,
      '--run_name',
      run_name,
      '--T_kld',
      str(args.T_kld),
      '--num_epochs',
      str(args.num_epochs),
  ])

  # 2. Evaluate Quantitative Metrics on best.pth
  run_cmd([
      sys.executable,
      'src/metrics.py',
      '--checkpoint',
      ckpt_path,
      '--processed_dir',
      args.processed_dir,
      '--output_dir',
      tables_dir,
      '--num_images',
      str(args.eval_images),
  ])

  # 3. Generate Qualitative Comparison Heatmaps
  run_cmd([
      sys.executable,
      'src/explain.py',
      '--checkpoint',
      ckpt_path,
      '--processed_dir',
      args.processed_dir,
      '--output_dir',
      figures_dir,
      '--num_images',
      '5',
  ])


if __name__ == '__main__':
  main()