"""
Week 6 Experiment: Threshold Sensitivity Analysis of Faithfulness Metrics.
Sweeps mask threshold fraction theta in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
to evaluate the stability of Average Drop (AD) and Increase in Confidence (IC).
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from explain import build_model, GradCAMHooks, generate_all_maps
from metrics import get_confidence, apply_mask, average_drop, average_increase

CAPE_REPO_PATH = next((p for p in [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
] if p and os.path.isdir(p)), None)

if CAPE_REPO_PATH is not None:
    sys.path.insert(0, CAPE_REPO_PATH)
    from configs import transforms as cape_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/experiment')
    p.add_argument('--num_images', type=int, default=150)
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, _ = pickle.load(f)

    _, test_transform = cape_transforms.cub_transform(args.image_size)

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    methods = ['CAM', 'Grad-CAM', 'Grad-CAM++', 'CAPE', 'mu-CAPE']

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    print(f"Running threshold sweep across {n} images for theta in {thresholds}...")

    records = []

    for count, idx in enumerate(indices):
        raw_image = test_data[idx]
        image_tensor = test_transform(Image.fromarray(raw_image)).to(device)

        with torch.no_grad():
            out = net(image_tensor.unsqueeze(0))
            probs = out['orig']['outcome'].softmax(dim=1)
            pred_class = probs.argmax(dim=1).item()

        heatmaps = generate_all_maps(net, image_tensor, pred_class, hooks)

        for theta in thresholds:
            for m in methods:
                head = 'cape' if m == 'CAPE' else 'orig'
                y_c = get_confidence(net, image_tensor, pred_class, head=head)
                keep_img = apply_mask(image_tensor, heatmaps[m], keep=True, threshold_frac=theta)
                o_c = get_confidence(net, keep_img, pred_class, head=head)

                ad = average_drop(y_c, o_c)
                ic = average_increase(y_c, o_c)

                records.append({
                    'image_idx': idx,
                    'method': m,
                    'theta': theta,
                    'AD': ad * 100.0,
                    'IC': ic * 100.0
                })

        if (count + 1) % 25 == 0:
            print(f"  Processed {count + 1}/{n} images")

    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, 'threshold_sweep_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved raw experiment data to {csv_path}")

    # Generate Report Plot
    summary = df.groupby(['method', 'theta'])[['AD', 'IC']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in methods:
        sub = summary[summary['method'] == m]
        axes[0].plot(sub['theta'], sub['AD'], marker='o', label=m)
        axes[1].plot(sub['theta'], sub['IC'], marker='s', label=m)

    axes[0].set_title('Average Drop (AD ↓) vs Mask Threshold (θ)')
    axes[0].set_xlabel('Threshold Fraction (θ)')
    axes[0].set_ylabel('Average Drop (%)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].set_title('Increase in Confidence (IC ↑) vs Mask Threshold (θ)')
    axes[1].set_xlabel('Threshold Fraction (θ)')
    axes[1].set_ylabel('Increase in Confidence (%)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'experiment_threshold_sensitivity.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == '__main__':
    main()