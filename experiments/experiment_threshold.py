import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Resolve repository root and add both root and src/ to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from explain import build_model, GradCAMHooks, generate_all_maps
from configs import transforms as cape_transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/experiment_threshold')
    p.add_argument('--num_images', type=int, default=150)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def apply_thresholded_mask(img_tensor, hmap, theta):
    h, w = img_tensor.shape[-2:]
    mask = torch.from_numpy(hmap).float().to(img_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
    # Binarize at theta fraction of max
    cutoff = theta * mask.max()
    binary_mask = (mask >= cutoff).float()
    return img_tensor * binary_mask.unsqueeze(0)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)
    _, test_transform = cape_transforms.cub_transform(448)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, _ = pickle.load(f)

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    methods = ['Grad-CAM', 'CAPE', 'mu-CAPE']
    results = {m: [] for m in methods}

    print(f"Running Mask Threshold Sensitivity on {n} images across theta={thresholds}...")

    # Pre-extract maps and initial confidences
    samples = []
    for idx in indices[:n]:
        raw = test_data[idx]
        img = Image.fromarray(raw).convert('RGB')
        x = test_transform(img).to(device)
        with torch.no_grad():
            c = net(x.unsqueeze(0))['orig']['outcome'].argmax(dim=1).item()
            y_orig = net(x.unsqueeze(0))['orig']['outcome'].softmax(dim=1)[0, c].item()
            y_cape = net(x.unsqueeze(0))['cape']['outcome'][0, c].item()
        maps = generate_all_maps(net, x, c, hooks)
        samples.append({'x': x, 'c': c, 'y_orig': y_orig, 'y_cape': y_cape, 'maps': maps})

    for th in thresholds:
        ad_epoch = {m: [] for m in methods}
        for s in samples:
            for m in methods:
                head = 'cape' if m in ['CAPE', 'mu-CAPE'] else 'orig'
                y_c = s['y_cape'] if head == 'cape' else s['y_orig']
                masked = apply_thresholded_mask(s['x'], s['maps'][m], th)
                with torch.no_grad():
                    raw_o = net(masked.unsqueeze(0))[head]['outcome']
                    o_c = raw_o[0, s['c']].item() if head == 'cape' else raw_o.softmax(dim=1)[0, s['c']].item()
                ad_epoch[m].append(max(y_c - o_c, 0.0) / max(y_c, 1e-6))
                
        for m in methods:
            results[m].append(float(np.mean(ad_epoch[m])) * 100.0)
        print(f"  theta={th:.2f} | Grad-CAM AD: {results['Grad-CAM'][-1]:.1f}% | CAPE AD: {results['CAPE'][-1]:.1f}% | mu-CAPE AD: {results['mu-CAPE'][-1]:.1f}%")

    # Plot sensitivity curves
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, results['Grad-CAM'], '^--', label='Grad-CAM', color='#34495e', lw=2)
    plt.plot(thresholds, results['CAPE'], 'o-', label='CAPE (PF)', color='#e74c3c', lw=2)
    plt.plot(thresholds, results['mu-CAPE'], 's-', label='mu-CAPE (PF)', color='#2ecc71', lw=2)
    plt.axvline(0.20, color='gray', linestyle=':', label='Default Benchmark (theta=0.20)')
    plt.xlabel('Mask Binarization Threshold (theta)')
    plt.ylabel('Average Drop (%) [Lower is Better]')
    plt.title('Faithfulness Robustness Across Mask Thresholds')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, 'experiment_threshold_sensitivity.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved sensitivity plot to {plot_path}")

    csv_path = os.path.join(args.output_dir, 'threshold_sweep_data.csv')
    with open(csv_path, 'w') as f:
        f.write("theta," + ",".join(methods) + "\n")
        for i, th in enumerate(thresholds):
            f.write(f"{th}," + ",".join(f"{results[m][i]:.2f}" for m in methods) + "\n")
    print(f"Saved numerical data to {csv_path}")

if __name__ == '__main__':
    main()