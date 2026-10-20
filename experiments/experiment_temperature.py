import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Path resolution: add repo root and src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from explain import build_model, GradCAMHooks, grad_cam, normalize_map
from configs import transforms as cape_transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/experiment_temperature')
    p.add_argument('--num_images', type=int, default=150)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--image_size', type=int, default=448)
    return p.parse_args()

def compute_cape_with_tau(net, x, target_class, tau):
    """
    Computes CAPE and mu-CAPE using the network's spatial logits rescaled by tau.
    """
    with torch.no_grad():
        out = net(x)
        # cls_map: spatial logits tensor [B, K, H, W]
        cls_map = out['cape']['cls_map'] if 'cls_map' in out['cape'] else out['cape']['weighted_contribution']
        
        # Spatial softmax parameterized by tau
        P_hat = F.softmax(cls_map / tau, dim=1) # [1, K, H, W]
        P_c = P_hat[:, target_class:target_class + 1]
        
        # Contrastive mutual suppression
        K = P_hat.shape[1]
        sum_other = (P_hat.sum(dim=1, keepdim=True) - P_c) / (K - 1)
        contrast = F.relu(P_c - sum_other)
        
        # Base CAM weighting
        base_cam = F.relu(out['orig']['weighted_contribution'][:, target_class:target_class + 1])
        base_cam_norm = base_cam / max(base_cam.max(), 1e-8)
        
        mu_cape = contrast * base_cam_norm
        
    return normalize_map(P_c), normalize_map(mu_cape)

def apply_mask(img_tensor, hmap):
    h, w = img_tensor.shape[-2:]
    mask = torch.from_numpy(hmap).float().to(img_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
    return img_tensor * torch.clamp(mask, 0.0, 1.0).unsqueeze(0)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)
    _, test_transform = cape_transforms.cub_transform(args.image_size)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, _ = pickle.load(f)

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    
    tau_list = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    results = {'tau': tau_list, 'cape_ad': [], 'mucape_ad': [], 'cape_cov': [], 'mucape_cov': []}

    print(f"Running Temperature Sweep on {n} images across tau={tau_list}...")

    # Pre-compute Grad-CAM baseline AD
    gcam_drops = []
    for idx in indices:
        raw = test_data[idx]
        img = Image.fromarray(raw).convert('RGB')
        x = test_transform(img).to(device)
        
        with torch.no_grad():
            out = net(x.unsqueeze(0))
            c = out['orig']['outcome'].argmax(dim=1).item()
            y_c = out['orig']['outcome'].softmax(dim=1)[0, c].item()
            
        net.zero_grad()
        x_g = x.unsqueeze(0).clone().requires_grad_(True)
        score = net(x_g)['orig']['outcome'][0, c]
        score.backward()
        
        gcam_map = normalize_map(grad_cam(hooks.features, hooks.gradients))
        masked = apply_mask(x, gcam_map)
        with torch.no_grad():
            o_c = net(masked.unsqueeze(0))['orig']['outcome'].softmax(dim=1)[0, c].item()
            
        gcam_drops.append(max(y_c - o_c, 0.0) / max(y_c, 1e-6))
        hooks.features, hooks.gradients = None, None

    baseline_gcam_ad = float(np.mean(gcam_drops)) * 100.0

    for tau in tau_list:
        c_ad, m_ad, c_cov, m_cov = [], [], [], []
        for idx in indices:
            raw = test_data[idx]
            img = Image.fromarray(raw).convert('RGB')
            x = test_transform(img).to(device)
            
            with torch.no_grad():
                out = net(x.unsqueeze(0))
                c = out['orig']['outcome'].argmax(dim=1).item()
                y_c_cape = out['cape']['outcome'][0, c].item()
                
            cape_m, mucape_m = compute_cape_with_tau(net, x.unsqueeze(0), c, tau)
            
            o_c_cape = net(apply_mask(x, cape_m).unsqueeze(0))['cape']['outcome'][0, c].item()
            o_c_mu = net(apply_mask(x, mucape_m).unsqueeze(0))['cape']['outcome'][0, c].item()
            
            c_ad.append(max(y_c_cape - o_c_cape, 0.0) / max(y_c_cape, 1e-6))
            m_ad.append(max(y_c_cape - o_c_mu, 0.0) / max(y_c_cape, 1e-6))
            
            c_cov.append(float((cape_m > 0.2).mean()) * 100.0)
            m_cov.append(float((mucape_m > 0.2).mean()) * 100.0)

        results['cape_ad'].append(float(np.mean(c_ad)) * 100.0)
        results['mucape_ad'].append(float(np.mean(m_ad)) * 100.0)
        results['cape_cov'].append(float(np.mean(c_cov)))
        results['mucape_cov'].append(float(np.mean(m_cov)))
        print(f"  tau={tau:.2f} | CAPE AD: {results['cape_ad'][-1]:.1f}% (Cov: {results['cape_cov'][-1]:.1f}%) | mu-CAPE AD: {results['mucape_ad'][-1]:.1f}% (Cov: {results['mucape_cov'][-1]:.1f}%)")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(tau_list, results['cape_ad'], 'o-', label='CAPE (PF)', color='#e74c3c', lw=2)
    ax1.plot(tau_list, results['mucape_ad'], 's-', label='mu-CAPE (PF)', color='#2ecc71', lw=2)
    ax1.axhline(baseline_gcam_ad, linestyle='--', color='#34495e', label=f'Grad-CAM ({baseline_gcam_ad:.1f}%)')
    ax1.set_xlabel('Spatial Temperature (tau)')
    ax1.set_ylabel('Average Drop (%) [Lower is Better]')
    ax1.set_title('Faithfulness vs. Temperature')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(tau_list, results['cape_cov'], 'o-', label='CAPE Coverage', color='#e74c3c', lw=2)
    ax2.plot(tau_list, results['mucape_cov'], 's-', label='mu-CAPE Coverage', color='#2ecc71', lw=2)
    ax2.set_xlabel('Spatial Temperature (tau)')
    ax2.set_ylabel('Mask Coverage (%)')
    ax2.set_title('Spatial Support vs. Temperature')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'experiment_temperature_analysis.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved plot to {plot_path}")

    csv_path = os.path.join(args.output_dir, 'temperature_sweep_data.csv')
    with open(csv_path, 'w') as f:
        f.write("tau,CAPE_AD,muCAPE_AD,CAPE_Cov,muCAPE_Cov\n")
        for i, t in enumerate(tau_list):
            f.write(f"{t},{results['cape_ad'][i]:.2f},{results['mucape_ad'][i]:.2f},{results['cape_cov'][i]:.2f},{results['mucape_cov'][i]:.2f}\n")
    print(f"Saved numerical data to {csv_path}")

if __name__ == '__main__':
    main()