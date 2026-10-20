import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from explain import build_model, GradCAMHooks, grad_cam
from configs import transforms as cape_transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/experiment_temperature')
    p.add_argument('--num_images', type=int, default=150)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()

def compute_cape_maps_with_tau(net, x, target_class, tau):
    """Computes CAPE and mu-CAPE at an arbitrary inference temperature tau."""
    with torch.no_grad():
        feats = net.net.conv(x) if hasattr(net.net, 'conv') else net.net(x)
        # spatial logit tensor [1, 200, H, W]
        S = net.fc(feats) if hasattr(net, 'fc') else net.net.fc(feats)
        
        # Spatial softmax parameterized by tau
        P_hat = F.softmax(S / tau, dim=1) # [1, K, H, W]
        P_c = P_hat[0, target_class].cpu().numpy()
        
        # mu-CAPE contrastive term
        K = P_hat.shape[1]
        sum_other = (P_hat.sum(dim=1) - P_hat[:, target_class]) / (K - 1)
        contrast = F.relu(P_hat[:, target_class] - sum_other).squeeze().cpu().numpy()
        
        # BaseCAM weighting
        w = net.fc.weight[target_class] if hasattr(net, 'fc') else net.net.fc.weight[target_class]
        base_cam = F.relu((feats[0] * w.view(-1, 1, 1)).sum(dim=0)).cpu().numpy()
        if base_cam.max() > 0:
            base_cam /= base_cam.max()
            
        mu_cape = contrast * base_cam
        if mu_cape.max() > 0:
            mu_cape /= mu_cape.max()
            
    return np.clip(P_c, 0.0, 1.0), np.clip(mu_cape, 0.0, 1.0)

def apply_mask(img_tensor, hmap):
    h, w = img_tensor.shape[-2:]
    mask = torch.from_numpy(hmap).float().to(img_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
    return img_tensor * torch.clamp(mask, 0.0, 1.0).unsqueeze(0)

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
    
    tau_list = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
    results = {'tau': tau_list, 'cape_ad': [], 'mucape_ad': [], 'cape_cov': [], 'mucape_cov': [], 'gcam_ad': []}

    print(f"Running Temperature Sweep on {n} images across tau={tau_list}...")

    # Pre-compute Grad-CAM baseline AD
    gcam_drops = []
    for idx in indices[:n]:
        raw = test_data[idx]
        img = Image.fromarray(raw).convert('RGB')
        x = test_transform(img).to(device)
        
        with torch.no_grad():
            y_c = net(x.unsqueeze(0))['orig']['outcome'].softmax(dim=1)[0].max().item()
            c = net(x.unsqueeze(0))['orig']['outcome'].argmax(dim=1).item()
            
        net.zero_grad()
        x_g = x.clone().requires_grad_(True)
        score = net(x_g.unsqueeze(0))['orig']['outcome'][0, c]
        score.backward()
        gcam_map = grad_cam(hooks.features, hooks.gradients)
        gcam_map = (gcam_map - gcam_map.min()) / max(gcam_map.max() - gcam_map.min(), 1e-8)
        
        masked = apply_mask(x, gcam_map)
        with torch.no_grad():
            o_c = net(masked.unsqueeze(0))['orig']['outcome'].softmax(dim=1)[0, c].item()
        gcam_drops.append(max(y_c - o_c, 0.0) / max(y_c, 1e-6))
        hooks.features, hooks.gradients = None, None
        
    baseline_gcam_ad = float(np.mean(gcam_drops)) * 100.0

    for tau in tau_list:
        c_ad, m_ad, c_cov, m_cov = [], [], [], []
        for idx in indices[:n]:
            raw = test_data[idx]
            img = Image.fromarray(raw).convert('RGB')
            x = test_transform(img).to(device)
            
            with torch.no_grad():
                c = net(x.unsqueeze(0))['orig']['outcome'].argmax(dim=1).item()
                y_c_cape = net(x.unsqueeze(0))['cape']['outcome'][0, c].item()
                
            cape_m, mucape_m = compute_cape_maps_with_tau(net, x.unsqueeze(0), c, tau)
            
            # AD
            o_c_cape = net(apply_mask(x, cape_m).unsqueeze(0))['cape']['outcome'][0, c].item()
            o_c_mu = net(apply_mask(x, mucape_m).unsqueeze(0))['cape']['outcome'][0, c].item()
            c_ad.append(max(y_c_cape - o_c_cape, 0.0) / max(y_c_cape, 1e-6))
            m_ad.append(max(y_c_cape - o_c_mu, 0.0) / max(y_c_cape, 1e-6))
            
            # Coverage (% pixels > 0.2)
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
    print(f"\nSaved analysis plot to {plot_path}")

    csv_path = os.path.join(args.output_dir, 'temperature_sweep_data.csv')
    with open(csv_path, 'w') as f:
        f.write("tau,CAPE_AD,muCAPE_AD,CAPE_Cov,muCAPE_Cov\n")
        for i, t in enumerate(tau_list):
            f.write(f"{t},{results['cape_ad'][i]:.2f},{results['mucape_ad'][i]:.2f},{results['cape_cov'][i]:.2f},{results['mucape_cov'][i]:.2f}\n")
    print(f"Saved numerical data to {csv_path}")

if __name__ == '__main__':
    main()