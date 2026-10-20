"""
Implements Section 4.3 evaluation metrics from Chowdhury et al. (CVPR 2024).

Metrics evaluated:
  AD   (Average Drop in confidence)     — lower is better
  IC   (Average Increase in confidence) — higher is better
  ADD  (AD in Deletion)                 — higher is better (paper's convention)
  ADCC (harmonic mean of coherency, complexity, AD) — higher is better
  mIoU (overlap between top-2 class explanation maps) — lower is better
  BC   (Borda Count ranking across methods, computed last from the other five)

Usage:
    python src/metrics.py --checkpoint <best.pth> --processed_dir <path> --num_images 5794
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from explain import build_model, GradCAMHooks, generate_all_maps

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
    p.add_argument('--output_dir', type=str, default='results/tables')
    p.add_argument('--num_images', type=int, default=200)
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--iou_threshold_frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--checkpoint_every', type=int, default=50)
    return p.parse_args()


def get_confidence(net, image_tensor, target_class, head='orig'):
    """
    Extracts class prediction probability from the appropriate model head:
      - 'orig': raw logits -> requires softmax
      - 'cape': already calibrated spatial probability ensemble -> do NOT softmax again
    """
    with torch.no_grad():
        out = net(image_tensor.unsqueeze(0))
        raw_out = out[head]['outcome']
        if head == 'orig':
            probs = raw_out.softmax(dim=1)
        else:
            probs = raw_out
        return probs[0, target_class].item()


def apply_mask(image_tensor, heatmap_2d, keep=True):
    """
    Standard element-wise explanation masking per Section 4.3 (Chattopadhay et al., Poppi et al.).
    keep=True  -> I_keep = I * E       (AD / IC)
    keep=False -> I_del  = I * (1 - E) (ADD)
    """
    h, w = image_tensor.shape[-2:]
    mask = torch.from_numpy(heatmap_2d).float().to(image_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
    mask = torch.clamp(mask, 0.0, 1.0)

    if not keep:
        mask = 1.0 - mask

    return image_tensor * mask.unsqueeze(0)


def average_drop(y_c, o_c):
    return max(y_c - o_c, 0.0) / max(y_c, 1e-6)


def average_increase(y_c, o_c):
    return 1.0 if o_c > y_c else 0.0


def ad_in_deletion(y_c, d_c):
    return max(y_c - d_c, 0.0) / max(y_c, 1e-6)


def coherency(e_c, e_c_prime):
    a = e_c.flatten()
    b = e_c_prime.flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        corr = 0.0
    else:
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    return (corr + 1.0) / 2.0


def complexity(e_c):
    return float(np.abs(e_c).sum() / e_c.size)


def adcc_score(coh, com, ad):
    eps = 1e-8
    terms = [1.0 / max(coh, eps), 1.0 / max(1.0 - com, eps), 1.0 / max(1.0 - ad, eps)]
    return 3.0 / sum(terms)


def compute_iou(map1, map2, threshold_frac=0.2):
    t1 = threshold_frac * map1.max()
    t2 = threshold_frac * map2.max()
    mask1 = map1 > t1
    mask2 = map2 > t2
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def borda_count(results):
    directions = {'AD': 'lower', 'IC': 'higher', 'ADD': 'higher', 'ADCC': 'higher', 'mIoU': 'lower'}
    methods = list(results.keys())
    bc = {m: 0 for m in methods}

    for metric, direction in directions.items():
        values = [(m, results[m][metric]) for m in methods]
        values.sort(key=lambda x: x[1], reverse=(direction == 'higher'))
        scores = [3, 2, 1] + [0] * max(0, len(values) - 3)
        for (m, _), s in zip(values, scores):
            bc[m] += s

    return bc


def evaluate_one_image(net, hooks, image_tensor, heatmaps, target_class):
    per_method = {}

    for method, e_c in heatmaps.items():
        # CAPE explains the fine-tuned 'cape' head; baselines explain 'orig'
        head = 'cape' if method in ['CAPE', 'mu-CAPE'] else 'orig'

        y_c = get_confidence(net, image_tensor, target_class, head=head)

        keep_masked = apply_mask(image_tensor, e_c, keep=True)
        o_c = get_confidence(net, keep_masked, target_class, head=head)

        del_masked = apply_mask(image_tensor, e_c, keep=False)
        d_c = get_confidence(net, del_masked, target_class, head=head)

        keep_masked_heatmaps = generate_all_maps(net, keep_masked, target_class, hooks)
        e_c_prime = keep_masked_heatmaps[method]

        ad = average_drop(y_c, o_c)
        ic = average_increase(y_c, o_c)
        add = ad_in_deletion(y_c, d_c)
        coh = coherency(e_c, e_c_prime)
        com = complexity(e_c)
        adcc = adcc_score(coh, com, ad)

        per_method[method] = {'AD': ad, 'IC': ic, 'ADD': add, 'ADCC': adcc}

    return per_method


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, _ = pickle.load(f)

    _, test_transform = cape_transforms.cub_transform(args.image_size)

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    print(f'Evaluating on {n} test images (CAPE/mu-CAPE evaluated on "cape" head, baselines on "orig")')

    methods = ['CAM', 'Grad-CAM', 'Grad-CAM++', 'CAPE', 'mu-CAPE']
    running = {m: {'AD': [], 'IC': [], 'ADD': [], 'ADCC': []} for m in methods}
    all_maps_by_method = {m: [] for m in methods}

    for count, idx in enumerate(indices):
        raw_image = test_data[idx]
        if isinstance(raw_image, np.ndarray) and raw_image.shape[-1] == 3:
            image_pil = Image.fromarray(raw_image).convert('RGB')
        else:
            image_pil = Image.fromarray(raw_image)

        image_tensor = test_transform(image_pil).to(device)

        with torch.no_grad():
            out = net(image_tensor.unsqueeze(0))
            probs = out['orig']['outcome'].softmax(dim=1)
            top2_conf, top2_idx = probs.topk(2, dim=1)
            pred_class = top2_idx[0, 0].item()
            second_class = top2_idx[0, 1].item()

        heatmaps_top1 = generate_all_maps(net, image_tensor, pred_class, hooks)
        per_method = evaluate_one_image(net, hooks, image_tensor, heatmaps_top1, pred_class)

        heatmaps_top2 = generate_all_maps(net, image_tensor, second_class, hooks)

        for m in methods:
            for k in ['AD', 'IC', 'ADD', 'ADCC']:
                running[m][k].append(per_method[m][k])
            iou = compute_iou(heatmaps_top1[m], heatmaps_top2[m], args.iou_threshold_frac)
            all_maps_by_method[m].append(iou)

        if (count + 1) % 25 == 0:
            print(f'  {count + 1}/{n} images completed')

    results = {}
    for m in methods:
        results[m] = {
            'AD': float(np.mean(running[m]['AD'])) * 100.0,
            'IC': float(np.mean(running[m]['IC'])) * 100.0,
            'ADD': float(np.mean(running[m]['ADD'])) * 100.0,
            'ADCC': float(np.mean(running[m]['ADCC'])) * 100.0,
            'mIoU': float(np.mean(all_maps_by_method[m])) * 100.0,
        }

    bc = borda_count(results)
    for m in methods:
        results[m]['BC'] = bc[m]

    print(f'\n--- Reproduction Results (n={n} images) ---')
    print(f'{"Method":14s} {"AD":>6s} {"IC":>6s} {"ADD":>6s} {"ADCC":>6s} {"mIoU":>6s} {"BC":>4s}')
    for m in methods:
        r = results[m]
        print(f'{m:14s} {r["AD"]:6.1f} {r["IC"]:6.1f} {r["ADD"]:6.1f} {r["ADCC"]:6.1f} {r["mIoU"]:6.1f} {r["BC"]:4d}')

    out_path = os.path.join(args.output_dir, 'metrics_results.csv')
    with open(out_path, 'w') as f:
        f.write('Method,AD,IC,ADD,ADCC,mIoU,BC\n')
        for m in methods:
            r = results[m]
            f.write(f'{m},{r["AD"]:.2f},{r["IC"]:.2f},{r["ADD"]:.2f},{r["ADCC"]:.2f},{r["mIoU"]:.2f},{r["BC"]}\n')
    print(f'\nSaved metrics table to: {out_path}')


if __name__ == '__main__':
    main()