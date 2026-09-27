"""
Implements the paper's Section 4.3 evaluation metrics, from the formulas in
the paper text (Chowdhury et al., CVPR 2024) — the reference repo
(AIML-MED/CAPE) did not include an evaluation/metrics script for us to adapt,
so this is built directly from the published equations, not their code.
Cross-checked against the numbers in a report/PROVENANCE.md note if results
look inconsistent with Table 1.

Metrics:
  AD   (Average Drop in confidence)      — lower is better
  IC   (Average Increase in confidence)  — higher is better
  ADD  (AD in Deletion)                  — higher is better (paper's convention)
  ADCC (harmonic mean of coherency, complexity, AD) — higher is better
  mIoU (overlap between top-2 class explanation maps) — lower is better (per paper)
  BC   (Borda Count ranking across methods, computed last from the other five)

Usage:
    python src/metrics.py --checkpoint <best.pth> --processed_dir <path> --num_images 200
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
from explain import build_model, GradCAMHooks, generate_all_maps, normalize_map, parse_args as explain_parse_args  # noqa: E402

CAPE_REPO_PATH = next((p for p in [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
] if p and os.path.isdir(p)), None)
sys.path.insert(0, CAPE_REPO_PATH)
from configs import transforms as cape_transforms  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='results/tables')
    p.add_argument('--num_images', type=int, default=200,
                    help='Number of test images to evaluate on (paper uses the full test '
                         'set; we default to a subset for tractable runtime — note this '
                         'in the report if the full 5,794 is not used).')
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--iou_threshold_frac', type=float, default=0.2,
                    help="mask threshold as a fraction of each map's max value, per the paper's mIoU definition")
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()


def get_confidence(net, image_tensor, target_class):
    with torch.no_grad():
        out = net(image_tensor.unsqueeze(0))
        probs = out['orig']['outcome'].softmax(dim=1)
        return probs[0, target_class].item()


def apply_mask(image_tensor, heatmap_2d, keep):
    """
    Elementwise-multiplies the image by the (upsampled) explanation map.
    keep=True  -> masked image = image * E   (used for AD / IC — keep only highlighted region)
    keep=False -> masked image = image * (1 - E) (used for ADD — delete the highlighted region)
    """
    h, w = image_tensor.shape[-2:]
    mask = torch.from_numpy(heatmap_2d).float().to(image_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
    if not keep:
        mask = 1 - mask
    return image_tensor * mask.unsqueeze(0)


def average_drop(y_c, o_c):
    """AD(x) = max(y_c - o_c, 0) / y_c"""
    return max(y_c - o_c, 0.0) / max(y_c, 1e-8)


def average_increase(y_c, o_c):
    """IC(x) = 1(y_c < o_c)"""
    return 1.0 if o_c > y_c else 0.0


def ad_in_deletion(y_c, d_c):
    """ADD(x) = max(y_c - d_c, 0) / y_c — same shape as AD, computed on the
    deletion-masked image (region removed) instead of the keep-masked image."""
    return max(y_c - d_c, 0.0) / max(y_c, 1e-8)


def coherency(e_c, e_c_prime):
    """coh(Ec, E'c) = (2*corr(Ec, E'c) + 1) / 2 — min-max-normalized Pearson correlation
    between the original explanation map and the explanation map recomputed from the
    masked (Ec-only) image."""
    a = e_c.flatten()
    b = e_c_prime.flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        corr = 0.0
    else:
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    return (2 * corr + 1) / 2


def complexity(e_c):
    """com(Ec) = |Ec| — L1 norm, normalized by map size so it's comparable across resolutions."""
    return float(np.abs(e_c).sum() / e_c.size)


def adcc_score(coh, com, ad):
    """ADCC(x) = 3 / (1/coh + 1/(1-com) + 1/(1-ad)) — harmonic mean of three terms."""
    eps = 1e-8
    terms = [1.0 / max(coh, eps), 1.0 / max(1 - com, eps), 1.0 / max(1 - ad, eps)]
    return 3.0 / sum(terms)


def compute_iou(map1, map2, threshold_frac):
    """IoU between two binarized maps, each thresholded at threshold_frac * its own max."""
    t1 = threshold_frac * map1.max()
    t2 = threshold_frac * map2.max()
    mask1 = map1 > t1
    mask2 = map2 > t2
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def borda_count(results):
    """
    Ranks methods on each metric (respecting each metric's better-direction),
    scores 1st=3, 2nd=2, 3rd=1, rest=0, and sums across metrics.
    `results`: {method_name: {metric_name: value}}
    """
    directions = {'AD': 'lower', 'IC': 'higher', 'ADD': 'higher', 'ADCC': 'higher', 'mIoU': 'lower'}
    methods = list(results.keys())
    bc = {m: 0 for m in methods}

    for metric, direction in directions.items():
        values = [(m, results[m][metric]) for m in methods]
        values.sort(key=lambda x: x[1], reverse=(direction == 'higher'))
        scores = [3, 2, 1] + [0] * (len(values) - 3)
        for (m, _), s in zip(values, scores):
            bc[m] += s

    return bc


def evaluate_one_image(net, hooks, image_tensor, heatmaps, target_class):
    """Computes AD, IC, ADD, coherency, complexity, ADCC for every method's map on one image."""
    y_c = get_confidence(net, image_tensor, target_class)
    per_method = {}

    for method, e_c in heatmaps.items():
        keep_masked = apply_mask(image_tensor, e_c, keep=True)
        o_c = get_confidence(net, keep_masked, target_class)

        del_masked = apply_mask(image_tensor, e_c, keep=False)
        d_c = get_confidence(net, del_masked, target_class)

        # Recompute the explanation map on the keep-masked image, for coherency.
        # Re-uses the same heatmap-generation path so it's method-consistent.
        keep_masked_heatmaps = generate_all_maps(net, keep_masked, target_class, hooks)
        e_c_prime = keep_masked_heatmaps[method]

        ad = average_drop(y_c, o_c)
        ic = average_increase(y_c, o_c)
        add = ad_in_deletion(y_c, d_c)
        coh = coherency(e_c, e_c_prime)
        com = complexity(e_c)
        adcc = adcc_score(coh, com, ad)

        per_method[method] = {'AD': ad, 'IC': ic, 'ADD': add, 'ADCC': adcc}

    return per_method, y_c


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, test_labels = pickle.load(f)

    _, test_transform = cape_transforms.cub_transform(args.image_size)

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    print(f'Evaluating on {n} test images (full test set is {len(test_data)})')

    methods = ['CAM', 'Grad-CAM', 'Grad-CAM++', 'CAPE']
    running = {m: {'AD': [], 'IC': [], 'ADD': [], 'ADCC': []} for m in methods}
    all_maps_by_method = {m: [] for m in methods}  # for mIoU, need top-2 class maps per image

    for count, idx in enumerate(indices):
        raw_image = test_data[idx]
        image_tensor = test_transform(Image.fromarray(raw_image)).to(device)

        with torch.no_grad():
            out = net(image_tensor.unsqueeze(0))
            probs = out['orig']['outcome'].softmax(dim=1)
            top2_conf, top2_idx = probs.topk(2, dim=1)
            pred_class = top2_idx[0, 0].item()
            second_class = top2_idx[0, 1].item()

        heatmaps_top1 = generate_all_maps(net, image_tensor, pred_class, hooks)
        per_method, y_c = evaluate_one_image(net, hooks, image_tensor, heatmaps_top1, pred_class)

        heatmaps_top2 = generate_all_maps(net, image_tensor, second_class, hooks)

        for m in methods:
            for k in ['AD', 'IC', 'ADD', 'ADCC']:
                running[m][k].append(per_method[m][k])
            iou = compute_iou(heatmaps_top1[m], heatmaps_top2[m], args.iou_threshold_frac)
            all_maps_by_method[m].append(iou)

        if (count + 1) % 20 == 0:
            print(f'  {count + 1}/{n} images evaluated')

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

    print('\n--- Results (n={} images) ---'.format(n))
    print(f'{"Method":12s} {"AD":>6s} {"IC":>6s} {"ADD":>6s} {"ADCC":>6s} {"mIoU":>6s} {"BC":>4s}')
    for m in methods:
        r = results[m]
        print(f'{m:12s} {r["AD"]:6.1f} {r["IC"]:6.1f} {r["ADD"]:6.1f} {r["ADCC"]:6.1f} {r["mIoU"]:6.1f} {r["BC"]:4d}')

    out_path = os.path.join(args.output_dir, 'metrics_results.csv')
    with open(out_path, 'w') as f:
        f.write('Method,AD,IC,ADD,ADCC,mIoU,BC\n')
        for m in methods:
            r = results[m]
            f.write(f'{m},{r["AD"]:.2f},{r["IC"]:.2f},{r["ADD"]:.2f},{r["ADCC"]:.2f},{r["mIoU"]:.2f},{r["BC"]}\n')
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
