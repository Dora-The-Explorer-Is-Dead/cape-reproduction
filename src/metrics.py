"""
Implements the paper's Section 4.3 evaluation metrics (Chowdhury et al., CVPR 2024).

Fixes applied:
1. Binarized thresholding: Converts the heatmap into a binary mask (threshold_frac * max)
   before masking, matching standard CAM literature (Chattopadhay et al.).
2. Head routing: Routes CAPE to the trained 'cape' head and other methods to the 'orig' head.
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
from explain import build_model, GradCAMHooks, generate_all_maps, normalize_map

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
    p.add_argument('--mask_threshold_frac', type=float, default=0.2,
                   help="Binary cutoff as a fraction of max activation (standard is 0.2)")
    p.add_argument('--iou_threshold_frac', type=float, default=0.2,
                   help="Threshold fraction for mIoU computation")
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--checkpoint_every', type=int, default=50)
    return p.parse_args()


def get_confidence(net, image_tensor, target_class, head='orig'):
    """Queries the specific classifier head corresponding to the explanation method."""
    with torch.no_grad():
        out = net(image_tensor.unsqueeze(0))
        probs = out[head]['outcome'].softmax(dim=1)
        return probs[0, target_class].item()


def apply_mask(image_tensor, heatmap_2d, keep=True, threshold_frac=0.2):
    """
    Binarizes the explanation map at (threshold_frac * max_val) and applies it to the image.
    keep=True  -> Keeps only highlighted regions (for AD / IC)
    keep=False -> Deletes highlighted regions (for ADD)
    """
    h, w = image_tensor.shape[-2:]
    mask = torch.from_numpy(heatmap_2d).float().to(image_tensor.device)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()

    cutoff = threshold_frac * mask.max()
    binary_mask = (mask >= cutoff).float()

    if not keep:
        binary_mask = 1.0 - binary_mask

    return image_tensor * binary_mask.unsqueeze(0)


def average_drop(y_c, o_c):
    return max(y_c - o_c, 0.0) / max(y_c, 1e-8)


def average_increase(y_c, o_c):
    return 1.0 if o_c > y_c else 0.0


def ad_in_deletion(y_c, d_c):
    return max(y_c - d_c, 0.0) / max(y_c, 1e-8)


def coherency(e_c, e_c_prime):
    a = e_c.flatten()
    b = e_c_prime.flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        corr = 0.0
    else:
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    return (corr + 1) / 2


def complexity(e_c):
    return float(np.abs(e_c).sum() / e_c.size)


def adcc_score(coh, com, ad):
    eps = 1e-8
    terms = [1.0 / max(coh, eps), 1.0 / max(1 - com, eps), 1.0 / max(1 - ad, eps)]
    return 3.0 / sum(terms)


def compute_iou(map1, map2, threshold_frac):
    t1 = threshold_frac * map1.max()
    t2 = threshold_frac * map2.max()
    mask1 = map1 > t1
    mask2 = map2 > t2
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def borda_count(results):
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


def evaluate_one_image(net, hooks, image_tensor, heatmaps, target_class, threshold_frac=0.2):
    per_method = {}

    for method, e_c in heatmaps.items():
        head = 'cape' if method == 'CAPE' else 'orig'
        y_c = get_confidence(net, image_tensor, target_class, head=head)

        keep_masked = apply_mask(image_tensor, e_c, keep=True, threshold_frac=threshold_frac)
        o_c = get_confidence(net, keep_masked, target_class, head=head)

        del_masked = apply_mask(image_tensor, e_c, keep=False, threshold_frac=threshold_frac)
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
    print(f'Evaluating on {n} test images (full test set is {len(test_data)})')

    methods = ['CAM', 'Grad-CAM', 'Grad-CAM++', 'CAPE', 'mu-CAPE']
    ckpt_path = os.path.join(args.output_dir, 'metrics_checkpoint.pkl')
    start_count = 0

    if os.path.isfile(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        running = ckpt['running']
        all_maps_by_method = ckpt['all_maps_by_method']
        start_count = ckpt['count']
        print(f'Resuming from checkpoint: {start_count}/{n} images already done.')
    else:
        running = {m: {'AD': [], 'IC': [], 'ADD': [], 'ADCC': []} for m in methods}
        all_maps_by_method = {m: [] for m in methods}

    try:
        for count, idx in enumerate(indices):
            if count < start_count:
                continue
            raw_image = test_data[idx]
            image_tensor = test_transform(Image.fromarray(raw_image)).to(device)

            with torch.no_grad():
                out = net(image_tensor.unsqueeze(0))
                probs = out['orig']['outcome'].softmax(dim=1)
                top2_conf, top2_idx = probs.topk(2, dim=1)
                pred_class = top2_idx[0, 0].item()
                second_class = top2_idx[0, 1].item()

            heatmaps_top1 = generate_all_maps(net, image_tensor, pred_class, hooks)
            per_method = evaluate_one_image(net, hooks, image_tensor, heatmaps_top1, pred_class,
                                            threshold_frac=args.mask_threshold_frac)

            heatmaps_top2 = generate_all_maps(net, image_tensor, second_class, hooks)

            for m in methods:
                for k in ['AD', 'IC', 'ADD', 'ADCC']:
                    running[m][k].append(per_method[m][k])
                iou = compute_iou(heatmaps_top1[m], heatmaps_top2[m], args.iou_threshold_frac)
                all_maps_by_method[m].append(iou)

            if (count + 1) % 20 == 0:
                print(f'  {count + 1}/{n} images evaluated')

            if (count + 1) % args.checkpoint_every == 0:
                with open(ckpt_path, 'wb') as f:
                    pickle.dump({'running': running, 'all_maps_by_method': all_maps_by_method, 'count': count + 1}, f)
                print(f'  -> Checkpoint saved at {count + 1} images')

    except Exception as e:
        with open(ckpt_path, 'wb') as f:
            pickle.dump({'running': running, 'all_maps_by_method': all_maps_by_method, 'count': count}, f)
        print(f'\nInterrupted at image {count}: {e}')
        raise

    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)

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

    print(f'\n--- Results (n={n} images) ---')
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