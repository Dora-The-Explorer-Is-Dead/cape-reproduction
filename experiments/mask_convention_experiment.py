"""
Mask-convention experiment (Milestone 3/4).

QUESTION
    How much does the ranking of explanation methods under Average Drop (AD) depend on
    how a heatmap is turned into a mask?

WHAT THIS DOES
    For every test image and every method (CAM, Grad-CAM, Grad-CAM++, CAPE, mu-CAPE) it
    builds the method's 14x14 heatmap for the predicted class, then turns the map into a
    mask in several different ways, keeps only the masked part of the image, and measures
    how much the model's confidence for that class drops (AD, lower = more faithful).

    Three families of mask ("conditions"):
      soft      the continuous map is the mask (the paper's E_c * x; what metrics.py uses)
      theta_T   binary mask: keep pixels where map >= T * max(map)   (T = 0.05 ... 0.70)
      topk_K    binary mask: keep exactly the K% highest-valued cells of the 14x14 map
                (K = 5 ... 70). Every method keeps the SAME area, so differences can only
                come from WHICH cells a method ranks first, not from how sparse its values are.

    A 'Random' control (topk only) keeps K% of cells chosen at random, i.e. what an
    uninformative map would score at the same area.

    Every masked image is scored by BOTH classifier heads (one forward pass gives both):
      orig   the vanilla classifier          cape   CAPE's own prediction
    so methods can be compared under the SAME head. The results table in the README pairs
    the CAM family with the vanilla head and CAPE/mu-CAPE with the CAPE head; that pairing
    is the 'soft' condition read off as (CAM family, orig) and (CAPE family, cape).

    Predicted class = argmax of the vanilla head, as in metrics.py.
    Uncertainty: 95% paired-bootstrap confidence intervals over images.

Usage:
    python src/mask_convention_experiment.py --checkpoint <best.pth> \
        --processed_dir <dir with test.pkl> --output_dir <out dir> --num_images 1000
"""

import os
import sys
import json
import time
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explain import build_model, GradCAMHooks, generate_all_maps  # noqa: E402  (also puts the reference repo on sys.path)
from configs import transforms as cape_transforms  # noqa: E402

METHODS = ['CAM', 'Grad-CAM', 'Grad-CAM++', 'CAPE', 'mu-CAPE']
HEADS = ['orig', 'cape']
THETAS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
TOPK_FRACS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
PAIRS = [('CAPE', 'Grad-CAM'), ('mu-CAPE', 'Grad-CAM'), ('mu-CAPE', 'CAPE')]

# (name, kind, parameter)
CONDITIONS = ([('soft', 'soft', None)]
              + [(f'theta_{t:.2f}', 'theta', t) for t in THETAS]
              + [(f'topk_{int(round(f * 100)):02d}', 'topk', f) for f in TOPK_FRACS])


def series_keys():
    keys = [(c, m) for c, _, _ in CONDITIONS for m in METHODS]
    keys += [(c, 'Random') for c, kind, _ in CONDITIONS if kind == 'topk']
    return keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--processed_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--num_images', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16,
                   help='masked images scored per forward pass. Lower it (e.g. 8) on CUDA out-of-memory.')
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--checkpoint_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--image_size', type=int, default=448)
    return p.parse_args()


# ----------------------------------------------------------------------------------------
# Pure numpy helpers (no model involved)
# ----------------------------------------------------------------------------------------

def topk_cell_mask(hmap, frac):
    """(H,W) float32 array of 0/1 keeping the top `frac` of cells by value. Stable ordering,
    so ties (e.g. the exact zeros a ReLU'd CAM has) are broken deterministically."""
    h, w = hmap.shape
    k = max(1, int(round(frac * h * w)))
    order = np.argsort(-hmap.reshape(-1), kind='stable')
    keep = np.zeros(h * w, dtype=np.float32)
    keep[order[:k]] = 1.0
    return keep.reshape(h, w)


def random_cell_mask(h, w, frac, rng):
    k = max(1, int(round(frac * h * w)))
    keep = np.zeros(h * w, dtype=np.float32)
    keep[rng.choice(h * w, size=k, replace=False)] = 1.0
    return keep.reshape(h, w)


def make_boot_idx(n, n_boot, seed):
    """One shared set of resampled image indices, so every comparison is a PAIRED bootstrap."""
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def ci_from_idx(values, boot_idx):
    values = np.asarray(values, dtype=float)
    means = values[boot_idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(ad, area, keys, boot_idx):
    """Mean AD (%) with 95% CI per head/series, and mean kept area (%) per series."""
    summ = {h: {k: ci_from_idx(np.asarray(ad[h][k]) * 100.0, boot_idx) for k in keys} for h in HEADS}
    area_summ = {k: float(np.mean(area[k])) * 100.0 for k in keys}
    return summ, area_summ


def paired_differences(ad, cond_names, boot_idx):
    rows = []
    for h in HEADS:
        for c in cond_names:
            for a, b in PAIRS:
                d = (np.asarray(ad[h][(c, a)]) - np.asarray(ad[h][(c, b)])) * 100.0
                mean, lo, hi = ci_from_idx(d, boot_idx)
                rows.append({'head': h, 'condition': c, 'a': a, 'b': b,
                             'mean_diff': mean, 'ci_low': lo, 'ci_high': hi})
    return rows


def ranking_line(summ, head, cond, methods):
    present = [m for m in methods if (cond, m) in summ[head]]
    order = sorted(present, key=lambda m: summ[head][(cond, m)][0])
    return ' < '.join(order)


# ----------------------------------------------------------------------------------------
# Model-dependent parts
# ----------------------------------------------------------------------------------------

def build_mask(hmap, kind, param, size, device, rand_cells=None):
    """Returns a float (H,W) mask in [0,1] on `device`."""
    if kind in ('soft', 'theta'):
        m = torch.from_numpy(hmap).float().to(device)
        m = F.interpolate(m[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]
        m = torch.clamp(m, 0.0, 1.0)
        if kind == 'theta':
            top = float(m.max())
            if top <= 0.0:
                return m * 0.0  # a map with no positive evidence keeps nothing
            m = (m >= param * top).float()
        return m
    cells = rand_cells if rand_cells is not None else topk_cell_mask(hmap, param)
    m = torch.from_numpy(np.ascontiguousarray(cells)).float().to(device)
    return F.interpolate(m[None, None], size=size, mode='nearest')[0, 0]


def process_image(net, hooks, x, args, device, image_rng):
    """Returns ({head: {(cond, method): AD}}, {(cond, method): kept_area_fraction}) for one image."""
    size = tuple(x.shape[-2:])
    with torch.no_grad():
        out = net(x[None])
        c = int(out['orig']['outcome'].argmax(dim=1).item())
        y = {'orig': float(out['orig']['outcome'].softmax(dim=1)[0, c].item()),
             'cape': float(out['cape']['outcome'][0, c].item())}

    maps = generate_all_maps(net, x, c, hooks)   # {method: 14x14 numpy map in [0,1]}
    h_cells, w_cells = maps['CAM'].shape

    items = []   # (condition, method, mask)
    for cname, kind, param in CONDITIONS:
        for m in METHODS:
            items.append((cname, m, build_mask(maps[m], kind, param, size, device)))
        if kind == 'topk':
            rc = random_cell_mask(h_cells, w_cells, param, image_rng)
            items.append((cname, 'Random', build_mask(None, 'topk', param, size, device, rand_cells=rc)))

    ad = {h: {} for h in HEADS}
    area = {}
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        batch = torch.stack([x * mask[None] for (_, _, mask) in chunk])
        with torch.no_grad():
            o = net(batch)
            p = {'orig': o['orig']['outcome'].softmax(dim=1)[:, c].cpu().numpy(),
                 'cape': o['cape']['outcome'][:, c].cpu().numpy()}
        for j, (cname, m, mask) in enumerate(chunk):
            for h in HEADS:
                ad[h][(cname, m)] = max(y[h] - float(p[h][j]), 0.0) / max(y[h], 1e-6)
            area[(cname, m)] = float(mask.mean().item())
    return ad, area


def save_state(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)   # atomic: an interruption can't leave a half-written checkpoint


# ----------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------

def fmt(t):
    mean, lo, hi = t
    return f'{mean:5.1f}±{(hi - lo) / 2:.1f}'


def print_report(summ, area_summ, diffs, cond_names, n):
    all_methods = METHODS + ['Random']
    names = {'orig': 'VANILLA head', 'cape': 'CAPE head'}
    for h in HEADS:
        print(f'\n=== Average Drop (%), scored by the {names[h]} — mean ± 95% CI half-width (n={n}) ===')
        print(f'{"condition":11s}' + ''.join(f'{m:>13s}' for m in all_methods))
        for c in cond_names:
            row = f'{c:11s}'
            for m in all_methods:
                row += f'{fmt(summ[h][(c, m)]):>13s}' if (c, m) in summ[h] else f'{"-":>13s}'
            print(row)

    print('\n=== Kept area (% of the image the mask keeps) ===')
    print(f'{"condition":11s}' + ''.join(f'{m:>13s}' for m in all_methods))
    for c in cond_names:
        row = f'{c:11s}'
        for m in all_methods:
            row += f'{area_summ[(c, m)]:13.1f}' if (c, m) in area_summ else f'{"-":>13s}'
        print(row)

    for h in HEADS:
        print(f'\n=== Ranking by AD, best (lowest) first — {names[h]} ===')
        for c in cond_names:
            print(f'{c:11s} {ranking_line(summ, h, c, all_methods)}')

    print('\n=== Paired AD differences (a - b, in % points; negative = a is MORE faithful) — mean [95% CI] ===')
    for h in HEADS:
        print(f'-- {names[h]}')
        print(f'{"condition":11s}' + ''.join(f'{a + " - " + b:>30s}' for a, b in PAIRS))
        for c in cond_names:
            row = f'{c:11s}'
            for a, b in PAIRS:
                r = next(r for r in diffs if r['head'] == h and r['condition'] == c and r['a'] == a and r['b'] == b)
                row += f'{r["mean_diff"]:+7.1f} [{r["ci_low"]:+.1f},{r["ci_high"]:+.1f}]'.rjust(30)
            print(row)

    print('\n=== Check against the README table (soft mask; CAM family = vanilla head, CAPE family = CAPE head) ===')
    for m in METHODS:
        h = 'cape' if m in ('CAPE', 'mu-CAPE') else 'orig'
        print(f'  {m:11s} AD = {fmt(summ[h][("soft", m)])}')


def write_csvs(summ, area_summ, diffs, keys, out_dir):
    with open(os.path.join(out_dir, 'ad_summary.csv'), 'w') as f:
        f.write('head,condition,method,mean_AD,ci_low,ci_high,kept_area_pct\n')
        for h in HEADS:
            for (c, m) in keys:
                mean, lo, hi = summ[h][(c, m)]
                f.write(f'{h},{c},{m},{mean:.3f},{lo:.3f},{hi:.3f},{area_summ[(c, m)]:.3f}\n')
    with open(os.path.join(out_dir, 'paired_differences.csv'), 'w') as f:
        f.write('head,condition,a,b,mean_diff,ci_low,ci_high\n')
        for r in diffs:
            f.write(f'{r["head"]},{r["condition"]},{r["a"]},{r["b"]},'
                    f'{r["mean_diff"]:.3f},{r["ci_low"]:.3f},{r["ci_high"]:.3f}\n')


def make_plots(summ, area_summ, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'CAM': '#7f8c8d', 'Grad-CAM': '#34495e', 'Grad-CAM++': '#8e44ad',
              'CAPE': '#e74c3c', 'mu-CAPE': '#2ecc71', 'Random': '#000000'}
    names = {'orig': 'vanilla head', 'cape': 'CAPE head'}
    theta_names = [f'theta_{t:.2f}' for t in THETAS]
    topk_names = [f'topk_{int(round(f * 100)):02d}' for f in TOPK_FRACS]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
    for r, h in enumerate(HEADS):
        for col, (cnames, xs, xlabel, title) in enumerate([
                (theta_names, THETAS, 'theta (keep map >= theta * max)', 'Binary threshold mask'),
                (topk_names, [100 * f for f in TOPK_FRACS], 'area kept (% of cells)', 'Same-area top-k mask')]):
            ax = axes[r][col]
            for m in METHODS + (['Random'] if col == 1 else []):
                means = [summ[h][(c, m)][0] for c in cnames]
                lo = [summ[h][(c, m)][1] for c in cnames]
                hi = [summ[h][(c, m)][2] for c in cnames]
                ax.plot(xs, means, marker='o', label=m, color=colors[m], lw=2,
                        linestyle='--' if m == 'Random' else '-')
                ax.fill_between(xs, lo, hi, color=colors[m], alpha=0.15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Average Drop (%)  [lower = more faithful]')
            ax.set_title(f'{title} — scored by {names[h]}')
            ax.grid(alpha=0.3)
            if r == 0 and col == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ad_vs_mask_convention.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        ax.plot(THETAS, [area_summ[(c, m)] for c in theta_names], marker='o', label=m, color=colors[m], lw=2)
        ax.axhline(area_summ[('soft', m)], color=colors[m], linestyle=':', alpha=0.6)
    ax.set_xlabel('theta')
    ax.set_ylabel('area the mask keeps (% of image)')
    ax.set_title('How much image each method keeps at the same theta\n(dotted = soft mask)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kept_area_vs_theta.png'), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)
    _, test_transform = cape_transforms.cub_transform(args.image_size)

    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, _ = pickle.load(f)

    n = min(args.num_images, len(test_data))
    indices = np.random.RandomState(args.seed).choice(len(test_data), size=n, replace=False)
    keys = series_keys()
    cond_names = [c for c, _, _ in CONDITIONS]
    print(f'Mask-convention experiment on {n} test images, {len(CONDITIONS)} conditions x '
          f'{len(METHODS)} methods x {len(HEADS)} heads.')

    config = {'checkpoint': args.checkpoint, 'num_images': n, 'seed': args.seed,
              'thetas': THETAS, 'topk_fracs': TOPK_FRACS, 'methods': METHODS}
    ckpt_path = os.path.join(args.output_dir, 'experiment_checkpoint.pkl')
    ad = {h: {k: [] for k in keys} for h in HEADS}
    area = {k: [] for k in keys}
    done = 0
    if os.path.isfile(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state = pickle.load(f)
        if state.get('config') != config:
            raise RuntimeError(
                f'A checkpoint from a different run configuration exists at {ckpt_path} '
                f'(saved: {state.get("config")}, current: {config}). Use a different --output_dir '
                f'or delete that file.')
        ad, area, done = state['ad'], state['area'], state['done']
        print(f'Resuming from checkpoint: {done}/{n} images already done.')

    def snapshot():
        return {'ad': ad, 'area': area, 'done': done, 'config': config}

    start = time.time()
    start_done = done
    try:
        for count in range(done, n):
            idx = int(indices[count])
            x = test_transform(Image.fromarray(test_data[idx]).convert('RGB')).to(device)
            image_rng = np.random.RandomState(args.seed * 100003 + idx)   # same random-control cells on any rerun
            ad_i, area_i = process_image(net, hooks, x, args, device, image_rng)
            for h in HEADS:
                for k in keys:
                    ad[h][k].append(ad_i[h][k])
            for k in keys:
                area[k].append(area_i[k])
            done = count + 1

            if done % 10 == 0 or done == n:
                per_image = (time.time() - start) / max(1, done - start_done)
                print(f'  {done}/{n} images | {per_image:.2f}s/image | about {per_image * (n - done) / 60:.0f} min left')
            if done % args.checkpoint_every == 0:
                save_state(ckpt_path, snapshot())
    except BaseException as e:   # includes KeyboardInterrupt (the stop button)
        save_state(ckpt_path, snapshot())
        print(f'\nStopped after {done} images ({type(e).__name__}). Checkpoint saved - '
              f're-run the identical command to resume.')
        raise

    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    with open(os.path.join(args.output_dir, 'per_image.pkl'), 'wb') as f:
        pickle.dump({'ad': ad, 'area': area, 'indices': indices[:n], 'config': config}, f)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    boot_idx = make_boot_idx(n, args.n_boot, args.seed)
    summ, area_summ = summarize(ad, area, keys, boot_idx)
    diffs = paired_differences(ad, cond_names, boot_idx)

    print_report(summ, area_summ, diffs, cond_names, n)
    write_csvs(summ, area_summ, diffs, keys, args.output_dir)
    make_plots(summ, area_summ, args.output_dir)
    print(f'\nSaved CSVs, plots and per-image data to {args.output_dir}')


if __name__ == '__main__':
    main()
