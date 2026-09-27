"""
Generates CAM, Grad-CAM, Grad-CAM++, and CAPE explanation heatmaps from a
trained checkpoint, for a handful of test images, and saves a comparison
figure per image.

CAM and CAPE come directly from the model's own forward pass — model.py's
get_cam_faster() already computes the CAM formula (Zhou et al.: a 1x1 conv
of the classifier weights over backbone features), and CAPE's heatmap is
the 'weighted_contribution' the model produces from contribution_calculator().
No extra code needed for these two beyond running a forward pass.

Grad-CAM and Grad-CAM++ are NOT produced by model.py — they need gradients
of the target class score with respect to the backbone's last conv feature
map, which we compute here via forward/backward hooks, following the
standard formulas from Selvaraju et al. (Grad-CAM) and Chattopadhay et al.
(Grad-CAM++).

Usage:
    python src/explain.py --checkpoint <path to best.pth> --num_images 5
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

CAPE_REPO_PATH = next((p for p in [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
] if p and os.path.isdir(p)), None)
if CAPE_REPO_PATH is None:
    raise FileNotFoundError('Could not find the cloned AIML-MED/CAPE reference repo.')
sys.path.insert(0, CAPE_REPO_PATH)

from models.model import Net
from configs import transforms as cape_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                    help='Path to best.pth (from train_pf.py, or downloaded from Hugging Face)')
    p.add_argument('--processed_dir', type=str, required=True,
                    help='Folder containing processed/train.pkl and test.pkl')
    p.add_argument('--output_dir', type=str, default='results/figures')
    p.add_argument('--num_images', type=int, default=5)
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--network', type=str, default='resnet50')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()


def build_model(args, device):
    config = {
        'network': args.network,
        'pretrained': True,
        'cls_cols_dict': {'orig': args.num_class, 'cape': args.num_class},
    }
    net = Net(config).to(device)
    # NOTE: train_pf.py's checkpoint dict wraps the model under 'model_state_dict'
    # for the resumable 'latest.pth' format, but 'best.pth' is saved as a bare
    # state_dict (see train_pf.py: torch.save(net.state_dict(), ...best.pth)).
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    net.load_state_dict(state)
    net.eval()
    return net


class GradCAMHooks:
    """Captures the backbone's last feature map and its gradient for one forward/backward pass."""
    def __init__(self, backbone):
        self.features = None
        self.gradients = None
        backbone.register_forward_hook(self._save_features)
        backbone.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]


def grad_cam(features, gradients):
    """Standard Grad-CAM: global-average-pool the gradients into per-channel
    weights, weight the feature map channels, sum, ReLU."""
    weights = gradients.mean(dim=(2, 3), keepdim=True)          # (B, C, 1, 1)
    cam = (weights * features).sum(dim=1, keepdim=True)          # (B, 1, H, W)
    return F.relu(cam)


def grad_cam_pp(features, gradients):
    """Grad-CAM++ (Chattopadhay et al.): weights channels using a second-order
    approximation instead of a plain average, which better handles multiple
    instances of a class in one image."""
    grads_2 = gradients ** 2
    grads_3 = gradients ** 3
    sum_feat_grad3 = (features * grads_3).sum(dim=(2, 3), keepdim=True)
    denom = 2 * grads_2 + sum_feat_grad3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = grads_2 / denom
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    return F.relu(cam)


def normalize_map(m):
    """Min-max normalize a single-channel map to [0, 1] for display/masking."""
    m = m.squeeze().detach().cpu().numpy()
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m


def generate_all_maps(net, image_tensor, target_class, hooks):
    """
    Runs one forward (+ backward, for Grad-CAM/++) pass and returns all four
    heatmaps for the given target class, each normalized to [0, 1].
    """
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor.requires_grad_(False)

    outputs = net(image_tensor)

    # CAM: model.py's own per-pixel activation map for the orig head, at the target class
    cam_map = outputs['orig']['cls_map'][:, target_class:target_class + 1]
    cam_map = normalize_map(cam_map)

    # CAPE: the model's own reformulated contribution map for the target class
    cape_map = outputs['cape']['weighted_contribution'][:, target_class:target_class + 1]
    cape_map = normalize_map(cape_map)

    # Grad-CAM / Grad-CAM++: need a fresh forward pass with grad enabled,
    # since the one above wasn't tracked for backprop into the backbone.
    net.zero_grad()
    image_tensor.requires_grad_(True)
    outputs_grad = net(image_tensor)
    score = outputs_grad['orig']['outcome'][0, target_class]
    score.backward()

    gc_map = normalize_map(grad_cam(hooks.features, hooks.gradients))
    gcpp_map = normalize_map(grad_cam_pp(hooks.features, hooks.gradients))

    return {
        'CAM': cam_map,
        'Grad-CAM': gc_map,
        'Grad-CAM++': gcpp_map,
        'CAPE': cape_map,
    }


def save_comparison_figure(orig_image_np, heatmaps, predicted_class, confidence, out_path):
    fig, axes = plt.subplots(1, len(heatmaps) + 1, figsize=(4 * (len(heatmaps) + 1), 4))

    axes[0].imshow(orig_image_np)
    axes[0].set_title(f'Original\nPred: class {predicted_class} ({confidence:.1%})')
    axes[0].axis('off')

    for ax, (name, hmap) in zip(axes[1:], heatmaps.items()):
        h_up = np.array(Image.fromarray((hmap * 255).astype(np.uint8)).resize(
            (orig_image_np.shape[1], orig_image_np.shape[0]), Image.BILINEAR)) / 255.0
        ax.imshow(orig_image_np)
        ax.imshow(h_up, cmap='jet', alpha=0.5)
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    net = build_model(args, device)
    hooks = GradCAMHooks(net.net)

    import pickle
    with open(os.path.join(args.processed_dir, 'test.pkl'), 'rb') as f:
        test_data, test_labels = pickle.load(f)

    _, test_transform = cape_transforms.cub_transform(args.image_size)

    indices = np.random.RandomState(args.seed).choice(len(test_data), size=args.num_images, replace=False)

    for i, idx in enumerate(indices):
        raw_image = test_data[idx]
        true_label = test_labels[idx]

        image_pil = Image.fromarray(raw_image)
        image_tensor = test_transform(image_pil).to(device)

        # get predicted class first, with no grad, then re-run with grad for Grad-CAM
        with torch.no_grad():
            out = net(image_tensor.unsqueeze(0))
            probs = out['orig']['outcome'].softmax(dim=1)
            confidence, predicted_class = probs.max(dim=1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()

        heatmaps = generate_all_maps(net, image_tensor, predicted_class, hooks)

        out_path = os.path.join(args.output_dir, f'explain_{i}_true{true_label}_pred{predicted_class}.png')
        save_comparison_figure(raw_image, heatmaps, predicted_class, confidence, out_path)
        print(f'[{i+1}/{args.num_images}] true={true_label} pred={predicted_class} '
              f'conf={confidence:.1%} -> saved {out_path}')

    print(f'\nDone. Figures saved to {args.output_dir}')


if __name__ == '__main__':
    main()
