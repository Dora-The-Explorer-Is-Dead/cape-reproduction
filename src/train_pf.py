"""
CAPE Post-Fitting (PF) training script.
Adapted from AIML-MED/CAPE's main.py + trainer.py (Trainer.train / Trainer.test),
trimmed down and reworked to:
  - accept a --subset_fraction argument, so the same script runs both the
    5% quick-check pass and the full 30-epoch run
  - checkpoint every epoch (not just every 5) so an interruption never loses
    more than one epoch of progress
  - automatically resume from the latest checkpoint if one exists
  - log to TensorBoard

Config values (lr, T_kld, epochs, etc.) match configs/cub/resnet50_PF.py in
the reference repo unless overridden via CLI flags.

Usage:
    # Quick subset run to catch bugs fast
    python src/train_pf.py --subset_fraction 0.05 --num_epochs 3 --run_name subset_check

    # Full reproduction run
    python src/train_pf.py --subset_fraction 1.0 --num_epochs 30 --run_name full_pf
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Path to the cloned AIML-MED/CAPE reference repo. Checked in order:
# 1. CAPE_REPO_PATH environment variable, if set
# 2. common local/Kaggle/Colab locations
_CANDIDATES = [
    os.environ.get('CAPE_REPO_PATH'),
    os.path.expanduser('~/cape-reference'),
    '/kaggle/working/cape-reference',
    '/content/cape-reference',
    './cape-reference',
    '../cape-reference',
]
CAPE_REPO_PATH = next((p for p in _CANDIDATES if p and os.path.isdir(p)), None)
if CAPE_REPO_PATH is None:
    raise FileNotFoundError(
        "Could not find the cloned AIML-MED/CAPE reference repo in any of the usual "
        "locations. Either clone it to one of: " + ", ".join(c for c in _CANDIDATES if c) +
        " — or set the CAPE_REPO_PATH environment variable to its location, e.g.:\n"
        "  import os; os.environ['CAPE_REPO_PATH'] = '/kaggle/working/cape-reference'"
    )
print(f'Using CAPE reference repo at: {CAPE_REPO_PATH}')
sys.path.insert(0, CAPE_REPO_PATH)

from models.model import Net
from models.losses import SoftTargetKDLoss
from configs import transforms as cape_transforms

import pickle
from PIL import Image
from torch.utils.data import Dataset


class PickledCUB200(Dataset):
    """
    Loads directly from processed/train.pkl or test.pkl, written by
    resumable_extract.py. Unlike the official CUB200 class, this does NOT
    require a CUB_200_2011.tgz to be present in the same folder — needed
    because on Kaggle the source dataset is read-only and we never build a
    tgz from it, only the pickles (via resumable_extract.py --output_dir
    pointed at a writable location).
    """
    def __init__(self, processed_dir, train=True, transform=None):
        pkl_path = os.path.join(processed_dir, 'train.pkl' if train else 'test.pkl')
        with open(pkl_path, 'rb') as f:
            self.data, self.labels = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', type=str, default=None,
                    help='Folder containing processed/train.pkl and test.pkl, written by '
                         'resumable_extract.py --output_dir. Defaults to '
                         '<output_dir>/processed if not set — i.e. run resumable_extract.py '
                         'with the same --output_dir you pass here as --output_dir first.')
    p.add_argument('--reload_path', type=str,
                    default=os.path.join(CAPE_REPO_PATH, 'saved_models', 'cub_resnet50_TS.pth'),
                    help='TS checkpoint used to initialize PF training, per the authors\' config')
    p.add_argument('--output_dir', type=str, default=os.path.expanduser('~/cape-reproduction/results'))
    p.add_argument('--run_name', type=str, required=True)

    # Values below match configs/cub/resnet50_PF.py — override via CLI if running a sweep (e.g. temperature)
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0,
                    help='CUB200 loads the whole dataset into memory upfront, so >0 workers on '
                         'Windows tries to pickle/copy that entire in-memory dataset per worker '
                         'process and can throw MemoryError. Keep at 0 unless you rework the '
                         'dataset to lazy-load from disk per item.')
    p.add_argument('--num_epochs', type=int, default=30)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--T_kld', type=float, default=2.0)
    p.add_argument('--loss_alpha', type=float, default=0.0)  # PF: classification loss weight
    p.add_argument('--loss_beta', type=float, default=1.0)   # PF: distillation loss weight
    p.add_argument('--num_class', type=int, default=200)
    p.add_argument('--network', type=str, default='resnet50')

    p.add_argument('--subset_fraction', type=float, default=1.0,
                    help='Fraction of the TRAIN set to use (e.g. 0.05 for a 5%% quick check). Test set is always full.')
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_subset(dataset, fraction, seed):
    if fraction >= 1.0:
        return dataset
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    return Subset(dataset, indices)


def build_model(args, device):
    config = {
        'network': args.network,
        'pretrained': True,
        'cls_cols_dict': {'orig': args.num_class, 'cape': args.num_class},
    }
    net = Net(config).to(device)

    state_dict = torch.load(args.reload_path, map_location=device)
    net.load_state_dict(state_dict)

    # PF: freeze backbone + vanilla classifier; only 'cape' head + temperature params train
    for p in net.net.parameters():
        p.requires_grad = False
    for p in net.classifiers['orig'].parameters():
        p.requires_grad = False

    return net


def run_epoch(net, loader, kl_loss_fn, classification_loss_fn, optimizer, opts, device, train):
    net.train() if train else net.eval()
    if train:
        # backbone BN layers stay in eval mode even during PF training, matching trainer.py
        net.net.eval()

    num_total, acc_orig, acc_cape = 0, 0, 0
    running_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = 'Train' if train else 'Test'
    with ctx:
        t = tqdm(loader, desc=desc)
        for images, targets in t:
            images = images.to(device)
            targets = targets.type(torch.LongTensor).to(device)

            if train:
                optimizer.zero_grad()

            outputs = net(images)

            class_loss = opts['loss_alpha'] * classification_loss_fn(outputs['orig']['outcome'], targets)
            kld_loss = opts['loss_beta'] * kl_loss_fn(outputs['cape']['outcome_soft'], outputs['orig']['outcome'].detach())
            loss = class_loss + kld_loss

            if train:
                loss.backward()
                optimizer.step()

            _, pred_orig = torch.max(outputs['orig']['outcome'].softmax(dim=1), 1)
            acc_orig += (pred_orig == targets).sum().item()
            _, pred_cape = torch.max(outputs['cape']['outcome'], 1)
            acc_cape += (pred_cape == targets).sum().item()
            num_total += targets.size(0)
            running_loss += loss.item()

            t.set_postfix_str(f'loss: {loss.item():.4f}')

    return {
        'loss': running_loss / max(1, len(loader)),
        'acc_orig': 100 * acc_orig / max(1, num_total),
        'acc_cape': 100 * acc_cape / max(1, num_total),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    run_dir = os.path.join(args.output_dir, 'logs', args.run_name)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints', args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # Data — loads straight from the pickles resumable_extract.py already produced.
    processed_dir = args.processed_dir or os.path.join(args.output_dir, 'processed')
    if not os.path.isfile(os.path.join(processed_dir, 'train.pkl')):
        raise FileNotFoundError(
            f'No train.pkl found at {processed_dir}. Run resumable_extract.py first, e.g.:\n'
            f'  python resumable_extract.py --data_root <dataset location> --output_dir {args.output_dir}'
        )
    train_transform, test_transform = cape_transforms.cub_transform(args.image_size)
    full_train_dataset = PickledCUB200(processed_dir, train=True, transform=train_transform)
    test_dataset = PickledCUB200(processed_dir, train=False, transform=test_transform)

    train_dataset = make_subset(full_train_dataset, args.subset_fraction, args.seed)
    print(f'Train samples: {len(train_dataset)} / {len(full_train_dataset)} '
          f'({args.subset_fraction*100:.1f}%)')
    print(f'Test samples: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    net = build_model(args, device)
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    print(f'Trainable params: {sum(p.numel() for p in trainable_params):,}')

    optimizer = torch.optim.SGD(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    kl_loss_fn = SoftTargetKDLoss(T=args.T_kld).to(device)
    classification_loss_fn = nn.CrossEntropyLoss().to(device)

    opts = {'loss_alpha': args.loss_alpha, 'loss_beta': args.loss_beta}

    # Resume if a checkpoint from this run already exists
    start_epoch = 0
    latest_ckpt = os.path.join(ckpt_dir, 'latest.pth')
    if os.path.isfile(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resuming from checkpoint: epoch {start_epoch}')

    best_acc_cape = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')
        train_stats = run_epoch(net, train_loader, kl_loss_fn, classification_loss_fn,
                                 optimizer, opts, device, train=True)
        test_stats = run_epoch(net, test_loader, kl_loss_fn, classification_loss_fn,
                                optimizer, opts, device, train=False)

        print(f'Train loss: {train_stats["loss"]:.4f}  acc_cape: {train_stats["acc_cape"]:.2f}%')
        print(f'Test  loss: {test_stats["loss"]:.4f}  acc_cape: {test_stats["acc_cape"]:.2f}%')

        for split, stats in [('train', train_stats), ('test', test_stats)]:
            writer.add_scalar(f'Loss/{split}', stats['loss'], epoch)
            writer.add_scalar(f'Accuracy_orig/{split}', stats['acc_orig'], epoch)
            writer.add_scalar(f'Accuracy_cape/{split}', stats['acc_cape'], epoch)

        # Save every epoch — cheap insurance against interruptions, unlike the
        # original repo's save_interval=5 which can lose up to 4 epochs of work.
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_ckpt)

        if test_stats['acc_cape'] > best_acc_cape:
            best_acc_cape = test_stats['acc_cape']
            torch.save(net.state_dict(), os.path.join(ckpt_dir, 'best.pth'))
            print(f'New best acc_cape: {best_acc_cape:.2f}% — saved best.pth')

    writer.close()
    print(f'\nDone. Best test acc_cape: {best_acc_cape:.2f}%')
    print(f'Checkpoints in: {ckpt_dir}')
    print(f'TensorBoard logs in: {run_dir}  (view with: tensorboard --logdir {run_dir})')


if __name__ == '__main__':
    main()
