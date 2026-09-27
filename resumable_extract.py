"""
Resumable extraction for CUB-200-2011, adapted from AIML-MED/CAPE's datasets.py
(CUB200._extract). The original processes all ~11,788 images in memory and only
writes to disk once, at the very end — so any interruption loses everything.

This version:
  - saves a checkpoint every CHECKPOINT_EVERY images, and resumes from the last
    checkpoint on restart instead of starting over from image 1
  - reads from EITHER a CUB_200_2011.tgz file OR a plain extracted folder
    (auto-detected) — Kaggle's "Add Input" datasets mount already-extracted,
    so no tgz is needed there; a tgz is only required for the tarfile-based
    path some local downloads (e.g. from data.caltech.edu) come as
  - takes --data_root / --output_dir so the same script works locally,
    on Kaggle (/kaggle/input is read-only, /kaggle/working is writable),
    or on Colab (Drive paths)

Usage:
    # Local: data_root contains CUB_200_2011.tgz OR an extracted CUB_200_2011/ folder
    python resumable_extract.py --data_root ~/cape-reproduction/data

    # Kaggle: input is read-only, so output must go elsewhere
    python resumable_extract.py \
        --data_root /kaggle/input/cub2002011 \
        --output_dir /kaggle/working/data

Run it to completion once; after that, the real CUB200 class (or this script,
re-run) will find the finished processed/train.pkl and processed/test.pkl and
skip extraction entirely.
"""

import os
import argparse
import pickle
import tarfile
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, default=os.path.expanduser('~/cape-reproduction/data'),
                    help='Folder containing either CUB_200_2011.tgz or an extracted CUB_200_2011/ folder. '
                         'Nested layouts (CUB_200_2011/CUB_200_2011/...) are auto-detected.')
    p.add_argument('--output_dir', type=str, default=None,
                    help='Where to write processed/train.pkl and test.pkl. Defaults to --data_root, '
                         'but MUST be set to a writable path (e.g. /kaggle/working/...) when '
                         'data_root is read-only, as on Kaggle.')
    p.add_argument('--checkpoint_every', type=int, default=500)
    return p.parse_args()


def find_source(data_root):
    """Returns ('tgz', path) or ('folder', path) — whichever form of the dataset is present."""
    tgz_path = os.path.join(data_root, 'CUB_200_2011.tgz')
    if os.path.isfile(tgz_path):
        return 'tgz', tgz_path

    # Look for an extracted CUB_200_2011 folder, handling one level of double-nesting
    candidate = os.path.join(data_root, 'CUB_200_2011')
    if os.path.isdir(candidate):
        if os.path.isfile(os.path.join(candidate, 'images.txt')):
            return 'folder', candidate
        nested = os.path.join(candidate, 'CUB_200_2011')
        if os.path.isfile(os.path.join(nested, 'images.txt')):
            return 'folder', nested

    # Some Kaggle mirrors mount images.txt directly at data_root with no CUB_200_2011/ wrapper
    if os.path.isfile(os.path.join(data_root, 'images.txt')):
        return 'folder', data_root

    raise FileNotFoundError(
        f'Could not find CUB_200_2011.tgz or an extracted CUB_200_2011 folder under {data_root}. '
        f'Run `ls {data_root}` and check the actual layout, then adjust --data_root.'
    )


def save_checkpoint(checkpoint_path, next_idx, train_data, train_labels, test_data, test_labels):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'next_idx': next_idx,
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        print(f"Resuming from checkpoint: {ckpt['next_idx']} images already done.")
        return ckpt['next_idx'], ckpt['train_data'], ckpt['train_labels'], ckpt['test_data'], ckpt['test_labels']
    return 0, [], [], [], []


def read_image_by_index(source_type, source, idx, rel_path):
    """Returns a PIL Image for one file, regardless of tgz vs. plain-folder source."""
    if source_type == 'tgz':
        member = source.extractfile(source.getmember(rel_path))
        if not member:
            raise RuntimeError(f'Could not read {rel_path} from tgz')
        return Image.open(member)
    else:  # folder
        # rel_path looks like 'CUB_200_2011/images/xxx.jpg' — strip the leading
        # 'CUB_200_2011/' since `source` already points at that folder.
        parts = rel_path.split('/', 1)
        local_path = os.path.join(source, parts[1]) if len(parts) > 1 else os.path.join(source, rel_path)
        return Image.open(local_path)


def main():
    args = parse_args()
    output_dir = args.output_dir or args.data_root
    processed_dir = os.path.join(output_dir, 'processed')
    checkpoint_path = os.path.join(processed_dir, 'extract_checkpoint.pkl')
    train_pkl = os.path.join(processed_dir, 'train.pkl')
    test_pkl = os.path.join(processed_dir, 'test.pkl')

    if os.path.isfile(train_pkl) and os.path.isfile(test_pkl):
        print("train.pkl and test.pkl already exist — nothing to do.")
        return

    os.makedirs(processed_dir, exist_ok=True)

    source_type, source_path = find_source(args.data_root)
    print(f'Source: {source_type} at {source_path}')

    if source_type == 'tgz':
        tar = tarfile.open(source_path, 'r:gz')
        images_txt = tar.extractfile(tar.getmember('CUB_200_2011/images.txt')).read().decode('utf-8').splitlines()
        split_txt = tar.extractfile(tar.getmember('CUB_200_2011/train_test_split.txt')).read().decode('utf-8').splitlines()
        source = tar
    else:
        with open(os.path.join(source_path, 'images.txt')) as f:
            images_txt = f.read().splitlines()
        with open(os.path.join(source_path, 'train_test_split.txt')) as f:
            split_txt = f.read().splitlines()
        source = source_path

    id2name = np.genfromtxt(images_txt, dtype=str)
    id2train = np.genfromtxt(split_txt, dtype=int)
    total = id2name.shape[0]
    print(f'Total images to process: {total}')

    next_idx, train_data, train_labels, test_data, test_labels = load_checkpoint(checkpoint_path)

    try:
        for _id in range(next_idx, total):
            rel_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = read_image_by_index(source_type, source, _id, rel_path)
            label = int(id2name[_id, 1][:3]) - 1

            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[_id, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

            done = _id + 1
            if done % 100 == 0:
                print(f'{done}/{total} images processed')

            if done % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_path, done, train_data, train_labels, test_data, test_labels)
                print(f'  -> checkpoint saved at {done} images')

    except KeyboardInterrupt:
        print('\nInterrupted — saving checkpoint before exiting...')
        save_checkpoint(checkpoint_path, _id, train_data, train_labels, test_data, test_labels)
        print(f'Saved at {_id} images. Re-run this script to resume from here.')
        raise

    except Exception as e:
        print(f'\nError during extraction: {e}')
        print('Saving checkpoint before exiting...')
        save_checkpoint(checkpoint_path, _id, train_data, train_labels, test_data, test_labels)
        print(f'Saved at {_id} images. Re-run this script to resume from here.')
        raise

    if source_type == 'tgz':
        source.close()

    print(f'Finished. Train: {len(train_data)}, Test: {len(test_data)}')
    with open(train_pkl, 'wb') as f:
        pickle.dump((train_data, train_labels), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(test_pkl, 'wb') as f:
        pickle.dump((test_data, test_labels), f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    print(f'Wrote train.pkl and test.pkl to {processed_dir}')


if __name__ == '__main__':
    main()
