"""
Resumable extraction for CUB-200-2011, adapted from AIML-MED/CAPE's datasets.py
(CUB200._extract). The original processes all ~11,788 images in memory and only
writes to disk once, at the very end — so any interruption loses everything.

This version saves a checkpoint every CHECKPOINT_EVERY images, and resumes from
the last checkpoint on restart instead of starting over from image 1.

Usage:
    python resumable_extract.py

Run it to completion once; after that, the real CUB200 class in datasets.py
will find the finished processed/train.pkl and processed/test.pkl and skip
extraction entirely, exactly as it would after its own normal run.
"""

import os
import pickle
import tarfile
import numpy as np
from PIL import Image

# ---- Adjust these two if your paths differ ----
DATA_ROOT = os.path.expanduser(r'~/cape-reproduction/data')
CHECKPOINT_EVERY = 500
# -------------------------------------------------

CUB_TGZ_PATH = os.path.join(DATA_ROOT, 'CUB_200_2011.tgz')
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed')
CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, 'extract_checkpoint.pkl')
TRAIN_PKL = os.path.join(PROCESSED_DIR, 'train.pkl')
TEST_PKL = os.path.join(PROCESSED_DIR, 'test.pkl')


def save_checkpoint(next_idx, train_data, train_labels, test_data, test_labels):
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump({
            'next_idx': next_idx,
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint():
    if os.path.isfile(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            ckpt = pickle.load(f)
        print(f"Resuming from checkpoint: {ckpt['next_idx']} images already done.")
        return ckpt['next_idx'], ckpt['train_data'], ckpt['train_labels'], ckpt['test_data'], ckpt['test_labels']
    return 0, [], [], [], []


def main():
    if os.path.isfile(TRAIN_PKL) and os.path.isfile(TEST_PKL):
        print("train.pkl and test.pkl already exist — nothing to do.")
        return

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    tar = tarfile.open(CUB_TGZ_PATH, 'r:gz')
    images_txt = tar.extractfile(tar.getmember('CUB_200_2011/images.txt'))
    train_test_split_txt = tar.extractfile(tar.getmember('CUB_200_2011/train_test_split.txt'))

    images_txt = images_txt.read().decode('utf-8').splitlines()
    train_test_split_txt = train_test_split_txt.read().decode('utf-8').splitlines()

    id2name = np.genfromtxt(images_txt, dtype=str)
    id2train = np.genfromtxt(train_test_split_txt, dtype=int)
    total = id2name.shape[0]
    print(f'Total images to process: {total}')

    next_idx, train_data, train_labels, test_data, test_labels = load_checkpoint()

    try:
        for _id in range(next_idx, total):
            image_path = 'CUB_200_2011/images/' + id2name[_id, 1]
            image = tar.extractfile(tar.getmember(image_path))
            if not image:
                raise RuntimeError(f'Could not read {image_path}')
            image = Image.open(image)
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

            if done % CHECKPOINT_EVERY == 0:
                save_checkpoint(done, train_data, train_labels, test_data, test_labels)
                print(f'  -> checkpoint saved at {done} images')

    except KeyboardInterrupt:
        print('\nInterrupted — saving checkpoint before exiting...')
        save_checkpoint(_id, train_data, train_labels, test_data, test_labels)
        print(f'Saved at {_id} images. Re-run this script to resume from here.')
        raise

    except Exception as e:
        print(f'\nError during extraction: {e}')
        print('Saving checkpoint before exiting...')
        save_checkpoint(_id, train_data, train_labels, test_data, test_labels)
        print(f'Saved at {_id} images. Re-run this script to resume from here.')
        raise

    tar.close()

    print(f'Finished. Train: {len(train_data)}, Test: {len(test_data)}')
    with open(TRAIN_PKL, 'wb') as f:
        pickle.dump((train_data, train_labels), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TEST_PKL, 'wb') as f:
        pickle.dump((test_data, test_labels), f, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print('Wrote train.pkl and test.pkl — the real CUB200 class will now load instantly.')


if __name__ == '__main__':
    main()
