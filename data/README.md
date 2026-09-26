# Dataset: CUB-200-2011

Downloaded via Kaggle (browser download) rather than the original Caltech host,
due to broken redirect links on data.caltech.edu at the time of this project.

Source: https://www.kaggle.com/datasets/wenewone/cub2002011

## Setup
1. Download the zip from the Kaggle URL above (requires a free Kaggle account)
2. Extract it into data/CUB_200_2011/
3. This Kaggle upload nests the real dataset one level deeper than expected,
   at CUB_200_2011/CUB_200_2011/, alongside two unused folders (cvpr2016_cub/
   and segmentations/ — irrelevant to CAPE, safe to ignore).
4. Repack it into a tgz from the nested path:
   tar -czf CUB_200_2011.tgz -C CUB_200_2011/CUB_200_2011 images.txt train_test_split.txt images

Dataset stats: 200 classes, 5,994 training images, 5,794 test images, 11,788 total.
Note: raw data is gitignored — this repo does not contain the dataset itself.