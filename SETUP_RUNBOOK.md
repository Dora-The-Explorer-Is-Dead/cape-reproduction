# Setup Runbook — CAPE Reproduction

Everything needed to get from a fresh machine to a passing sanity check. Follow it in order. This has been corrected against what actually worked, not the first attempt — some sub-steps below fixed real mistakes from earlier tries.

**Use Git Bash for all commands below**, not Windows cmd.exe.

---

## 1. Clone the repo

```bash
git clone https://github.com/Dora-The-Explorer-Is-Dead/cape-reproduction
cd cape-reproduction
```

If "repository not found," check the exact URL on GitHub.com — usernames only contain letters, numbers, and hyphens (no underscores).

## 2. Set up the Python environment

```bash
python -m venv venv
source venv/Scripts/activate
```
> On Windows the activation script is at `venv/Scripts/activate`, not `venv/bin/activate`.

You'll see `(venv)` at the start of your prompt once it's active.

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm tensorboard jupyter
```
> If `pip install ...` gives `Permission denied`, use `python -m pip install ...` instead.
>
> If it still fails, check `echo $HOME` for "OneDrive" — a OneDrive-synced project folder can lock files mid-write. Move the project to a non-synced folder if so, and recreate the venv there.

## 3. Get the CUB-200-2011 dataset

The original Caltech host's links redirect to broken HTML pages — don't bother with `curl`/`wget` against `data.caltech.edu` or `vision.caltech.edu`.

**Use the Kaggle mirror, downloaded through the browser (not the API/CLI — simpler and avoids token setup issues):**

1. Go to `https://www.kaggle.com/datasets/wenewone/cub2002011` (or search "CUB 200 2011" on Kaggle if that page has moved) and log in.
2. Click **Download**, choose curl/direct download as offered, and let it save to your normal Downloads folder.
3. Move it into place and extract with Python (Git Bash's built-in `unzip` is too old for this file's Zip64 format and will fail — don't use it):
```bash
mv ~/Downloads/cub2002011.zip ~/cape-reproduction/data/
cd ~/cape-reproduction/data
python -c "import zipfile; zipfile.ZipFile('cub2002011.zip').extractall('CUB_200_2011')"
```
4. Check what you got:
```bash
ls CUB_200_2011
```
This particular upload **double-nests** the dataset (a `CUB_200_2011` folder inside `CUB_200_2011`) alongside two unused folders (`cvpr2016_cub/` — unrelated caption dataset, `segmentations/` — unused masks, both safe to ignore). Confirm the nested folder has the real files:
```bash
ls CUB_200_2011/CUB_200_2011
```
You should see `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `classes.txt`. **Don't try to flatten/rename this folder** — Windows file locks can cause `Permission denied` on the rename. Just reference the nested path directly, as below.

5. The official CAPE repo's dataset loader (`datasets.py`) expects a `CUB_200_2011.tgz` file directly in the data root, not a loose folder. Repack it — **this exact command matters**, an earlier version of this step had the folder-prefix wrong and caused a `KeyError` later on:
```bash
cd ~/cape-reproduction/data
tar -czf CUB_200_2011.tgz -C CUB_200_2011 CUB_200_2011/images.txt CUB_200_2011/train_test_split.txt CUB_200_2011/images
```
6. **Verify the internal paths are correct before moving on** — this is the step that catches the bug the hard way if skipped:
```bash
tar -tzf CUB_200_2011.tgz | head -5
```
You must see paths prefixed with `CUB_200_2011/`, e.g.:
```
CUB_200_2011/images.txt
CUB_200_2011/train_test_split.txt
CUB_200_2011/images/001.Black_footed_Albatross/...
```
If you instead see bare `images.txt` with no prefix, the tgz was built wrong and extraction will fail with `KeyError: "filename 'CUB_200_2011/images.txt' not found"` later — delete it and redo step 5.

Full details also live in `data/README.md` inside the repo.

## 4. Clone the official CAPE reference repo

Separate clone, **outside** `cape-reproduction` — reference material only, not part of our submission.

```bash
cd ~
git clone https://github.com/AIML-MED/CAPE.git cape-reference
cd cape-reference
```

Confirm the pretrained checkpoints are real files, not tiny Git LFS pointers:
```bash
ls -lh saved_models
```
Expect `cub_resnet50_PF.pth` and `cub_resnet50_TS.pth`, each ~94MB. If only a few KB, run `git lfs install && git lfs pull`.

## 5. Extract the dataset (resumable — do this before the notebook, not inside it)

The official `datasets.py` only saves to disk once, at the very end of processing all ~11,788 images — so any interruption (sleep, disconnect, crash) loses all progress and you start over from zero. Use the checkpointed version instead, which saves every 500 images and resumes automatically. `resumable_extract.py` lives in the repo root — see `PROVENANCE.md` for what it changes vs. the original.

```bash
cd ~/cape-reproduction
source venv/Scripts/activate
python resumable_extract.py
```

Before running, disable sleep so a long extraction isn't interrupted: **Settings → System → Power & battery → Screen and sleep → set to Never** while plugged in.

Run this directly in the terminal (not inside a Jupyter cell) so a browser tab issue can't affect it. If it does get interrupted, just re-run the same command — it picks up from the last checkpoint instead of starting over. It's done when you see:
```
Wrote train.pkl and test.pkl — the real CUB200 class will now load instantly.
```

This only needs to happen once — after this, both the sanity check notebook and any real training run will load the dataset near-instantly from the cached `.pkl` files.

## 6. Run the sanity check

```bash
cd ~/cape-reproduction
source venv/Scripts/activate
python -m pip install -r ~/cape-reference/requirements.txt
jupyter notebook experiments/01_sanity_check.ipynb
```

Run every cell top to bottom with **Shift+Enter**, waiting for `[*]` to turn into a number before running the next cell — don't skip ahead or run cells out of order, since later cells depend on variables from earlier ones.

Expected outputs:
- Batch loads near-instantly (dataset is already extracted from step 5)
- `cape cls_map shape: torch.Size([4, 200, 14, 14])` followed by `Shape checks passed.`
- 10 lines of `step N: loss=...`, with `All losses finite: True` and `Loss decreased overall: True`
- A small plot showing the loss trending downward

Don't move on to a real training run until every cell above passes cleanly.

## 7. Log what you did

Every time you add or adapt a file, add a row to `PROVENANCE.md` immediately — don't wait until report time to reconstruct it. Current entries as of this runbook:

| File / component | Status | Source |
|---|---|---|
| `resumable_extract.py` | Adapted | AIML-MED/CAPE `datasets.py::_extract()` — added checkpointing every 500 images |
| `experiments/01_sanity_check.ipynb` | Written by us | — |

---

## If something doesn't match this runbook

Environments drift — different Kaggle re-uploads, path differences, OS quirks. If a step fails in a way not covered above, don't silently work around it and move on. Post the exact command and exact error in the group chat, and once it's resolved, **update this runbook** so the next person doesn't hit the same thing.
