# Setup Runbook — CAPE Reproduction

For Ayaan and Taher: this is the exact sequence Umama went through to get the project running. Follow it in order — several steps hit real gotchas (noted below) that cost time to figure out the first time, so don't skip the notes.

**Use Git Bash for everything below**, not Windows cmd.exe — it's what all these commands are written for, and it avoids a whole class of path/syntax problems.

---

## 1. Clone the repo

```bash
git clone https://github.com/Dora-The-Explorer-Is-Dead/cape-reproduction
cd cape-reproduction
```

If you get "repository not found," double check the URL against what's actually on GitHub (usernames don't contain underscores — only letters, numbers, hyphens).

## 2. Set up the Python environment

```bash
python -m venv venv
source venv/Scripts/activate
```

> **Note:** On Windows, the activation script lives at `venv/Scripts/activate`, not `venv/bin/activate` (that's the Mac/Linux path). If `source venv/bin/activate` fails, use `Scripts` instead.

You'll know it worked when your prompt shows `(venv)` at the start.

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm tensorboard jupyter
```

> **Note:** If `pip install ...` gives `Permission denied`, don't fight it — use `python -m pip install ...` instead (calls pip as a module through Python rather than running `pip.exe` directly, which Git Bash sometimes can't execute).
>
> If that *still* fails, check whether your project folder is inside a OneDrive-synced path (`echo $HOME` — look for "OneDrive" in the result). OneDrive can lock files mid-write and break venvs. If so, move the whole project to a non-synced folder and recreate the venv there.

## 3. Get the CUB-200-2011 dataset

The original Caltech host's direct download links are broken/redirect to HTML pages instead of the file — don't bother with `curl`/`wget` against `data.caltech.edu` or `vision.caltech.edu`, they won't work cleanly.

**Use the Kaggle mirror instead:**
1. Go to `https://www.kaggle.com/datasets/wenewone/cub2002011` (or search "CUB 200 2011" on Kaggle if that's moved)
2. Click **Download** — this needs a free Kaggle account, but no API token/CLI setup needed if you just use the website button
3. Move the zip into place and extract:
```bash
mv ~/Downloads/cub2002011.zip ~/cape-reproduction/data/
cd ~/cape-reproduction/data
python -c "import zipfile; zipfile.ZipFile('cub2002011.zip').extractall('CUB_200_2011')"
```

> **Note:** Git Bash's built-in `unzip` is too old to handle this file (it's Zip64 format) and will fail with "End-of-central-directory signature not found." Use the Python one-liner above instead, or Windows' own File Explorer → right-click → Extract All.

4. This Kaggle upload nests the real dataset one level deeper than expected,
   at CUB_200_2011/CUB_200_2011/, alongside two unused folders (cvpr2016_cub/
   and segmentations/ — irrelevant to CAPE, safe to ignore). No need to
   physically flatten it — just point commands at the nested path.

5. Repack it into a .tgz from the nested path (run from inside data/) — the
   official CAPE repo's dataset loader expects a CUB_200_2011.tgz file
   sitting in the data root:
   cd ~/cape-reproduction/data
   tar -czf CUB_200_2011.tgz -C CUB_200_2011/CUB_200_2011 images.txt train_test_split.txt images

6. Verify it worked:
   tar -tzf CUB_200_2011.tgz | head -5
   Should show paths starting with CUB_200_2011/, e.g. CUB_200_2011/images.txt

Full details are also in `data/README.md` inside the repo.

## 4. Clone the official CAPE reference repo

This is a separate clone, **outside** `cape-reproduction` — it's reference material, not part of our submission.

```bash
cd ~
git clone https://github.com/AIML-MED/CAPE.git cape-reference
cd cape-reference
```

Confirm the pretrained checkpoints downloaded as real files, not tiny Git LFS pointers:
```bash
ls -lh saved_models
```
You should see `cub_resnet50_PF.pth` and `cub_resnet50_TS.pth`, each ~94MB. If they're only a few KB, run `git lfs install && git lfs pull`.

## 5. Run the sanity check

```bash
cd ~/cape-reproduction
source venv/Scripts/activate
python -m pip install -r ~/cape-reference/requirements.txt
jupyter notebook experiments/01_sanity_check.ipynb
```

Run every cell top to bottom. It should confirm:
- The model loads and the TS checkpoint restores correctly
- A forward pass produces 14x14x200 class activation maps
- The distillation loss is finite and decreases over a few steps on one batch

Don't move on to a real training run until this notebook fully passes.

---

## If something doesn't match this runbook

Environments drift (different Kaggle re-uploads, path differences, etc.) — if a step fails in a way not covered by the notes above, don't silently work around it and move on. Post in the group chat with the exact command and exact error output, and update this runbook once it's resolved so the next person doesn't hit the same thing.
