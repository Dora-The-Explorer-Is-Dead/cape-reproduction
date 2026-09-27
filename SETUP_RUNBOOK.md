# Setup Runbook — CAPE Reproduction

This is the lean path — what to actually do, and why each step exists. Earlier attempts hit dead ends (broken Caltech download links, a Windows unzip tool too old for this dataset, a bad `tar` command); none of that is reproduced here, only what actually works.

**Two environments are used together, on purpose:**
- **Local machine** — where the git repo lives, where you write/edit code, where you push to GitHub. This is your source of truth.
- **Kaggle Notebooks** — where the dataset lives and where training actually runs, because it has a free GPU and the dataset mounts pre-extracted with no download step. Code is pulled from GitHub into Kaggle, run there, and results (checkpoints, logs) are brought back manually.

---

**Contents:** [1](#1-clone-the-repo-local) Clone repo · [2](#2-set-up-the-python-environment-local) Local env · [3](#3-clone-the-official-cape-reference-repo-local) Reference repo · [4](#4-understand-tgz-briefly) `.tgz` explained · [5](#5-get-the-dataset--via-kaggle-recommended-path) Dataset (Kaggle) · [6](#6-turn-on-the-gpu-kaggle) GPU · [7](#7-clone-both-repos-into-the-kaggle-session) Kaggle clone · [8](#8-extract-the-dataset-into-pickles-kaggle) Extraction · [9](#9-understand-pf-briefly) PF explained · [10](#10-run-the-sanity-check-first--always-before-any-real-training) Sanity check · [11](#11-run-the-small-subset-training-on-kaggle--before-the-full-run) Subset run · [12](#12-run-the-full-reproduction-kaggle) Full training · [13](#13-bring-results-back) Results → Hugging Face · [14](#14-log-everything-as-you-go) Provenance · [15](#15-generate-heatmaps-and-metrics-kaggle) Heatmaps & metrics · [16](#16-recovering-from-checkpoints--what-to-run-after-any-interruption) Checkpoint recovery

---

## 1. Clone the repo (local)

```bash
git clone https://github.com/Dora-The-Explorer-Is-Dead/cape-reproduction
cd cape-reproduction
```

**Use Git Bash on Windows**, not cmd.exe — cmd doesn't understand `mkdir -p`, `touch`, or forward-slash paths the way every command below assumes.

## 2. Set up the Python environment (local)

```bash
python -m venv venv
source venv/Scripts/activate        # Windows: Scripts, not bin — that's the Mac/Linux path
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn tqdm tensorboard jupyter
```

*Why `python -m pip install` instead of plain `pip install`:* Git Bash on Windows sometimes can't execute `pip.exe` directly (`Permission denied`), but can always run `python.exe`, which can invoke pip as a module. Use this form everywhere.

You only need this environment locally for writing/testing code in small pieces (e.g. the sanity check on a tiny batch). Real training happens on Kaggle, which has its own environment.

## 3. Clone the official CAPE reference repo (local)

A separate clone, **outside** `cape-reproduction` — it's reference material we read from and adapt, not part of our submission.

```bash
cd ~
git clone https://github.com/AIML-MED/CAPE.git cape-reference
cd cape-reference
ls -lh saved_models
```

*Why:* the authors provide the actual paper architecture (`models/model.py`), the actual distillation loss (`models/losses.py`), and — usefully — their own pretrained CUB checkpoints (`cub_resnet50_TS.pth`, `cub_resnet50_PF.pth`, ~94MB each). Confirm those two files are really ~94MB and not a few KB (a few KB would mean Git LFS didn't pull the real weights — if so, run `git lfs install && git lfs pull`).

*What we don't do:* run their `main.py` directly. We understood their config and model/loss code, then wrote our own scripts (`resumable_extract.py`, `src/train_pf.py`) adapted from it — logged in `PROVENANCE.md`.

## 4. Understand `.tgz`, briefly

You'll see this word a lot. A `.tgz` (`.tar.gz`) is a folder of files bundled into one archive (`tar`) and then compressed (`gz`). The authors' dataset-loading code (`datasets.py`) expects the CUB dataset as exactly this: one `CUB_200_2011.tgz` file, with files inside it at the exact internal path `CUB_200_2011/images.txt` etc. — not a loose folder of already-extracted files. This detail caused real breakage earlier (a `KeyError` when an internal path didn't match), which is why our own extraction script (`resumable_extract.py`) works from *either* a tgz *or* an already-extracted folder — see step 6.

## 5. Get the dataset — via Kaggle (recommended path)

**Don't download the dataset to your local machine.** The original Caltech host's links are broken (redirect to HTML error pages), and even a working download means fighting Windows' outdated `unzip` and manually flattening a double-nested folder. Kaggle skips all of this:

1. Go to `kaggle.com` → **Code** → **New Notebook**
2. Right sidebar → **+ Add Input** → search "CUB 200 2011" → add a dataset that includes `images.txt`, `train_test_split.txt`, and an `images/` folder (preview it before adding to check)
3. It mounts pre-extracted at `/kaggle/input/<dataset-name>/` — no download or unzip needed at all
4. This is the exact link we're using: [CUB_200_2011](https://www.kaggle.com/datasets/wenewone/cub2002011)
```python
!ls /kaggle/input # this will tell you what folder is inside it which you will replace <dataset-name> with in the line below.
!ls /kaggle/input/<dataset-name> # keep going down the directories till you reach CUB_200_2011 which will have another CUB_200_2011 in it as the line below demonstrates
# execute this third line once your directory looks like this:
!ls /kaggle/input/<dataset-name>/CUB_200_2011/CUB_200_2011   # if double-nested
```
You're looking for `images.txt`, `images/`, `train_test_split.txt`, `classes.txt` — note however many folders deep they actually are, you'll pass that exact path as `--data_root` in step 6.

*Why Kaggle over local:* the whole point of a dataset host is to skip download/extraction pain — Kaggle's mount does that for you. If you ever do need the dataset locally too (e.g. offline work), the same Kaggle "Download" button works, but expect the same double-nesting and use Python's zipfile (`python -c "import zipfile; zipfile.ZipFile('<file>.zip').extractall('CUB_200_2011')"`) instead of `unzip`, which is too old on Git Bash for this file's format.

## 6. Turn on the GPU (Kaggle)

Right sidebar → **Session options** (a gear/settings icon if not labeled) → **Accelerator** → **GPU T4 x2**. First time, Kaggle may ask for phone verification — complete it, it's one-time.

Confirm it's active:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
```

## 7. Clone both repos into the Kaggle session

```python
!git clone https://github.com/Dora-The-Explorer-Is-Dead/cape-reproduction.git
!git clone https://github.com/AIML-MED/CAPE.git cape-reference
%cd cape-reproduction
!pip install -r /kaggle/working/cape-reference/requirements.txt
```

## 8. Extract the dataset into pickles (Kaggle)

```python
!python resumable_extract.py \
    --data_root /kaggle/input/<dataset-name>/CUB_200_2011/CUB_200_2011 \
    --output_dir /kaggle/working/data
```

*Why this script and not the authors' own extraction:* their `CUB200` class does the full ~11,788-image extraction in memory and only saves to disk once, at the very end — any interruption loses everything. `resumable_extract.py` checkpoints every 500 images and resumes automatically if interrupted. It also reads from a plain folder (what Kaggle gives you) as well as a tgz, so no repacking step is needed here at all — that repacking was only ever required for the *local* Windows attempt, and doesn't apply on Kaggle. `/kaggle/input` is read-only, which is why `--output_dir` points at `/kaggle/working` instead.

This produces `processed/train.pkl` and `processed/test.pkl` under `--output_dir`, used by every training run from here on.

## 9. Understand PF, briefly

CAPE has two training modes:
- **TS (Training from Scratch)** — backbone + classifier + CAPE layer all train together.
- **PF (Post-Fitting)** — start from an already-trained model (the authors' TS checkpoint), freeze everything except the small CAPE layer, and fine-tune just that. Far cheaper — this is what we implement.

`src/train_pf.py` does this: loads the TS checkpoint, freezes the backbone and the vanilla classifier, and trains only the CAPE head + three small temperature parameters, using a distillation loss that teaches CAPE's output to match the frozen classifier's predictions.

## 10. Run the sanity check first — always, before any real training

Locally, in Jupyter (`experiments/01_sanity_check.ipynb`), on a tiny in-memory batch: confirms shapes are right (14×14×200 class activation maps) and the loss is finite and decreasing over a few steps, using CPU. Cheap and fast — no need for a GPU or the full dataset for this part. Do this once per code change to the model/loss/training logic, before spending GPU time on Kaggle.

## 11. Run the small subset training on Kaggle — before the full run

```python
!python src/train_pf.py \
    --subset_fraction 0.05 --num_epochs 3 --run_name subset_check \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results
```

*Why:* catches any bug that only appears across multiple batches/epochs (not just the sanity check's single batch), in minutes instead of hours. Expect it to actually learn something (test accuracy well above the 0.5% random-chance baseline for 200 classes) since it's fine-tuning from an already-good checkpoint, not training from scratch.

## 12. Run the full reproduction (Kaggle)

```python
!python src/train_pf.py \
    --subset_fraction 1.0 --num_epochs 30 --run_name full_pf \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results
```
Matches the authors' config: `lr=1e-3`, `T_kld=2`, SGD, 30 epochs. Checkpoints every epoch (`latest.pth`) so an interrupted session can resume by re-running the identical command. Watch live via TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/results/logs/full_pf
```

## 13. Bring results back

Don't rely on Kaggle's own download UI — its `FileLink` helper and the
Output tab's download button both regularly fail (404s, stale listings) on
files this size, and `/kaggle/working` itself doesn't survive indefinitely
between sessions. Instead, push straight from Kaggle to Hugging Face, server
to server:

```python
!pip install huggingface_hub
from huggingface_hub import login, HfApi, create_repo

login(token="<your-hf-write-token>")   # huggingface.co → Settings → Access Tokens → New token (Write)
create_repo("<your-username>/cape-cub-pf", exist_ok=True)

api = HfApi()
api.upload_file(path_or_fileobj="/kaggle/working/results/checkpoints/full_pf/best.pth", path_in_repo="best.pth", repo_id="<your-username>/cape-cub-pf")
api.upload_folder(folder_path="/kaggle/working/results/logs/full_pf", path_in_repo="logs/full_pf", repo_id="<your-username>/cape-cub-pf")
```

Reference the resulting `https://huggingface.co/<your-username>/cape-cub-pf`
link from your README rather than committing the checkpoint to git (it's
90MB+, same reasoning as the authors' own checkpoints). Do this immediately
after training finishes, in the same live session — don't close the tab
first and try to recover the files afterward (see Section 16 for what
happens if you do).

## 14. Log everything as you go

Every new or adapted file gets a row in `PROVENANCE.md`

---

## 15. Generate heatmaps and metrics (Kaggle)

```python
!pip install huggingface_hub
from huggingface_hub import hf_hub_download
best_pth_path = hf_hub_download(repo_id="<your-username>/cape-cub-pf", filename="best.pth")

!python src/explain.py \
    --checkpoint {best_pth_path} \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results/figures \
    --num_images 5

!python src/metrics.py \
    --checkpoint {best_pth_path} \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results/tables \
    --num_images 5794 \
    --checkpoint_every 100
```

`explain.py` produces qualitative comparison figures (CAM, Grad-CAM,
Grad-CAM++, CAPE, mu-CAPE side by side) for a handful of test images.
`metrics.py` computes AD, IC, ADD, ADCC, mIoU, and BC across the full test
set — this is slow (roughly 20+ forward/backward passes per image across
five methods), so it checkpoints every 100 images by default. If it's
interrupted, rerun the identical command and it resumes automatically (see
Section 16 below).

**Push results straight to Hugging Face** rather than fighting Kaggle's
download UI (its `FileLink` and Output-tab download button both regularly
fail on larger files):
```python
from huggingface_hub import login, HfApi
login(token="<your-hf-write-token>")
api = HfApi()

api.upload_folder(folder_path="/kaggle/working/results/figures", path_in_repo="figures", repo_id="<your-username>/cape-cub-pf")
api.upload_file(path_or_fileobj="/kaggle/working/results/tables/metrics_results.csv", path_in_repo="metrics_results.csv", repo_id="<your-username>/cape-cub-pf")
```

## 16. Recovering from checkpoints — what to run after any interruption

Three scripts in this repo checkpoint their own progress, specifically
because Kaggle sessions have reset or lost `/kaggle/working` entirely more
than once during this project. **In every case, the fix is the same: rerun
the exact same command you were running before the interruption.** Each
script detects its own checkpoint file and resumes automatically — you do
not need to pass any special "resume" flag.

**`resumable_extract.py`** (dataset extraction — checkpoints every 500
images to `<output_dir>/processed/extract_checkpoint.pkl`):
```python
!python resumable_extract.py \
    --data_root /kaggle/input/<dataset-path> \
    --output_dir /kaggle/working/data
```
If this prints `Resuming from checkpoint: N images already done`, it worked.
If instead you get `_pickle.UnpicklingError: pickle data was truncated`, the
checkpoint file itself was cut off mid-write by the interruption and is
unrecoverable — delete it and the (likely also incomplete) pickles, then
start over from image 0:
```python
!rm -f /kaggle/working/data/processed/extract_checkpoint.pkl
!rm -f /kaggle/working/data/processed/train.pkl /kaggle/working/data/processed/test.pkl
!python resumable_extract.py --data_root /kaggle/input/<dataset-path> --output_dir /kaggle/working/data
```

**`train_pf.py`** (PF training — checkpoints every epoch to
`<output_dir>/checkpoints/<run_name>/latest.pth`):
```python
!python src/train_pf.py \
    --subset_fraction 1.0 --num_epochs 30 --run_name full_pf \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results
```
Using the **same `--run_name`** is what triggers resume — it looks for
`latest.pth` under that run's checkpoint folder and continues from
`epoch + 1` if found. A different `--run_name` starts a fresh run instead.
Unlike extraction, there's no separate "truncated checkpoint" failure mode
here, since `torch.save` either completes or the file simply won't load —
if `latest.pth` fails to load, delete it and restart that run from epoch 0.

**`metrics.py`** (evaluation — checkpoints every `--checkpoint_every` images,
default 100, to `<output_dir>/metrics_checkpoint.pkl`):
```python
!python src/metrics.py \
    --checkpoint {best_pth_path} \
    --processed_dir /kaggle/working/data/processed \
    --output_dir /kaggle/working/results/tables \
    --num_images 5794
```
Same rule: rerun the identical command, it prints `Resuming from checkpoint:
N/5794 images already done` if a checkpoint exists. The checkpoint is
deleted automatically once a run completes successfully, so a resume prompt
appearing on a fresh run you didn't expect means a previous run didn't
finish cleanly — check for a stale checkpoint file before assuming something
else is wrong.

**If `/kaggle/working` itself is gone** (not just one script's checkpoint,
but the whole working directory reset — this has happened after toggling
the GPU accelerator mid-session), none of the above checkpoints survive,
since they lived inside that wiped folder. In that case:
- Your **trained model checkpoint is safe** — it's on Hugging Face, pull it
  back with `hf_hub_download` (Section 15) rather than retraining.
- Your **extracted dataset pickles are gone** — re-run `resumable_extract.py`
  from scratch (Section 8).
- Avoid this going forward by turning the GPU accelerator on **before**
  cloning/extracting/training anything, and not toggling it again mid-session.


