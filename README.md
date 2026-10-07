# xlzd-agent

Reproduce the XLZD RESuM surrogate-model results by pointing a coding agent at a runbook and letting it drive. Clone the repo, set up the environment, and either run the pipeline yourself or hand the runbook to an agent — both produce the same trained models, validation metrics, and plots.

**You do not need to know anything about RESuM, dark matter detectors, or this codebase to use this repo.** Everything needed to produce the results is included: the model code, a bundled sample dataset, the environment specification, and a step-by-step runbook written to be followed by a person or an AI agent.

---

## What this produces

Running the pipeline end to end trains a surrogate model that predicts a detector's performance from its geometry, then validates it and draws the standard plots. The final outputs are:

- a trained CNP model and a trained MF-GP model,
- validation metrics (R², RMSE, coverage),
- five plots under `outputs/plots/` — the CNP shell-occupancy fits, the MF-GP validation points and parity plot, and the MF-GP mean/uncertainty surface over geometry space.

If you want to understand *what* those things are, read [`BACKGROUND.md`](BACKGROUND.md). If you only want to reproduce the results, you can skip it — the runbook does not assume you've read it.

---

## Prerequisites

You need:

1. **Python 3.12** and [`uv`](https://github.com/astral-sh/uv) (a fast Python environment manager). Install `uv` with:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Git with [Git LFS](https://git-lfs.com)**, because the sample dataset is stored with LFS. Install it once per machine:
   ```bash
   # macOS
   brew install git-lfs
   # Debian/Ubuntu
   sudo apt-get install git-lfs
   # then, once:
   git lfs install
   ```
3. **(Optional) an NVIDIA GPU.** The pipeline runs on CPU if no GPU is present — slower, but it completes. GPU setup notes are in [`SETUP.md`](SETUP.md), including the fix for recent (Blackwell / RTX 50-series) cards.

---

## Quick start

```bash
# 1. Clone (Git LFS pulls the sample data automatically)
git clone https://github.com/<your-username>/xlzd-agent.git
cd xlzd-agent

# 2. Set up the environment (creates a venv, installs pinned deps)
bash setup_env.sh
source .venv/bin/activate

# 3. Verify the data and deps are in place
ls data/train_data/file_manifest.csv data/val_data/validation_manifest.csv
python3 -c "import torch, numpy, pandas, h5py, matplotlib; print('deps ok')"
```

Then **either** run it yourself:

```bash
cd positional_shells
# follow RUNBOOK.md, steps 1–6, in order
```

**or** hand it to a coding agent (see below).

---

## Running it with an agent

This repo is built so an AI coding agent can run the whole pipeline on its own. The runbook ([`RUNBOOK.md`](RUNBOOK.md)) is written as a sequence of steps, each with a command and a check that tells the agent whether the step succeeded before moving on.

To do this, open the repo in your agent (for example Claude Code), and give it:

> Read RUNBOOK.md and run the pipeline end to end. Run the steps in order, and after each step verify its expected output before continuing. If a step fails its check, stop and report the error.

The agent will prepare the data, train the CNP, run predictions, train and validate the MF-GP, and generate the plots — checking its own work at each stage. When it finishes, the trained models, metrics, and plots are under `outputs/`.

The reason this works without supervision is that each step in the runbook declares a **verifiable postcondition** — a specific file that must exist or a specific line that must print. The agent doesn't guess whether a step worked; it checks. That's what makes it safe to run unattended.

---

## What's in here

| Path | What it is |
|------|------------|
| `README.md` | This file. |
| `RUNBOOK.md` | The step-by-step pipeline procedure, for a person or an agent. |
| `SETUP.md` | Environment details, including GPU setup and the CPU fallback. |
| `BACKGROUND.md` | Optional: what RESuM / CNP / MF-GP are, for context. |
| `setup_env.sh` | One-shot environment setup script. |
| `requirements.txt` | Pinned Python dependencies. |
| `data/` | The bundled sample dataset (`train_data/`, `val_data/`), tracked by Git LFS. |
| `configs/` | The three TOML configs that drive the run. |
| `positional_shells/` | The pipeline entry points; run commands from here. |
| `common/`, `cnp.py`, `mfgp.py`, `shells/` | The model code. |
| `generate_plots.py` | Produces the plots from a completed run (runbook step 6). |

---

## About the bundled data

The dataset in `data/` is a **faithful sample** of the real training and validation data — the same files, the same format — trimmed only in quantity so the repo stays a reasonable size:

- all 5 high-fidelity training geometries, included unchanged,
- 50 of the ~200 low-fidelity training geometries, included unchanged,
- all validation geometries, downsampled to 10,000 events each (whole events, so each kept event keeps all its deposits).

This is enough to run the full pipeline and reproduce representative plots and metrics. It is a demonstration dataset: the results will have the right character, but exact numbers depend on the full dataset. To run against the complete data, replace `data/train_data` and `data/val_data` with the full set (or symlinks to it) and rerun — nothing else changes.
