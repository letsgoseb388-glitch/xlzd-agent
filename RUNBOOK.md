# Running the Shell RESuM Pipeline

This document describes how to run the full shell-based RESuM pipeline from a clean checkout to the final MF-GP validation metrics and plots. It is written to be followed step by step, either by a person or by a coding agent. Each step gives the command to run, what it needs beforehand, and what it should produce. **If a step does not produce the described output, stop and report the error rather than continuing** — each stage depends on the outputs of the one before it.

## What this pipeline does

The pipeline turns raw detector simulation data into a trained surrogate model that predicts a detector's fiducial containment as a function of its geometry. It runs in six stages:

1. Prepare the raw simulation events into shell-labeled HDF5 blocks.
2. Train the Conditional Neural Process (CNP) on those blocks.
3. Run the trained CNP over the training and validation splits to produce per-geometry shell predictions.
4. Train the Multi-Fidelity Gaussian Process (MF-GP) on the CNP training predictions.
5. Evaluate the MF-GP on the held-out validation predictions and report metrics.
6. Generate the CNP and MF-GP visualizations from the outputs.

Stages 2 through 6 all depend on the outputs of the stage before them, so they must run in order.

## Prerequisites

Before running anything, confirm the following are in place.

**Repository and environment.** You have cloned this repo and set up the environment per `README.md` (run `bash setup_env.sh`, then `source .venv/bin/activate`). All commands below are run from the `positional_shells` directory inside the checkout, because the config files reference data and output locations with relative paths (for example `../data/train_data`). Running from anywhere else will break those paths. Change into it first:

```bash
cd positional_shells
```

**Python dependencies.** Verify the core packages import before starting:

```bash
python3 -c "import torch, numpy, pandas, h5py, matplotlib; print('deps ok'); print('torch', torch.__version__)"
```

If that prints a torch version, the environment is ready. If it raises an import error, activate the environment or re-run `setup_env.sh` before continuing.

**Data.** This repo ships with a bundled sample dataset under `data/`, pulled in automatically by Git LFS when you cloned. Confirm the two manifests are present and readable before starting:

```bash
ls ../data/train_data/file_manifest.csv ../data/val_data/validation_manifest.csv
```

Both paths should exist. If either is missing, the LFS files did not download — run `git lfs pull` from the repo root, then check again.

(If you later want to run against the full dataset instead of the sample, replace `../data/train_data` and `../data/val_data` with the full directories or symlinks to them. Nothing else changes.)

**Config files.** Three TOML files under `../configs` control the run: `shell.toml`, `cnp.toml`, and `mfgp.toml`. The defaults in this repo are a valid starting point and already point at the bundled `data/` directory. You do not need to change anything to get a run.

## Step 1: Prepare the shell data

This reads the raw training events, assigns every event to a shell, splits the high-fidelity events into training and validation according to `validation_fraction` in `shell.toml`, and writes the labeled HDF5 blocks the CNP will read.

```bash
python3 -c "
import shells
from pathlib import Path
config = shells.build_config(Path('../configs/shell.toml').resolve())
result = shells.prepare_shell_cnp_data(config)
print('PREPARE DONE:', result)
"
```

**Expected result.** The command prints progress as it labels events and writes blocks, then a `PreparationResult` naming the output directory and the training and validation HDF5 roots. It creates `../outputs/shell/` containing `shell_table_by_theta.csv`, `event_class_manifest.csv`, and `training/` and `validation/` subdirectories of HDF5 blocks. If `../outputs/shell/training` and `../outputs/shell/validation` are not created, do not continue.

## Step 2: Train the CNP

This trains the event classifier on the prepared blocks. It uses the GPU automatically if one is available. Training length is set by `epochs` and `steps_per_epoch` in `cnp.toml`.

```bash
python3 -c "
import shells
from pathlib import Path
result = shells.run_shell_cnp_training(
    Path('../configs/shell.toml').resolve(),
    Path('../configs/cnp.toml').resolve(),
)
print('CNP TRAIN DONE:', result.model_path)
"
```

**Expected result.** The command prints per-epoch training progress and finishes by writing a trained model to `../outputs/cnp/`, named from the `version` field in `cnp.toml` (for example `cnp_shell_v1_model.pth`), along with a training history CSV and training-curve plots. Record the printed model path. If no `.pth` model file is written, do not continue.

## Step 3: Run CNP predictions

This runs the trained CNP over both the training split and the held-out validation split, aggregates the per-event outputs into per-geometry shell predictions, and writes one CSV for each split. Both are needed: the training predictions feed MF-GP training, and the validation predictions feed MF-GP validation.

```bash
python3 -c "
import shells
from pathlib import Path
shell_cfg = Path('../configs/shell.toml').resolve()
cnp_cfg = Path('../configs/cnp.toml').resolve()
model = Path('../outputs/cnp/cnp_shell_v1_model.pth').resolve()

train_pred = shells.run_shell_cnp_prediction(shell_cfg, cnp_cfg, split='training', model_path=model)
print('TRAIN PRED:', train_pred.prediction_path)

val_pred = shells.run_shell_cnp_prediction(shell_cfg, cnp_cfg, split='validation', model_path=model)
print('VAL PRED:', val_pred.prediction_path)
"
```

**Expected result.** The command prints the number of predicted events and prediction groups for each split, then writes two CSVs to `../outputs/cnp/`, one for training and one for validation (for example `cnp_shell_v1_output.csv` and `cnp_shell_v1_output_validation.csv`). Record both paths. If either CSV is missing, do not continue.

## Step 4: Train the MF-GP

This trains the multi-fidelity Gaussian process on the CNP training predictions. The fiducial cutoff shell is set by `n_shell` in `mfgp.toml`.

```bash
python3 -c "
import shells
from pathlib import Path
result = shells.run_shell_mfgp_training(
    Path('../outputs/cnp/cnp_shell_v1_output.csv').resolve(),
    Path('../configs/mfgp.toml').resolve(),
)
print('MFGP TRAIN DONE:', result.model_path)
"
```

**Expected result.** The command prints the number of LF and HF points it trained on, the estimated fidelity scaling, and confirmation that the GP fit completed, then writes a trained model to `../outputs/mfgp/` (for example `mfgp_mfgp_v1_model.joblib`). Record the model path. If no model file is written, do not continue.

## Step 5: Validate the MF-GP

This evaluates the trained MF-GP on the held-out validation predictions and computes the validation metrics.

```bash
python3 -c "
import shells
from pathlib import Path
result = shells.run_shell_mfgp_prediction(
    Path('../outputs/cnp/cnp_shell_v1_output_validation.csv').resolve(),
    Path('../configs/mfgp.toml').resolve(),
    model_path=Path('../outputs/mfgp/mfgp_mfgp_v1_model.joblib').resolve(),
)
print('MFGP VALIDATE DONE')
print('METRICS:', result.metrics)
"
```

**Expected result.** The command prints a metrics dictionary containing at least `rmse`, `mae`, `r2`, and the one, two, and three sigma coverage fractions. A successful run has an `r2` close to 1 and coverage fractions near their expected values. (On the bundled sample, with few validation points, coverage can look very clean — that is expected.)

## Step 6: Generate the plots

This produces the standard CNP and MF-GP visualizations from the completed pipeline outputs. It reads the saved prediction files and the trained MF-GP model, and writes the plots to an outputs directory. It depends on Steps 3 through 5 having completed, since it reads their outputs.

```bash
python3 generate_plots.py
```

**Expected result.** The command prints each plot as it writes it and ends with `PLOTS DONE`. It creates `../outputs/plots/` containing: the CNP predicted-vs-true shell occupancy plots for the training and validation splits (`cnp_shell_occupancy_training.png`, `cnp_shell_occupancy_validation.png`), the MF-GP validation points and parity plots (`mfgp_validation_points.png`, `mfgp_validation_parity.png`), and the MF-GP predicted mean and uncertainty surface (`mfgp_mean_uncertainty.png`). If `../outputs/plots/` is not created or `PLOTS DONE` is not printed, report the error.

## Notes on file names and versions

The output file names above (for example `cnp_shell_v1_output.csv`) come from the `version` fields in the config files. If those fields are changed, the file names change with them. Rather than assuming the names, read the path printed by each step and pass it to the next step. An agent running this should capture the printed output paths at each stage and use those exact paths downstream, not hard-coded names.

## If a step fails

Each stage checks its inputs and will raise an error describing what is missing, most often a data path that does not resolve or a config value that is out of range. When a step fails, stop and report the full error and the step it failed on. Do not run later steps, because each one depends on the outputs of the one before it.
