# Running the Shell RESuM Pipeline

This document describes how to run the full shell-based RESuM pipeline from a clean checkout to the final MF-GP validation metrics and plots. It is written to be followed step by step, either by a person or by a coding agent. Each step gives the command to run, what it needs beforehand, and what it should produce. **If a step does not produce the described output, stop and report the error rather than continuing** — each stage depends on the outputs of the one before it.

## What this pipeline does

The pipeline turns raw detector simulation data into a trained surrogate model that predicts a detector's fiducial containment as a function of its geometry. It runs in these stages:

1. Prepare the raw training events (`data/train_data`) into shell-labeled HDF5 blocks. Every geometry, LF and HF, goes into training — there is no internal holdout.
1b. Prepare the **independent validation set** (`data/val_data`, 43 HF geometries) into its own shell-labeled HDF5 blocks under a separate output directory.
2. Train the Conditional Neural Process (CNP) on the Step 1 training blocks.
3. Run the trained CNP over the training blocks and the independent validation blocks to produce per-geometry shell predictions.
4. Train the Multi-Fidelity Gaussian Process (MF-GP) on the CNP training predictions.
5. Evaluate the MF-GP on the independent validation predictions and report metrics.
6. Generate the CNP and MF-GP visualizations from the outputs.

**Validation means the independent `data/val_data` set.** Those geometries never enter CNP or MF-GP training: Step 1b writes them only to `outputs/shell_independent/validation/`, CNP training reads only `outputs/shell/training/`, and Step 1b refuses to run if its output directory overlaps `outputs/shell/` or if any val_data geometry also appears in the training manifest.

Steps run in order; each depends on the outputs of the ones before it. (Step 1b only needs the prerequisites, but must finish before Step 3.)

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

This reads the raw training events from `../data/train_data`, assigns every event to a shell, and writes the labeled HDF5 blocks the CNP will read. `validation_fraction = 0.0` in `shell.toml` disables the internal holdout, so every HF event from `train_data` is used for training. (A non-zero fraction still holds out that share of HF events into `validation/`, but this pipeline validates on the independent set instead.)

This step clears and rebuilds `../outputs/shell/` on every run.

```bash
python3 -c "
import shells
from pathlib import Path
config = shells.build_config(Path('../configs/shell.toml').resolve())
result = shells.prepare_shell_cnp_data(config)
print('PREPARE DONE:', result)
"
```

**Expected result.** The command prints progress as it labels events and writes blocks, then a `PreparationResult`. The summary must show `HF validation events (fidelity=1): 0` and a non-zero `HF training events` count, and the split log prints `HF validation (fidelity=1): 0 (internal holdout disabled)`. It creates `../outputs/shell/` containing `shell_table_by_theta.csv`, `event_class_manifest.csv`, and `training/lf/` and `training/hf/` with HDF5 blocks. Check:

```bash
ls ../outputs/shell/training/lf/*.h5 ../outputs/shell/training/hf/*.h5 | head
find ../outputs/shell/validation -name '*.h5' 2>/dev/null | wc -l   # must print 0
```

If there are no training `.h5` files, or any `.h5` files exist under `../outputs/shell/validation`, do not continue.

## Step 1b: Prepare the independent validation set

This prepares `../data/val_data` (manifest `validation_manifest.csv`, 43 HF geometries) using the same shell settings as Step 1. It reuses `prepare_shell_cnp_data`; only the input and output directories change. All 43 geometries go into one validation pool with no train/val split. Output goes to `../outputs/shell_independent/`, which is separate from `../outputs/shell/`. Before writing anything, it checks that:

- the output directory does not overlap `../outputs/shell/`,
- no val_data filename or (R, Z) geometry appears in `train_data/file_manifest.csv`,
- every val_data row is fidelity 1.

If any check fails, it raises an error.

```bash
stat -c '%y' ../outputs/shell   # record this before running
python3 -c "
import shells
from pathlib import Path
result = shells.prepare_independent_validation_data(
    Path('../configs/shell.toml').resolve(),
    input_dir=Path('../data/val_data').resolve(),
    output_dir=Path('../outputs/shell_independent').resolve(),
    manifest_name='validation_manifest.csv',
)
print('INDEPENDENT PREPARE DONE:', result)
"
```

**Expected result.** It prints `[independent] 43 geometries, none shared with the 55 training geometries`, then `Total files loaded: 43`, `LF training events (fidelity=0): 0`, `HF training events (fidelity=1): 0` and a non-zero `HF validation events`. It also prints `INDEPENDENT PREPARE DONE`. Check:

```bash
ls ../outputs/shell_independent/validation/hf/*.h5 | head      # must list files
ls ../outputs/shell_independent/shell_table_by_theta.csv ../outputs/shell_independent/event_class_manifest.csv
test ! -e ../outputs/shell_independent/training && echo "no training dir: ok"
stat -c '%y' ../outputs/shell   # must match the value recorded above
```

If there are no validation `.h5` files, if a `training/` directory exists, or if `../outputs/shell` was modified, do not continue.

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

This runs the trained CNP over the training blocks (`../outputs/shell/training`) and the independent validation blocks (`../outputs/shell_independent/validation`). It aggregates the per-event outputs into per-geometry shell predictions and writes one CSV for each. Both are needed: the training predictions feed MF-GP training, and the independent predictions feed MF-GP validation.

```bash
python3 -c "
import shells
from pathlib import Path
shell_cfg = Path('../configs/shell.toml').resolve()
cnp_cfg = Path('../configs/cnp.toml').resolve()
model = Path('../outputs/cnp/cnp_shell_v1_model.pth').resolve()

train_pred = shells.run_shell_cnp_prediction(shell_cfg, cnp_cfg, split='training', model_path=model)
print('TRAIN PRED:', train_pred.prediction_path)

val_pred = shells.run_shell_cnp_prediction(
    shell_cfg, cnp_cfg, split='independent', model_path=model,
    prepared_dir=Path('../outputs/shell_independent').resolve(),
)
print('INDEPENDENT PRED:', val_pred.prediction_path, 'groups:', val_pred.n_groups)
"
```

**Expected result.** The command prints the number of predicted events and prediction groups for each, then writes two CSVs to `../outputs/cnp/`: `cnp_shell_v1_output.csv` (training) and `cnp_shell_v1_output_independent.csv` (independent validation). The independent run must print `groups: 43`, one per val_data geometry. Check:

```bash
ls ../outputs/cnp/cnp_shell_v1_output.csv ../outputs/cnp/cnp_shell_v1_output_independent.csv
```

Record both paths. If either CSV is missing, or the independent group count is not 43, do not continue.

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

This evaluates the trained MF-GP on the independent validation set (the Step 3 CNP output for `data/val_data`) and computes the validation metrics.

```bash
python3 -c "
import shells
from pathlib import Path
result = shells.run_shell_mfgp_prediction(
    Path('../outputs/cnp/cnp_shell_v1_output_independent.csv').resolve(),
    Path('../configs/mfgp.toml').resolve(),
    model_path=Path('../outputs/mfgp/mfgp_mfgp_v1_model.joblib').resolve(),
)
print('MFGP VALIDATE DONE')
print('METRICS:', result.metrics)
"
```

**Expected result.** The command prints `MFGP VALIDATE DONE` and a metrics dictionary containing at least `rmse`, `mae`, `r2`, and the one, two, and three sigma coverage fractions, computed over the 43 independent geometries. A successful run has an `r2` close to 1 and coverage fractions near their expected values. With `save_predictions = true` in `mfgp.toml`, the validation predictions CSV (for example `../outputs/mfgp/mfgp_mfgp_v1_predictions.csv`) and its metrics file (`mfgp_mfgp_v1_predictions_metrics.json`) are written. Check that both exist and the CSV has 43 rows:

```bash
ls ../outputs/mfgp/mfgp_mfgp_v1_predictions_metrics.json
python3 -c "import pandas as pd; df = pd.read_csv('../outputs/mfgp/mfgp_mfgp_v1_predictions.csv'); print('rows:', len(df))"
```

If no metrics are printed or the row count is not 43, report the error.

## Step 6: Generate the plots

This produces the standard CNP and MF-GP visualizations from the completed pipeline outputs. It reads the saved prediction files and the trained MF-GP model, and writes the plots to an outputs directory. It depends on Steps 3 through 5 having completed, since it reads their outputs. The "validation" CNP plot is now drawn from the independent set (`cnp_shell_v1_output_independent.csv`).

```bash
python3 generate_plots.py
```

**Expected result.** The command prints each plot as it writes it and ends with `PLOTS DONE`. It creates `../outputs/plots/` containing: the CNP predicted-vs-true shell occupancy plots for the training split and the independent validation set (`cnp_shell_occupancy_training.png`, `cnp_shell_occupancy_validation.png`), the MF-GP validation points and parity plots (`mfgp_validation_points.png`, `mfgp_validation_parity.png`), and the MF-GP predicted mean and uncertainty surface (`mfgp_mean_uncertainty.png`). If `../outputs/plots/` is not created or `PLOTS DONE` is not printed, report the error.

## Notes on file names and versions

The output file names above (for example `cnp_shell_v1_output.csv`) come from the `version` fields in the config files. If those fields are changed, the file names change with them. Rather than assuming the names, read the path printed by each step and pass it to the next step. An agent running this should capture the printed output paths at each stage and use those exact paths downstream, not hard-coded names.

## If a step fails

Each stage checks its inputs and will raise an error describing what is missing, most often a data path that does not resolve or a config value that is out of range. When a step fails, stop and report the full error and the step it failed on. Do not run later steps, because each one depends on the outputs of the one before it.
