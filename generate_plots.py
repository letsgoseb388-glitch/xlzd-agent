"""
Generate the standard CNP and MF-GP visualizations from a completed pipeline run.

This runs after the pipeline has produced its outputs, and reads the saved
files directly, so it does not depend on having the live result objects in
memory. It writes the plots to an outputs/plots directory.

Plots produced:
  - CNP predicted vs true shell occupancy (training and validation splits)
  - MF-GP validation points (HF truth vs prediction with 1/2/3 sigma)
  - MF-GP validation parity (prediction vs truth)
  - MF-GP predicted mean and uncertainty surface over the geometry space

Run from the positional_shells directory, after the pipeline has completed.
"""

import argparse
import sys
from pathlib import Path

# mfgp.py and common/ live in the repo root, one level up from here.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt

import mfgp
from common import visualize


def main():
    parser = argparse.ArgumentParser(
        description="Generate CNP and MF-GP plots from saved pipeline outputs."
    )
    parser.add_argument("--cnp-train", default="../outputs/cnp/cnp_shell_v1_output.csv",
                        help="CNP training-split prediction CSV.")
    parser.add_argument("--cnp-val", default="../outputs/cnp/cnp_shell_v1_output_validation.csv",
                        help="CNP validation-split prediction CSV.")
    parser.add_argument("--mfgp-predictions", default="../outputs/mfgp/mfgp_mfgp_v1_predictions.csv",
                        help="MF-GP validation predictions CSV.")
    parser.add_argument("--mfgp-model", default="../outputs/mfgp/mfgp_mfgp_v1_model.joblib",
                        help="Saved MF-GP model.")
    parser.add_argument("--mfgp-train-pred", default="../outputs/mfgp/mfgp_mfgp_v1_training_predictions.csv",
                        help="MF-GP training predictions CSV (HF points, used for the surface plot).")
    parser.add_argument("--mfgp-config", default="../configs/mfgp.toml",
                        help="MF-GP config, used to evaluate the model over the grid.")
    parser.add_argument("--outdir", default="../outputs/plots",
                        help="Directory to write the plots into.")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {outdir}")

    # --- CNP shell occupancy (training + validation) ---
    cnp_train = Path(args.cnp_train).resolve()
    cnp_val = Path(args.cnp_val).resolve()
    if cnp_train.exists():
        fig, _ = visualize.plot_cnp_pred_shell_occupancy(cnp_train)
        fig.savefig(outdir / "cnp_shell_occupancy_training.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  wrote cnp_shell_occupancy_training.png")
    else:
        print(f"  skip CNP training plot, not found: {cnp_train}")

    if cnp_val.exists():
        fig, _ = visualize.plot_cnp_pred_shell_occupancy(cnp_val)
        fig.savefig(outdir / "cnp_shell_occupancy_validation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  wrote cnp_shell_occupancy_validation.png")
    else:
        print(f"  skip CNP validation plot, not found: {cnp_val}")

    # --- MF-GP validation points + parity (from saved predictions CSV) ---
    mfgp_pred = Path(args.mfgp_predictions).resolve()
    if mfgp_pred.exists():
        df = pd.read_csv(mfgp_pred)
        visualize.plot_mfgp_validation_points(df, outpath=outdir / "mfgp_validation_points.png")
        print("  wrote mfgp_validation_points.png")
        visualize.plot_mfgp_validation_parity(df, outpath=outdir / "mfgp_validation_parity.png")
        print("  wrote mfgp_validation_parity.png")
    else:
        print(f"  skip MF-GP validation plots, not found: {mfgp_pred}")

    # --- MF-GP mean/uncertainty surface (reload model, evaluate over a grid) ---
    model_path = Path(args.mfgp_model).resolve()
    train_pred_path = Path(args.mfgp_train_pred).resolve()
    mfgp_config = Path(args.mfgp_config).resolve()
    if model_path.exists() and train_pred_path.exists():
        hf_train = pd.read_csv(train_pred_path)
        r_max = hf_train["detector_R"].max() + 100
        z_max = hf_train["detector_Z"].max() + 100
        r = np.linspace(0, r_max, 120)
        z = np.linspace(0, z_max, 120)
        R, Z = np.meshgrid(r, z)
        grid = pd.DataFrame({"detector_R": R.ravel(), "detector_Z": Z.ravel()})

        grid_result = mfgp.run_mfgp_prediction(
            mfgp_config,
            model_path=model_path,
            x=grid,
            output_path=outdir / "mfgp_grid_predictions.csv",
        )
        fig, _ = visualize.plot_mfgp_mean_uncertainty(
            grid_result,
            hf_points=hf_train,
            outpath=outdir / "mfgp_mean_uncertainty.png",
        )
        plt.close(fig)
        print("  wrote mfgp_mean_uncertainty.png")
    else:
        print(f"  skip MF-GP surface plot, missing model or training predictions")

    print("PLOTS DONE")


if __name__ == "__main__":
    main()
