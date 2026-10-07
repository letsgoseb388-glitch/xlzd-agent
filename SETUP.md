# Environment setup details

`setup_env.sh` handles the common cases automatically. This file covers what it does and the edge cases.

## What `setup_env.sh` does

1. Creates a Python **3.12** virtual environment in `.venv` using `uv`.
2. Detects whether an NVIDIA GPU is present (via `nvidia-smi`) and installs the matching `torch` build.
3. Installs the remaining dependencies from `requirements.txt`.
4. Prints a verification line showing the torch version and whether CUDA is available.

Activate it afterward with `source .venv/bin/activate`.

## Why Python 3.12 specifically

The CUDA `torch` wheels are only published for specific Python versions. At the time this repo was built, Python 3.14 was too new — there was no matching CUDA wheel — so the setup pins 3.12, which has wheels for both the CPU and the CUDA builds.

## GPU: recent cards (Blackwell / RTX 50-series)

If you have an RTX 50-series card (or another Blackwell-architecture GPU), the default `torch` builds will not work out of the box. The symptom is subtle and worth knowing: `torch.cuda.is_available()` returns `True`, but any actual GPU computation fails with

```
CUDA error: no kernel image is available for execution on the device
```

This happens because older torch builds only ship compiled kernels for older GPU architectures, and the newest cards aren't covered. The fix, which `setup_env.sh` applies automatically when it sees a GPU, is to install a newer torch from the CUDA 12.8 index:

```bash
uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
```

You can confirm the GPU actually works (not just that CUDA "is available") with:

```bash
python3 -c "import torch; x = torch.randn(1000, 1000, device='cuda'); print((x @ x).sum().item())"
```

If that prints a number, the GPU is doing real work. If it errors, the kernel/architecture mismatch is still there.

## No GPU

If no GPU is present, `setup_env.sh` installs the CPU build of torch and the pipeline runs entirely on CPU. The only stage that is meaningfully slower is CNP training (Step 2 of the runbook); everything else is quick. The results are identical — only the training time differs.

## If `uv` is not installed

Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then re-run `bash setup_env.sh`.
