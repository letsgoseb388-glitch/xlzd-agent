# For a coding agent

If you are a coding agent (for example Claude Code) opening this repo, your task is to reproduce the XLZD RESuM results by running the pipeline end to end.

Do this:

1. Set up the environment: run `bash setup_env.sh` from the repo root, then activate it with `source .venv/bin/activate`.
2. Confirm the dependencies import and the data is present (the "Prerequisites" section of `RUNBOOK.md`).
3. Read `RUNBOOK.md` and run **Steps 1 through 6 in order, from the `positional_shells/` directory.**
4. After each step, **check its expected output before continuing.** Each step in the runbook states a specific file that must exist or a specific line that must print. Do not move to the next step until the current step's check passes.
5. If a step fails its check, **stop and report** the full error and which step failed. Do not run later steps — each depends on the output of the one before it.

When you finish, the trained models, validation metrics, and the five plots are under `outputs/`. Report the validation metrics from Step 5 and confirm the five plot files from Step 6 exist.

Do not modify the model code or the configs to make a step pass. If something genuinely cannot run (for example no GPU and a CUDA-only path), report it rather than working around it.
