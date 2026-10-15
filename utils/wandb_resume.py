"""Helpers for persisting wandb run ids across resumed REPA training runs."""

from __future__ import annotations

from pathlib import Path

import wandb


def get_wandb_init_kwargs(
    save_dir: str,
    run_name: str,
    should_resume: bool,
    force_new_run: bool = False,
):
    run_id_path = Path(save_dir) / "wandb_run_id.txt"

    # Added for wandb resume: persist a stable run id in the experiment
    # directory so checkpoint resumes can keep logging into the same wandb run.
    if should_resume and (not force_new_run) and run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        if run_id:
            return {"wandb": {"name": run_name, "id": run_id, "resume": "allow"}}

    run_id = wandb.util.generate_id()
    run_id_path.write_text(run_id, encoding="utf-8")
    return {"wandb": {"name": run_name, "id": run_id}}
