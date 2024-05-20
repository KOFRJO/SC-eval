import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
import multiprocessing
import subprocess
from tqdm import tqdm

import rich
from rich.syntax import Syntax
import pandas as pd

from fd_shifts.main import omegaconf_resolve
from fd_shifts import logger
from fd_shifts.experiments.configs import list_experiment_configs, get_experiment_config

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "./logs_bootstrap/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
python fd_shifts/main.py analysis_bootstrap \
    --experiment={experiment} \
    --n_bs={n_bs} \
    --exclude_noise_study={exclude_noise_study} \
    --no_iid={no_iid} \
    --iid_only={iid_only} {overrides}
"""


async def worker(name, queue: asyncio.Queue[str]):
    while True:
        # Get a "work item" out of the queue.
        cmd = await queue.get()
        logger.info(f"{name} running {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
        )

        # Wait for the subprocess exit.
        await proc.wait()

        if proc.returncode != 0:
            logger.error(f"{name} running {cmd} finished abnormally")
        else:
            logger.info(f"{name} running {cmd} finished")

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


def run_command(command):
    subprocess.run(command, shell=True)


async def run(
    _experiments: list[str],
    dry_run: bool,
    iid_only: bool = False,
    no_iid: bool = False,
    exclude_noise_study: bool = False,
    n_bs: int = 500,
    num_workers: int = 12,
):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    Path("./logs").mkdir(exist_ok=True)

    queue = []

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{experiment.replace('/', '_').replace('.','_')}"

        overrides = {}

        cmd = BASH_BASE_COMMAND.format(
            experiment=experiment,
            n_bs=n_bs,
            iid_only=iid_only,
            no_iid=no_iid,
            exclude_noise_study=exclude_noise_study,
            overrides=" ".join(f"--config.{k}={v}" for k, v in overrides.items()),
        ).strip()

        cmd = BASH_LOCAL_COMMAND.format(
            command=cmd, log_file_name=log_file_name
        ).strip()
        if not dry_run:
            rich.print(Syntax(cmd, "bash", word_wrap=True, background_color="default"))
            queue.append(cmd)

    if queue == []:
        return
    # Create a tqdm progress bar
    with tqdm(total=len(queue), desc="Experiments") as pbar:
        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_workers)
        # Map the list of commands to the worker pool
        for _ in pool.imap_unordered(run_command, queue):
            pbar.update()
        # Close the pool to prevent any more tasks from being submitted
        pool.close()
        # Wait for all processes to finish
        pool.join()


def filter_experiments(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    exclude_backbone: str | None,
    run_nr: int | None,
    rew: float | None,
    name: str | None,
) -> filter:
    _experiments = list_experiment_configs()
    
    # No CLIP
    _experiments = filter(
        lambda e: get_experiment_config(e).exp.group_name != "clip",
        _experiments,
    )
    
    # No ViT
    _experiments = filter(
        lambda e: get_experiment_config(e).exp.group_name != "vit",
        _experiments,
    )

    if dataset is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).data.dataset == dataset,
            _experiments,
        )

    if dropout is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.dropout_rate == dropout,
            _experiments,
        )
    if rew is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.dg_reward == rew,
            _experiments,
        )
    if run_nr is not None:
        _experiments = filter(
            lambda e: f"_run{run_nr}_" in e,
            _experiments,
        )

    if model is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.name == model + "_model",
            _experiments,
        )

    if backbone is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.network.name == backbone,
            _experiments,
        )

    if exclude_model is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.name != exclude_model + "_model",
            _experiments,
        )
    
    if exclude_backbone is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.network.name != exclude_backbone,
            _experiments,
        )

    if name is not None:
        _experiments = filter(lambda e: e == name, _experiments)

    return _experiments


_FILTERS = {}

def register_filter(name):
    def _inner_wrapper(func):
        _FILTERS[name] = func
        return func
    return _inner_wrapper


def launch(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    exclude_backbone: str | None,
    dry_run: bool,
    run_nr: int | None,
    rew: float | None,
    cluster: bool,
    name: str | None,
    iid_only: bool,
    no_iid: bool,
    exclude_noise_study: bool,
    n_bs: int,
    num_workers: int,
    custom_filter: str | None,
):
    _experiments = list(
        filter_experiments(
            dataset,
            dropout,
            model,
            backbone,
            exclude_model,
            exclude_backbone,
            run_nr,
            rew,
            name,
        )
    )

    if custom_filter is not None:
        print(f"Applying custom filter {custom_filter}...")
        _experiments = _FILTERS[custom_filter](_experiments)
    
    _experiments = list(_experiments)
    
    print(f"Launching {len(_experiments)} experiments:")
    for exp in _experiments:
        rich.print(exp)

    if cluster:
        raise NotImplementedError()
    else:
        asyncio.run(
            run(
                _experiments,
                dry_run,
                iid_only,
                no_iid,
                exclude_noise_study,
                n_bs,
                num_workers,
            )
        )


def add_filter_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--dropout", default=None, type=int, choices=(0, 1))
    parser.add_argument(
        "--model", default=None, type=str, choices=("vit", "dg", "devries", "confidnet")
    )
    parser.add_argument("--backbone", default=None, type=str, choices=("vit",))
    parser.add_argument("--exclude-backbone", default=None, type=str)
    parser.add_argument(
        "--exclude-model",
        default=None,
        type=str,
        choices=("vit", "dg", "devries", "confidnet"),
    )

    parser.add_argument("--run", default=None, type=int)
    parser.add_argument("--reward", default=None, type=float)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--custom-filter", default=None, type=str)
    return parser


def add_arguments(parser: argparse.ArgumentParser):
    add_filter_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--iid-only", action="store_true")
    parser.add_argument("--no_iid", action="store_true")
    parser.add_argument("--exclude-noise-study", action="store_true")
    parser.add_argument("--n-bs", default=500, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    
    launch(
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        exclude_backbone=args.exclude_backbone,
        dry_run=args.dry_run,
        run_nr=args.run,
        rew=args.reward,
        cluster=args.cluster,
        name=args.name,
        iid_only=args.iid_only,
        no_iid=args.no_iid,
        exclude_noise_study=args.exclude_noise_study,
        n_bs=args.n_bs,
        num_workers=args.num_workers,
        custom_filter=args.custom_filter,
    )
