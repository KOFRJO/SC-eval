import concurrent.futures
import functools
from itertools import product
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from fd_shifts import logger
from fd_shifts.configs import Config
from fd_shifts.experiments.configs import get_experiment_config, list_experiment_configs
from fd_shifts.experiments.tracker import list_bootstrap_analysis_output_files
from fd_shifts.reporting import (
    _load_file,
    _filter_unused,
    assign_hparams_from_names,
    DATASETS,
    filter_best_lr,
    filter_best_hparams,
    rename_confids,
    rename_studies,
)
from fd_shifts.reporting import tables
from fd_shifts.reporting.plots_bootstrap import (
    bs_blob_plot,
    bs_box_scatter_plot,
    bs_podium_plot,
    bs_significance_map,
    bs_significance_map_colored,
    bs_kendall_tau_violin,
    bs_kendall_tau_comparing_metrics,
)


def _load_bootstrap_experiment(
    name: str,
    stratified_bs: bool = False,
    filter_study_name: list = None,
    filter_dataset: list = None,
    original_new_class_mode: bool = False,
) -> pd.DataFrame | None:
    from fd_shifts.main import omegaconf_resolve

    config = get_experiment_config(name)
    config = omegaconf_resolve(config)
    
    if filter_dataset is not None and config.data.dataset not in filter_dataset:
        return

    data = list(
        map(
            functools.partial(_load_file, config, name),
            list_bootstrap_analysis_output_files(
                config, stratified_bs, filter_study_name, original_new_class_mode
            ),
        )
    )

    if len(data) == 0 or any(map(lambda d: d is None, data)):
        return
    
    data = pd.concat(data)  # type: ignore
    data = (
        data.assign(
            experiment=config.data.dataset + ("vit" if "vit" in name else ""),
            run=int(name.split("run")[1].split("_")[0]),
            dropout=config.model.dropout_rate,
            rew=config.model.dg_reward if config.model.dg_reward is not None else 0,
            lr=config.trainer.optimizer.init_args["init_args"]["lr"],
        )
        .dropna(subset=["name", "model"])
        .drop_duplicates(
            subset=["name", "study", "model", "network", "confid", "bootstrap_index"]
        )
    )
    
    return data


def load_all(
    stratified_bs: bool = False,
    filter_study_name: list = None,
    filter_dataset: list = None,
    original_new_class_mode: bool = False,
    include_vit: bool = False,
):
    dataframes = []
    # TODO: make this async
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        dataframes = list(
            filter(
                lambda d: d is not None,
                executor.map(
                    functools.partial(
                        _load_bootstrap_experiment,
                        stratified_bs=stratified_bs,
                        filter_study_name=filter_study_name,
                        filter_dataset=filter_dataset,
                        original_new_class_mode=original_new_class_mode,
                    ),
                    filter(
                        (
                            (lambda exp: ("clip" not in exp)) if include_vit
                            else lambda exp: (
                                ("clip" not in exp) and (not exp.startswith("vit"))
                            )
                        ),
                        list_experiment_configs()
                    ),
                ),
            )
        )

    data = pd.concat(dataframes)  # type: ignore
    data = data.loc[~data["study"].str.contains("tinyimagenet_original")]
    data = data.loc[~data["study"].str.contains("tinyimagenet_proposed")]

    # data = data.query(
    #     'not (experiment in ["cifar10", "cifar100", "super_cifar100"]'
    #     'and not name.str.contains("vgg13"))'
    # )

    data = data.query(
        'not ((experiment.str.contains("super_cifar100")'
        ")"
        'and not (study == "iid_study"))'
    )

    data = data.query(
        'not (experiment.str.contains("openset")' 'and study.str.contains("iid_study"))'
    )

    data = data.assign(study=data.experiment + "_" + data.study)

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100",
            "cifar100_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100", "cifar100"
        ),
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100vit",
            "cifar100vit_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100vit", "cifar100vit"
        ),
    )

    data = data.assign(ece=data.ece.mask(data.ece < 0))

    return data


def create_plots_per_study(
    study: str,
    dset: str,
    metrics: list,
    out_dir: Path,
    stratified_bs: bool = False,
    original_new_class_mode: bool = False,
    metric_hparam_search: str = None,
):
    logger.info(f"Reporting bootstrap results for dataset '{dset}', study '{study}'")
    
    data_raw = load_all(
        stratified_bs=stratified_bs,
        filter_study_name=[study],
        filter_dataset=[dset],
        original_new_class_mode=original_new_class_mode,
    )

    data_raw = assign_hparams_from_names(data_raw)
    
    for metric in metrics:
        metric_to_optimize = (
            metric if metric_hparam_search is None else metric_hparam_search
        )
        data, selection_df = filter_best_lr(data_raw, metric=metric_to_optimize)
        selection_df.to_csv(
            out_dir / f"filter_best_lr_{dset}_{metric_to_optimize}.csv", decimal="."
        )
        data, selection_df = filter_best_hparams(
            data, bootstrap_analysis=True, metric=metric_to_optimize
        )
        selection_df.to_csv(
            out_dir / f"filter_best_hparams_{dset}_{metric_to_optimize}.csv", decimal="."
        )
        data = _filter_unused(data)
        
        # Filter MCD data
        # data = data[~data.confid.str.contains("mcd")]
        
        data = rename_confids(data)
        data = rename_studies(data)
        
        data = data[data.confid.isin(CONFIDS_TO_REPORT)]
        
        logger.info("Removing 'val_tuning' studies and aggregating noise studies")
        data = data[~data['study'].str.contains("val_tuning")]
        
        if study == "noise_study":
            data = data.groupby(
                ["study", "confid", "run", "bootstrap_index"]
            ).mean().reset_index()
        
        # First, do all plots without aggregation across runs, then aggregate
        # for aggregate_runs in (False, True):
        for aggregate_runs in (True,):
            if aggregate_runs:
                data, _ = tables.aggregate_over_runs(data, metric_columns=metrics)
                group_columns = ["bootstrap_index"]
                blob_dir = out_dir / "blob_run_avg"
                podium_dir = out_dir / "podium_run_avg"
                box_dir = out_dir / "box_run_avg"
                significance_map_dir = out_dir / "significance_map_run_avg"
                kendall_violin_dir = out_dir / "kendall_violin_run_avg"
            else:
                data = data[["confid", "study", "run", "bootstrap_index"] + metrics]
                group_columns = ["bootstrap_index", "run"]
                blob_dir = out_dir / "blob"
                podium_dir = out_dir / "podium"
                box_dir = out_dir / "box"
                significance_map_dir = out_dir / "significance_map"
                kendall_violin_dir = out_dir / "kendall_violin"
            
            blob_dir.mkdir(exist_ok=True)
            podium_dir.mkdir(exist_ok=True)
            box_dir.mkdir(exist_ok=True)
            significance_map_dir.mkdir(exist_ok=True)
            kendall_violin_dir.mkdir(exist_ok=True)
        
            # Compute method ranking per bootstrap sample (and run if aggregate_runs=False)
            data["rank"] = data.groupby(group_columns)[metric].rank(method="min")
            
            # Compute ranking histogram per method
            histograms = data.groupby("confid")["rank"].value_counts().unstack(fill_value=0)
            
            # Sort methods by mean rank
            histograms["mean_rank"] = (histograms.columns * histograms).sum(axis=1) / histograms.sum(axis=1)
            histograms["median_rank"] = data.groupby("confid")["rank"].median().astype(int)
            histograms = histograms.sort_values(by=["mean_rank", "median_rank"])

            medians = histograms.median_rank
            histograms = histograms.drop(columns="mean_rank")
            histograms = histograms.drop(columns="median_rank")

            filename = f"blob_plot_{dset}_{study}_{metric}.pdf"
            bs_blob_plot(
                histograms=histograms,
                medians=medians,
                out_dir=blob_dir,
                filename=filename,
            )
            
            filename = f"podium_plot_{dset}_{study}_{metric}.pdf"
            bs_podium_plot(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=podium_dir,
                filename=filename,
            )
            
            filename = f"box_plot_{dset}_{study}_{metric}.pdf"
            bs_box_scatter_plot(
                data=data,
                metric=metric,
                out_dir=box_dir,
                filename=filename,
            )
            
            filename = f"significance_map_{dset}_{study}_{metric}.pdf"
            bs_significance_map(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
            )
            
            filename = f"colored_significance_map_{dset}_{study}_{metric}.pdf"
            bs_significance_map_colored(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
            )
            
            if "aurc" in metrics and "augrc" in metrics:
                filename = f"kendall_violin_{dset}_{study}_{metric}.pdf"
                bs_kendall_tau_violin(
                    data=data,
                    metric=metric,
                    histograms=histograms,
                    out_dir=kendall_violin_dir,
                    filename=filename,
                )


CONFIDS_TO_REPORT = [
    "MSR",
    "MLS",
    "PE",
    "MCD-MSR",
    "MCD-PE",
    "MCD-EE",
    "DG-MCD-MSR",
    "ConfidNet",
    "DG-Res",
    "Devries et al.",
    "TEMP-MLS",
    "DG-PE",
    "DG-TEMP-MLS",
]


def report_bootstrap_results(
    out_path: str | Path = "./output/bootstrap",
    stratified_bs: bool = False,
    metric_hparam_search: str = None,
):
    """"""
    if stratified_bs:
        out_path = "./output/bootstrap-stratified"
    
    if metric_hparam_search is not None:
        out_path = str(out_path) + f"-optimized-{metric_hparam_search}"
    
    data_dir: Path = Path(out_path).expanduser().resolve()
    data_dir.mkdir(exist_ok=True, parents=True)
    
    datasets = [d for d in DATASETS if d != "super_cifar100"]
    studies = ["iid_study"]
    # studies = ["iid_study", "noise_study", "new_class_study"]
    metrics = ["aurc", "augrc"]

    logger.info(
        f"Reporting bootstrap results for datasets '{datasets}', studies '{studies}'"
    )
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            # Submit tasks to the executor
            future_to_arg = {
                executor.submit(
                    create_plots_per_study,
                    study=study,
                    dset=dset,
                    metrics=metrics,
                    out_dir=data_dir,
                    stratified_bs=stratified_bs,
                    original_new_class_mode=False,
                    metric_hparam_search=metric_hparam_search,
                ): dict(study=study, dset=dset)
                for dset, study in product(datasets, studies)
            }
            try:
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_arg),
                    total=len(future_to_arg),
                ):
                    arg = future_to_arg[future]
                    # Get the result from the future (this will raise an exception if the
                    # function call raised an exception)
                    future.result()
            except Exception as exc:
                # Handle the exception
                print(f"Function call with argument {arg} raised an exception: {exc}")
                # Raise an error or take appropriate action
                raise RuntimeError("One or more executor failed") from exc
            finally:
                # Ensure executor and associated processes are properly terminated
                executor.shutdown()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info(
            "Executor shut down. Kill running futures using\n"
            "'ps -ef | grep 'main.py report_bootstrap' | grep -v grep | awk '{print $2}' | "
            "xargs -r kill -9'"
        )
        raise
