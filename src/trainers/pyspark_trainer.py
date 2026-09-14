"""
pyspark_trainer
===============

PySpark toolkit for the Avazu CTR MLP: session setup and grid search.

Provides two public entry points:

- ``start_spark_session``: builds a SparkSession tuned for a single-machine
  setup, with explicit control over driver memory, CPU cores, shuffle
  partitions and Adaptive Query Execution.
- ``run_grid_search_spark``: runs a manual grid search over the MLP
  hyperparameters with stratified K-fold cross-validation and JSON
  checkpoints after every combination.

Leakage guarantees
------------------
To match the behavior of the scikit-learn pipeline exactly, every
state-dependent transformation is fit on the training partition of the
fold being processed:

- Dominant categories and top-K lists are computed inside the fold loop,
  on the training partition only.
- All encoders (``StringIndexer``, ``OneHotEncoder``, ``TargetEncoder``)
  and the ``StandardScaler`` are part of the pipeline that is fit on the
  training partition of each fold.

No statistic computed on the validation fold is ever used to fit any
transformation.

Dependencies
------------
pyspark, numpy, pandas, tqdm
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    TargetEncoder,
    VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "1.2.0"

__author__ = (
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

_UNINORTE_AFFILIATION = {
    "name": "Universidad del Norte",
    "department": "Department of Mathematics, Physics and Data Science",
    "city": "Barranquilla",
    "country": "Colombia",
}

__authors_detail__ = [
    {
        "name": "Juan Camilo Mendoza Arango",
        "email": "cjarango@uninorte.edu.co",
        "orcid": "0000-0001-6872-8311",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "cjarango",
    },
    {
        "name": "Miguel Ángel Pérez Vargas",
        "email": "vargasmiguel@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "miguelpvmr",
    },
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding strategy
# ---------------------------------------------------------------------------
DICHOTOMIZE = ["C1", "C15", "C16", "device_type", "device_conn_type"]

TOP_K = {
    "banner_pos": 2,
    "C18": 3,
    "app_category": 3,
    "site_category": 3,
    "app_domain": 6,
    "C19": 10,
    "C20": 10,
    "C21": 10,
}

TE_HYBRID = ["app_id", "site_domain", "site_id"]
TE_PURE = ["C14", "C17", "device_model"]

NUMERIC_SCALED = ["log_freq_device_id", "log_freq_device_ip"]
NUMERIC_PASSTHROUGH = ["day", "hour_sin", "hour_cos"]

TIME_BAND = "time_band"
TARGET = "click"

TIME_BAND_LEVELS = 4

__all__ = ["run_grid_search_spark", "start_spark_session"]


# ===========================================================================
# Public API
# ===========================================================================

def start_spark_session(
    app_name: str = "Avazu-CTR-MLP",
    memory_gb: int = 10,
    cores: int | str = 8,
    shuffle_partitions: int | None = None,
    driver_max_result_gb: int = 2,
    memory_fraction: float = 0.6,
    storage_fraction: float = 0.5,
    adaptive: bool = True,
    log_level: str = "WARN",
) -> SparkSession:
    """Build a SparkSession configured for a single-machine setup.

    In local mode, the driver and the executor share the same JVM, so
    ``memory_gb`` is effectively the total memory Spark can use. The
    remaining memory of the machine is left for the operating system and
    other processes. As a rule of thumb, assign 50-65% of the total RAM
    to Spark on a workstation.

    Args:
        app_name (str): Name of the Spark application. Defaults to
            ``"Avazu-CTR-MLP"``.
        memory_gb (int): Memory in GB assigned to the driver. On a 16 GB
            machine, 10 GB is a reasonable value, leaving 6 GB for the
            OS. Defaults to 10.
        cores (int or str): Number of CPU cores to use. Pass ``"*"`` to
            use all available cores. Defaults to 8.
        shuffle_partitions (int or None): Number of partitions after a
            shuffle. When ``None``, defaults to
            ``max(200, cores * 3)``. Defaults to ``None``.
        driver_max_result_gb (int): Maximum size in GB for results
            returned to the driver. Defaults to 2.
        memory_fraction (float): Fraction of the driver memory reserved
            for execution and storage. Defaults to 0.6.
        storage_fraction (float): Fraction of the previous region
            reserved for cached data. Defaults to 0.5.
        adaptive (bool): If ``True``, enables Adaptive Query Execution.
            Defaults to ``True``.
        log_level (str): Spark log level. Defaults to ``"WARN"``.

    Returns:
        SparkSession: The configured session.

    Raises:
        ValueError: If ``memory_gb`` is not positive, if ``cores`` is not
            a positive integer or ``"*"``, or if ``memory_fraction`` or
            ``storage_fraction`` fall outside ``(0, 1)``.
    """
    if not isinstance(memory_gb, (int, float)) or memory_gb <= 0:
        raise ValueError("'memory_gb' must be a positive number.")

    if not (cores == "*" or (isinstance(cores, int) and cores > 0)):
        raise ValueError("'cores' must be '*' or a positive integer.")

    if not (0 < memory_fraction < 1):
        raise ValueError("'memory_fraction' must be in (0, 1).")

    if not (0 < storage_fraction < 1):
        raise ValueError("'storage_fraction' must be in (0, 1).")

    if shuffle_partitions is None:
        numeric_cores = cores if isinstance(cores, int) else 8
        shuffle_partitions = max(200, numeric_cores * 3)

    master = f"local[{cores}]"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.driver.maxResultSize", f"{driver_max_result_gb}g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.memory.fraction", str(memory_fraction))
        .config("spark.memory.storageFraction", str(storage_fraction))
    )

    if adaptive:
        builder = (
            builder
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)

    logger.info(
        "Spark session started | master=%s | driver_memory=%sGB | "
        "shuffle_partitions=%d",
        master, memory_gb, shuffle_partitions,
    )

    return spark


def run_grid_search_spark(
    train_df: DataFrame,
    param_grid: dict,
    model_path: Path,
    results_filename: str = "grid_spark.json",
    smoothing: float = 25.0,
    num_folds: int = 5,
    seed: int = 42,
    num_classes: int = 2,
) -> pd.DataFrame:
    """Run a grid search over the MLP hyperparameters with stratified K-fold CV.

    For each combination in ``param_grid``, trains the full pipeline
    ``num_folds`` times. In every iteration, one fold is held out for
    validation and the remaining folds are used for training. All
    state-dependent transformations are computed inside the fold loop on
    the training partition only, so no transformation sees the validation
    data during its fit.

    The results of every combination, including the AUC and fit time of
    each individual fold, are written to a JSON file as soon as the
    combination finishes. If the checkpoint file already exists, the
    search resumes from where it left off.

    Args:
        train_df (pyspark.sql.DataFrame): Training data.
        param_grid (dict): Mapping with keys ``layers``, ``stepSize`` and
            ``maxIter``. The ``layers`` values exclude the input and
            output layers, which are added automatically.
        model_path (pathlib.Path): Directory where the JSON checkpoint is
            written. Created if missing.
        results_filename (str): Name of the checkpoint JSON file.
            Defaults to ``"grid_spark.json"``.
        smoothing (float): Smoothing parameter for the target encoder.
            Defaults to 25.0.
        num_folds (int): Number of cross-validation folds. Defaults to 5.
        seed (int): Random seed used to build the folds and to initialize
            the MLP. Defaults to 42.
        num_classes (int): Number of output classes. Defaults to 2.

    Returns:
        pandas.DataFrame: Results sorted by ``mean_auc`` in descending
        order, with columns ``signature``, ``params``, ``fold_scores``,
        ``fold_times``, ``mean_auc``, ``std_auc``, ``mean_fit_time`` and
        ``total_fit_time``.

    Raises:
        ValueError: If any key of ``param_grid`` is missing or empty.
    """
    for key in ("layers", "stepSize", "maxIter"):
        if key not in param_grid or not param_grid[key]:
            raise ValueError(f"'param_grid' must contain a non-empty '{key}'.")

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    results_path = model_path / results_filename

    logger.info("Caching raw training data")
    train_df = train_df.cache()
    train_df.count()

    logger.info("Creating %d stratified folds", num_folds)
    folds = _make_stratified_folds(train_df, TARGET, num_folds, seed)
    for fold_df in folds:
        fold_df.cache()
        fold_df.count()

    logger.info("Precomputing per-fold statistics")
    fold_stats: list[tuple[dict, dict]] = []
    for fold_idx in tqdm(range(num_folds), desc="Fold stats", unit="fold"):
        train_folds = [folds[j] for j in range(num_folds) if j != fold_idx]
        train_part = train_folds[0]
        for tf in train_folds[1:]:
            train_part = train_part.union(tf)

        dominants = _compute_dominants(train_part)
        top_k_lists = _compute_top_k_lists(train_part)
        fold_stats.append((dominants, top_k_lists))

    logger.info("Probing feature dimension")
    num_features = _probe_num_features(
        folds=folds,
        fold_stats=fold_stats,
        num_folds=num_folds,
        smoothing=smoothing,
        seed=seed,
    )
    logger.info("Total number of features: %d", num_features)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("results", [])
        done = {entry["signature"] for entry in data}
        logger.info(
            "Resuming from checkpoint: %d combinations already done",
            len(done),
        )
    else:
        data = []
        done = set()
        payload = {
            "metadata": {
                "num_folds": num_folds,
                "smoothing": smoothing,
                "num_features": num_features,
                "grid_size": None,
            },
            "results": data,
        }

    keys = list(param_grid.keys())
    combinations = [
        dict(zip(keys, combo))
        for combo in itertools.product(*param_grid.values())
    ]
    total = len(combinations)
    payload["metadata"]["grid_size"] = total
    logger.info("Grid size: %d combinations", total)

    evaluator = BinaryClassificationEvaluator(
        labelCol=TARGET,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    bar = tqdm(combinations, desc="GridSearch", unit="combo")
    grid_start = time.time()

    for params in bar:
        signature = str(sorted(params.items()))

        if signature in done:
            continue

        fold_scores: list[float] = []
        fold_times: list[float] = []

        for fold_idx in range(num_folds):
            bar.set_postfix_str(f"fold {fold_idx + 1}/{num_folds}")

            val_fold = folds[fold_idx]

            train_folds = [
                folds[j] for j in range(num_folds) if j != fold_idx
            ]
            train_part = train_folds[0]
            for tf in train_folds[1:]:
                train_part = train_part.union(tf)

            dominants, top_k_lists = fold_stats[fold_idx]
            train_part_t = _apply_categorical_transforms(
                train_part, dominants, top_k_lists,
            )
            val_fold_t = _apply_categorical_transforms(
                val_fold, dominants, top_k_lists,
            )

            hidden = params["layers"]
            if isinstance(hidden, (tuple, list)):
                hidden_list = list(hidden)
            else:
                hidden_list = [hidden]
            layers = [num_features] + hidden_list + [num_classes]

            pipeline = _build_full_pipeline(
                smoothing=smoothing,
                layers=layers,
                step_size=params["stepSize"],
                max_iter=params["maxIter"],
                seed=seed,
            )

            fold_start = time.time()
            model = pipeline.fit(train_part_t)
            fold_time = time.time() - fold_start

            predictions = model.transform(val_fold_t)
            auc = float(evaluator.evaluate(predictions))

            fold_scores.append(auc)
            fold_times.append(fold_time)

            bar.set_postfix_str(
                f"fold {fold_idx + 1}/{num_folds} AUC={auc:.4f}"
            )

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        mean_fit_time = float(np.mean(fold_times))
        total_fit_time = float(np.sum(fold_times))

        entry = {
            "signature": signature,
            "params": {k: _to_jsonable(v) for k, v in params.items()},
            "fold_scores": fold_scores,
            "fold_times": fold_times,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_fit_time": mean_fit_time,
            "total_fit_time": total_fit_time,
        }
        data.append(entry)
        payload["results"] = data
        payload["metadata"]["total_wall_time"] = time.time() - grid_start

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        bar.set_postfix_str(f"mean AUC={mean_auc:.4f}")

    bar.close()

    total_wall = time.time() - grid_start
    logger.info("Total wall time: %.2f hours", total_wall / 3600)

    df_results = (
        pd.DataFrame(data)
        .sort_values("mean_auc", ascending=False)
        .reset_index(drop=True)
    )

    best = df_results.iloc[0]
    logger.info("Best AUC: %.4f", best["mean_auc"])
    logger.info("Best params: %s", best["params"])

    return df_results


# ===========================================================================
# Private helpers
# ===========================================================================

def _compute_dominants(train_df: DataFrame) -> dict:
    """Return the dominant category for each dichotomized or hybrid column.

    Only uses column frequencies, never the target column.

    Args:
        train_df (pyspark.sql.DataFrame): Training data.

    Returns:
        dict: Mapping from column name to its dominant value.
    """
    dominants = {}
    for col_name in DICHOTOMIZE + TE_HYBRID:
        row = (
            train_df
            .groupBy(col_name)
            .count()
            .orderBy(F.desc("count"))
            .first()
        )
        dominants[col_name] = row[col_name]
    return dominants


def _compute_top_k_lists(train_df: DataFrame) -> dict:
    """Return the list of top-K categories for each Top-K column.

    Only uses column frequencies, never the target column.

    Args:
        train_df (pyspark.sql.DataFrame): Training data.

    Returns:
        dict: Mapping from column name to the list of its top-K values.
    """
    top_k_lists = {}
    for col_name, k in TOP_K.items():
        rows = (
            train_df
            .groupBy(col_name)
            .count()
            .orderBy(F.desc("count"))
            .limit(k)
            .collect()
        )
        top_k_lists[col_name] = [row[col_name] for row in rows]
    return top_k_lists


def _apply_categorical_transforms(
    df: DataFrame,
    dominants: dict,
    top_k_lists: dict,
) -> DataFrame:
    """Apply dichotomization and Top-K + Other on the DataFrame.

    Deterministic, no target involvement.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame.
        dominants (dict): Mapping from column to its dominant value.
        top_k_lists (dict): Mapping from column to its top-K values.

    Returns:
        pyspark.sql.DataFrame: Transformed DataFrame.
    """
    for col_name in DICHOTOMIZE:
        dominant = dominants[col_name]
        df = df.withColumn(
            f"{col_name}_is_dominant",
            F.when(F.col(col_name) == dominant, 1.0).otherwise(0.0),
        )

    for col_name in TE_HYBRID:
        dominant = dominants[col_name]
        df = df.withColumn(
            f"{col_name}_is_dominant",
            F.when(F.col(col_name) == dominant, 1.0).otherwise(0.0),
        )

    for col_name, top_values in top_k_lists.items():
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name).isin(top_values), F.col(col_name))
             .otherwise(F.lit("Other")),
        )

    return df


def _make_stratified_folds(
    train_df: DataFrame,
    target_col: str,
    num_folds: int,
    seed: int,
) -> list:
    """Build stratified K folds by splitting each class separately.

    The dataset is split by class first, and ``randomSplit`` is applied
    inside each class with a class-specific seed. Fold ``i`` is the union
    of the ``i``-th partition of each class, guaranteeing that every fold
    preserves the class proportions of the full training set.

    Args:
        train_df (pyspark.sql.DataFrame): Training data.
        target_col (str): Name of the binary target column.
        num_folds (int): Number of folds.
        seed (int): Base random seed.

    Returns:
        list of pyspark.sql.DataFrame: List of length ``num_folds``.
    """
    classes = sorted(
        row[target_col]
        for row in train_df.select(target_col).distinct().collect()
    )

    weights = [1.0 / num_folds] * num_folds
    class_folds = {}
    for cls_idx, cls in enumerate(classes):
        class_df = train_df.filter(F.col(target_col) == cls)
        class_folds[cls] = class_df.randomSplit(
            weights, seed=seed + cls_idx,
        )

    folds = []
    for i in range(num_folds):
        combined = class_folds[classes[0]][i]
        for cls in classes[1:]:
            combined = combined.union(class_folds[cls][i])
        folds.append(combined)

    return folds


def _probe_num_features(
    folds: list,
    fold_stats: list,
    num_folds: int,
    smoothing: float,
    seed: int,
) -> int:
    """Return the feature dimension via a probe fit on the first fold.

    Uses the first fold's training partition with its own fold-specific
    statistics, builds the encoding stages without the MLP and reads the
    size of the resulting features vector. This guarantees the dimension
    matches whatever the actual pipeline produces in the main loop.

    Args:
        folds (list): List of fold DataFrames.
        fold_stats (list): List of ``(dominants, top_k_lists)`` per fold.
        num_folds (int): Number of folds.
        smoothing (float): Smoothing for the target encoder.
        seed (int): Random seed.

    Returns:
        int: Number of features feeding the MLP input layer.
    """
    train_folds = [folds[j] for j in range(num_folds) if j != 0]
    train_part = train_folds[0]
    for tf in train_folds[1:]:
        train_part = train_part.union(tf)

    dominants, top_k_lists = fold_stats[0]
    train_part_t = _apply_categorical_transforms(
        train_part, dominants, top_k_lists,
    )

    probe_stages = _build_encoding_stages(smoothing)
    sample = train_part_t.sample(fraction=0.01, seed=seed)
    probe_model = Pipeline(stages=probe_stages).fit(sample)
    sample_out = probe_model.transform(sample.limit(1))
    return int(
        sample_out.select(F.size(F.col("features"))).first()[0]
    )


def _build_encoding_stages(smoothing: float) -> list:
    """Build the encoding stages that precede the MLP.

    Args:
        smoothing (float): Smoothing for the target encoder.

    Returns:
        list: List of encoding stages.
    """
    categorical = list(TOP_K.keys()) + [TIME_BAND]
    idx_cols = [f"{c}_idx" for c in categorical]
    ohe_cols = [f"{c}_ohe" for c in categorical]
    te_inputs = list(TE_HYBRID) + list(TE_PURE)
    te_outputs = [f"{c}_te" for c in te_inputs]

    final_input_cols = (
        ["scaled_num"]
        + NUMERIC_PASSTHROUGH
        + [f"{c}_is_dominant" for c in DICHOTOMIZE]
        + [f"{c}_is_dominant" for c in TE_HYBRID]
        + te_outputs
        + ohe_cols
    )

    return [
        *[
            StringIndexer(
                inputCol=c,
                outputCol=f"{c}_idx",
                handleInvalid="keep",
            )
            for c in categorical
        ],
        OneHotEncoder(
            inputCols=idx_cols,
            outputCols=ohe_cols,
            dropLast=False,
            handleInvalid="keep",
        ),
        TargetEncoder(
            inputCols=te_inputs,
            outputCols=te_outputs,
            labelCol=TARGET,
            targetType="binary",
            smoothing=smoothing,
        ),
        VectorAssembler(
            inputCols=NUMERIC_SCALED,
            outputCol="num_to_scale",
            handleInvalid="skip",
        ),
        StandardScaler(
            inputCol="num_to_scale",
            outputCol="scaled_num",
            withMean=True,
            withStd=True,
        ),
        VectorAssembler(
            inputCols=final_input_cols,
            outputCol="features",
            handleInvalid="skip",
        ),
    ]


def _build_full_pipeline(
    smoothing: float,
    layers: list,
    step_size: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """Assemble the full pipeline with fresh encoders, scaler and MLP.

    Every call creates new stage instances, so when ``Pipeline.fit`` is
    invoked, all encoders and the scaler are fit on the training partition
    only.

    Args:
        smoothing (float): Smoothing for the target encoder.
        layers (list of int): Full MLP architecture.
        step_size (float): Learning rate.
        max_iter (int): Maximum iterations.
        seed (int): Random seed.

    Returns:
        pyspark.ml.Pipeline: The complete pipeline.
    """
    stages = _build_encoding_stages(smoothing)
    stages.append(
        MultilayerPerceptronClassifier(
            featuresCol="features",
            labelCol=TARGET,
            layers=layers,
            stepSize=step_size,
            maxIter=max_iter,
            seed=seed,
            blockSize=128,
        )
    )
    return Pipeline(stages=stages)


def _to_jsonable(x: Any) -> Any:
    """Recursively convert tuples, arrays and numpy types to JSON-safe values.

    Args:
        x: Value to convert.

    Returns:
        A JSON-serializable version of ``x``.
    """
    if isinstance(x, (tuple, list)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x