"""
sklearn_trainer
===============

scikit-learn toolkit for the Avazu CTR MLP: stratified sampling,
encoding and grid search with JSON checkpoints.

Public entry points
-------------------
Sampling
    get_stratified_sample

Transformers
    AvazuPreprocessor
    SelectiveScaler

Training
    MLPTrainerWithTracking
    plot_learning_curves_pair
    save_sklearn_pipeline

Grid search
    run_grid_search_manual_kfold
    grid_results_table
    show_grid_results

Rendering
    render_figure

Leakage guarantees
------------------
``AvazuPreprocessor`` computes all state-dependent statistics (dominant
category, top-K ranking, target rate per category) inside ``fit``. When
used inside a pipeline that is cloned per fold, these statistics are
refit on each fold's training partition only. No transformation sees
the validation fold during its fit.

Dependencies
------------
duckdb, numpy, pandas, matplotlib, scikit-learn, tqdm, joblib
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import logging
import time
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib import ticker
from pandas.io.formats.style_render import CSSDict
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

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

DEFAULT_TOP_K = {
    "banner_pos":    2,
    "C18":           3,
    "app_category":  3,
    "site_category": 3,
    "app_domain":    6,
    "C19":          10,
    "C20":          10,
    "C21":          10,
}

_VALID_FIGURE_FORMATS = ("svg", "png")

__all__ = [
    "AvazuPreprocessor",
    "MLPTrainerWithTracking",
    "SelectiveScaler",
    "get_stratified_sample",
    "grid_results_table",
    "plot_learning_curves_pair",
    "render_figure",
    "run_grid_search_manual_kfold",
    "save_sklearn_pipeline",
    "show_grid_results",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stratified_sample(
    con,
    relation: str,
    n: int = 1_000_000,
    target: str = "click",
    seed: int = 42,
    display: bool = True,
    set_labels: tuple = ("Original", "Sample"),
    column_titles: tuple = (
        "Set",
        "Records",
        "Click Rate",
        "Δ Click Rate",
    ),
) -> pd.DataFrame:
    """Draw a stratified sample of ``n`` rows from a DuckDB relation.

    The sample preserves the proportion of the target classes of the
    source relation. The number of rows drawn per class is proportional
    to its frequency in the source, and the sampling within each class is
    reproducible given the same seed. When ``display=True``, a styled
    HTML table summarizes the size and the target proportion of the
    original relation and of the resulting sample, together with the
    difference between the two proportions.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of a table or view.
        n (int): Total number of rows to draw. Defaults to 1,000,000.
        target (str): Column whose distribution must be preserved.
            Defaults to ``"click"``.
        seed (int): Seed for the reservoir sampler. Defaults to 42.
        display (bool): If ``True``, renders a styled HTML table that
            compares the original relation and the drawn sample.
            Defaults to ``True``.
        set_labels (tuple of 2 str): Display labels for the original
            relation and for the sample, in that order. Defaults to
            ``("Original", "Sample")``.
        column_titles (tuple of 4 str): Header labels for the summary
            table, in the order ``(set, records, rate, delta)``. Defaults
            to ``("Set", "Records", "Click Rate", "Δ Click Rate")``.

    Returns:
        pandas.DataFrame: The sampled rows.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if
            ``relation`` or ``target`` is not a string, if ``display``
            is not a boolean, or if ``set_labels`` or ``column_titles``
            is not a tuple/list of the expected length.
        ValueError: If ``n`` is not a positive integer, if ``n`` is not
            strictly smaller than the relation size, or if any element
            of ``set_labels`` or ``column_titles`` is not a string.
    """

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")
    _validate_string(relation, "relation")
    _validate_string(target, "target")
    _validate_positive_int(n, "n")
    _validate_bool(display, "display")
    _validate_string_tuple(set_labels, "set_labels", length=2)
    _validate_string_tuple(column_titles, "column_titles", length=4)

    total = int(
        con.sql(f"SELECT COUNT(*) AS n FROM {relation}").df()["n"].iloc[0]
    )

    if n >= total:
        raise ValueError(
            f"'n' ({n:,}) must be smaller than the relation size "
            f"({total:,})."
        )

    df_classes = con.sql(f"""
        SELECT "{target}" AS cls, COUNT(*) AS cnt
        FROM {relation}
        GROUP BY "{target}"
    """).df()

    positive_mask = df_classes["cls"].astype(int) == 1
    original_clicks = int(df_classes.loc[positive_mask, "cnt"].sum())
    original_prop = original_clicks / total if total > 0 else 0.0

    parts = []
    for _, row in df_classes.iterrows():
        cls = row["cls"]
        cnt = int(row["cnt"])
        n_cls = max(1, round(n * cnt / total))
        parts.append(f"""
            SELECT * FROM (
                SELECT * FROM {relation}
                WHERE "{target}" = {cls}
            ) AS filtered
            USING SAMPLE {n_cls} ROWS (reservoir, {seed})
        """)

    query = " UNION ALL ".join(parts)
    df_sample = con.sql(query).df()

    if display:
        sample_size = len(df_sample)
        sample_clicks = int((df_sample[target] == 1).sum())
        sample_prop = sample_clicks / sample_size if sample_size > 0 else 0.0
        delta_prop = sample_prop - original_prop

        col_set, col_records, col_rate, col_delta = column_titles

        rows = [
            {
                col_set: set_labels[0],
                col_records: f"{total:,}",
                col_rate: _format_apa7_no_zero(original_prop, decimals=4),
                col_delta: "—",
            },
            {
                col_set: set_labels[1],
                col_records: f"{sample_size:,}",
                col_rate: _format_apa7_no_zero(sample_prop, decimals=4),
                col_delta: _format_apa7_no_zero(delta_prop, decimals=4),
            },
        ]

        df_display = pd.DataFrame(rows)

        styled = (
            df_display.style
            .hide(axis="index")
            .set_table_styles(_table_style_rules(left_align_positions=(1,)))
        )
        _render_styled(styled)

    return df_sample

class AvazuPreprocessor(BaseEstimator, TransformerMixin):
    """Apply the encoding strategy defined in the EDA to raw Avazu columns.

    The transformer learns all state-dependent statistics (dominant
    category, top-K ranking, target rate per category) from the training
    data during ``fit`` and applies them to any subsequent data during
    ``transform``. It never touches the target of the data passed to
    ``transform``, so it can be safely applied to the test set.

    Encoding strategies applied:
        - Dichotomization: replace each column by a binary indicator of
          its dominant category.
        - Top-K + Other: keep the K most frequent categories and collapse
          the rest into a single ``Other`` level.
        - Hybrid: emit a binary indicator of the dominant category and a
          target-encoded column for the rest.
        - Target encoding: replace each category by its smoothed target
          rate.
        - One-hot: encode the time band and the top-K columns with
          ``OneHotEncoder``.

    Args:
        dichotomize (tuple of str): Columns to dichotomize. Defaults to
            ``("C1", "C15", "C16", "device_type", "device_conn_type")``.
        top_k (dict or None): Mapping from column name to the number of
            top categories to keep. When ``None``, ``DEFAULT_TOP_K`` is
            used. Defaults to ``None``.
        te_hybrid (tuple of str): Columns that receive the hybrid encoding
            (binary dominant + target encoding). Defaults to
            ``("app_id", "site_domain", "site_id")``.
        te_pure (tuple of str): Columns that receive pure target encoding.
            Defaults to ``("C14", "C17", "device_model")``.
        numeric (tuple of str): Columns passed through unchanged. Defaults
            to ``("day", "hour_sin", "hour_cos", "log_freq_device_id",
            "log_freq_device_ip")``.
        time_band (str): Name of the time band column, encoded with
            one-hot. Defaults to ``"time_band"``.
        smoothing (float): Smoothing parameter for target encoding.
            Defaults to 50.
        min_category_size (int): Categories with fewer than this many
            training occurrences are treated as ``Other`` for target
            encoding purposes. Defaults to 10.
    """

    def __init__(
        self,
        dichotomize=("C1", "C15", "C16", "device_type", "device_conn_type"),
        top_k=None,
        te_hybrid=("app_id", "site_domain", "site_id"),
        te_pure=("C14", "C17", "device_model"),
        numeric=(
            "day", "hour_sin", "hour_cos",
            "log_freq_device_id", "log_freq_device_ip",
        ),
        time_band="time_band",
        smoothing=50,
        min_category_size=10,
    ):
        self.dichotomize = dichotomize
        self.top_k = top_k
        self.te_hybrid = te_hybrid
        self.te_pure = te_pure
        self.numeric = numeric
        self.time_band = time_band
        self.smoothing = smoothing
        self.min_category_size = min_category_size

    def fit(self, X, y):
        """Learn the encoding statistics from the training data.

        Args:
            X (pandas.DataFrame): Raw feature matrix.
            y (pandas.Series or numpy.ndarray): Binary target.

        Returns:
            AvazuPreprocessor: The fitted transformer.
        """
        self.top_k_ = (
            self.top_k if self.top_k is not None else dict(DEFAULT_TOP_K)
        )

        y = pd.Series(np.asarray(y).ravel(), name="__target__")

        self.dominant_ = {
            col: X[col].value_counts().idxmax()
            for col in self.dichotomize
        }

        self.dominant_hybrid_ = {
            col: X[col].value_counts().idxmax()
            for col in self.te_hybrid
        }

        self.top_k_lists_ = {
            col: X[col].value_counts().head(k).index.tolist()
            for col, k in self.top_k_.items()
        }

        self.te_tables_ = self._build_te_tables(X, y)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_cols = list(self.top_k_.keys()) + [self.time_band]
        self.ohe_.fit(self._prepare_ohe_input(X, ohe_cols))
        self.ohe_columns_ = ohe_cols

        self.feature_names_out_ = self._compute_feature_names()

        return self

    def transform(self, X):
        """Apply the fitted encodings to new data.

        Args:
            X (pandas.DataFrame): Raw feature matrix.

        Returns:
            numpy.ndarray: The encoded feature matrix as float32.
        """
        parts = [X[list(self.numeric)].to_numpy(dtype=np.float32)]

        for col in self.dichotomize:
            dominant = self.dominant_[col]
            parts.append(
                (X[col] == dominant).to_numpy(dtype=np.float32).reshape(-1, 1)
            )

        for col in self.te_hybrid:
            dominant = self.dominant_hybrid_[col]
            is_dominant = (X[col] == dominant).to_numpy(dtype=np.float32)
            parts.append(is_dominant.reshape(-1, 1))

            te_col = self._apply_te(X, col, self.te_tables_[col])
            parts.append(te_col.reshape(-1, 1))

        for col in self.te_pure:
            te_col = self._apply_te(X, col, self.te_tables_[col])
            parts.append(te_col.reshape(-1, 1))

        ohe_input = self._prepare_ohe_input(X, self.ohe_columns_)
        parts.append(
            np.asarray(self.ohe_.transform(ohe_input)).astype(np.float32)
        )

        return np.hstack(parts).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names.

        Args:
            input_features: Ignored. Present for scikit-learn API
                compatibility.

        Returns:
            numpy.ndarray: Array of feature names.
        """
        return np.asarray(self.feature_names_out_)

    def _build_te_tables(self, X, y):
        """Compute the smoothed target rate per category for each TE column.

        Args:
            X (pandas.DataFrame): Raw feature matrix.
            y (pandas.Series): Binary target.

        Returns:
            dict: Mapping from column name to a dictionary with keys
            ``map`` (category to smoothed rate) and ``fallback`` (global
            mean).
        """
        global_mean = y.mean()
        tables = {}

        for col in list(self.te_hybrid) + list(self.te_pure):
            df = pd.DataFrame({"cat": X[col].values, "y": y.values})
            agg = df.groupby("cat")["y"].agg(["sum", "count"])
            agg["rate"] = (
                agg["sum"] + self.smoothing * global_mean
            ) / (agg["count"] + self.smoothing)
            tables[col] = {
                "map": agg["rate"].to_dict(),
                "fallback": global_mean,
            }

        return tables

    def _apply_te(self, X, col, table):
        """Apply the fitted target encoding table to a column.

        Args:
            X (pandas.DataFrame): Raw feature matrix.
            col (str): Name of the column.
            table (dict): Encoding table with keys ``map`` and
                ``fallback``.

        Returns:
            numpy.ndarray: Encoded column as float32.
        """
        return (
            X[col]
            .map(table["map"])
            .fillna(table["fallback"])
            .to_numpy(dtype=np.float32)
        )

    def _prepare_ohe_input(self, X, cols):
        """Prepare the input DataFrame for the one-hot encoder.

        Args:
            X (pandas.DataFrame): Raw feature matrix.
            cols (list of str): Columns to prepare.

        Returns:
            pandas.DataFrame: DataFrame with the columns cast to string
            and rare categories collapsed into ``Other``.
        """
        out = pd.DataFrame(index=X.index)
        for col in cols:
            if col == self.time_band:
                out[col] = X[col].astype(str)
            else:
                top = self.top_k_lists_[col]
                out[col] = X[col].where(X[col].isin(top), "Other").astype(str)
        return out

    def _compute_feature_names(self):
        """Compute the output feature names for downstream use.

        Returns:
            list of str: Feature names in the order produced by
            ``transform``.
        """
        names = list(self.numeric)
        names += [f"{c}_is_dominant" for c in self.dichotomize]
        for c in self.te_hybrid:
            names.append(f"{c}_is_dominant")
            names.append(f"{c}_te")
        names += [f"{c}_te" for c in self.te_pure]
        names += list(self.ohe_.get_feature_names_out(self.ohe_columns_))
        return names


class SelectiveScaler(BaseEstimator, TransformerMixin):
    """Scale only a subset of columns identified by their index.

    Wraps a ``StandardScaler`` and applies it only to the columns whose
    positions are listed in ``indices``. The remaining columns are passed
    through unchanged. Useful when the input is a NumPy array produced by
    an upstream transformer that does not preserve column names.

    Args:
        indices (tuple or list of int): 0-based positions of the columns
            to scale. Must be non-empty. Defaults to ``(3, 4)``.
    """

    def __init__(self, indices=(3, 4)):
        self.indices = indices

    def fit(self, X, y=None):
        """Fit the internal scaler on the selected columns.

        Args:
            X (numpy.ndarray): Input array.
            y: Ignored. Present for API compatibility.

        Returns:
            SelectiveScaler: The fitted transformer.
        """
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[:, self.indices])
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Scale the selected columns and pass the rest through.

        Args:
            X (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: The transformed array with the same shape.
        """
        X_out = np.array(X, dtype=np.float32, copy=True)
        X_out[:, self.indices] = self.scaler_.transform(X[:, self.indices])
        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return the feature names (passthrough).

        Args:
            input_features (array-like or None): Array of input feature
                names. When ``None``, generic names are returned.

        Returns:
            numpy.ndarray: The input feature names unchanged.
        """
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)]
            )
        return np.asarray(input_features)


class MLPTrainerWithTracking:
    """Train an MLP one epoch at a time while tracking learning curves.

    Wraps a scikit-learn pipeline of the form
    ``[preprocessor_1, ..., MLPClassifier]`` and drives the MLP epoch by
    epoch using ``partial_fit``, so that training loss, validation loss,
    training AUC and validation AUC can be recorded after every epoch and
    a manual early-stopping policy can be applied. The best weights
    (lowest validation loss, subject to ``min_delta``) are restored when
    early stopping triggers.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline whose last step is
            an ``MLPClassifier``; all preceding steps are treated as the
            preprocessor. Steps are cloned, so the original is not
            modified.
        epochs (int): Maximum number of epochs. Defaults to 150.
        patience (int): Number of consecutive epochs without an
            improvement greater than ``min_delta`` before early stopping.
            Defaults to 15.
        min_delta (float): Minimum improvement in validation loss required
            to reset the patience counter. Defaults to ``1e-4``.

    Raises:
        ValueError: If ``epochs`` or ``patience`` is not a positive
            integer, or if ``min_delta`` is not a non-negative number.
    """

    def __init__(self, pipeline, epochs=150, patience=15, min_delta=1e-4):
        _validate_positive_int(epochs, "epochs")
        _validate_positive_int(patience, "patience")
        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
            or min_delta < 0
        ):
            raise ValueError("'min_delta' must be a non-negative number.")

        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

        preprocessor_steps = [
            clone(estimator) for _, estimator in pipeline.steps[:-1]
        ]
        self.preprocessor = make_pipeline(*preprocessor_steps)

        self.mlp = clone(pipeline.steps[-1][1])
        self.mlp.early_stopping = False

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": [],
        }

    def fit(self, X_train, y_train, X_val, y_val):
        """Fit the pipeline epoch by epoch and record the learning curves.

        Args:
            X_train (pandas.DataFrame): Raw training features.
            y_train (pandas.Series or numpy.ndarray): Training target.
            X_val (pandas.DataFrame): Raw validation features.
            y_val (pandas.Series or numpy.ndarray): Validation target.

        Returns:
            pandas.DataFrame: The per-epoch history with columns
            ``["epoch", "train_loss", "val_loss", "train_auc",
            "val_auc"]``.
        """
        print("Transforming data with the preprocessing pipeline...")
        X_tr_trans = self.preprocessor.fit_transform(X_train, y_train)
        X_va_trans = self.preprocessor.transform(X_val)

        classes = np.unique(y_train)
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights = None

        print(f"Starting iterative training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            self.mlp.partial_fit(X_tr_trans, y_train, classes=classes)

            tr_proba = self.mlp.predict_proba(X_tr_trans)[:, 1]
            va_proba = self.mlp.predict_proba(X_va_trans)[:, 1]

            tr_loss = log_loss(y_train, tr_proba)
            va_loss = log_loss(y_val, va_proba)
            tr_auc = roc_auc_score(y_train, tr_proba)
            va_auc = roc_auc_score(y_val, va_proba)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_auc"].append(tr_auc)
            self.history["val_auc"].append(va_auc)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d}/{self.epochs:03d} | "
                    f"Train Loss: {tr_loss:.4f} - Val Loss: {va_loss:.4f} | "
                    f"Train AUC: {tr_auc:.4f} - Val AUC: {va_auc:.4f}"
                )

            if va_loss < (best_val_loss - self.min_delta):
                best_val_loss = va_loss
                patience_counter = 0
                best_weights = (
                    [coef.copy() for coef in self.mlp.coefs_],
                    [bias.copy() for bias in self.mlp.intercepts_],
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch}. "
                        f"Restoring best weights."
                    )
                    if best_weights is not None:
                        self.mlp.coefs_, self.mlp.intercepts_ = best_weights
                    break

        return pd.DataFrame(self.history)

    def plot_history(self, render=True, fmt="svg", dpi=100, **kwargs):
        """Render the accumulated training history as a two-panel figure.

        Thin wrapper around :func:`plot_learning_curves_pair` that feeds
        it the per-epoch history recorded by :meth:`fit`. All keyword
        arguments accepted by that function can be forwarded through
        ``**kwargs``.

        Args:
            render (bool): If ``True``, the figure is rendered inline.
                Defaults to ``True``.
            fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
            dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.
            **kwargs: Additional keyword arguments forwarded to
                :func:`plot_learning_curves_pair`.

        Returns:
            tuple: ``(fig, (ax_loss, ax_auc))``.
        """
        return plot_learning_curves_pair(
            pd.DataFrame(self.history),
            render=render,
            fmt=fmt,
            dpi=dpi,
            **kwargs,
        )


def run_grid_search_manual_kfold(
    pipe,
    param_grid: dict,
    X_train,
    y_train,
    model_path: Path,
    cv: int = 5,
    scoring: str = "roc_auc",
    results_filename: str = "grid_results.json",
    random_state: int = 42,
):
    """Run a grid search with manual stratified K-fold and JSON checkpoints.

    Iterates over the Cartesian product of ``param_grid`` values. For each
    combination, runs a stratified K-fold cross-validation where each fold
    is used once for validation and the remaining K-1 folds for training.
    The results of every combination, including the AUC and fit time of
    each individual fold, are written to a JSON file as soon as the
    combination finishes, so the search can be interrupted and resumed
    without losing progress. A tqdm progress bar shows the state in real
    time.

    The function does not refit the best combination on the full training
    set. That step is left to the caller.

    Args:
        pipe (sklearn.pipeline.Pipeline): Pipeline to tune. Cloned for
            each fold, so the original is never modified.
        param_grid (dict): Mapping from parameter names (with the pipeline
            step prefix, e.g. ``"mlpclassifier__alpha"``) to lists of
            candidate values.
        X_train (pandas.DataFrame): Raw feature matrix for training.
        y_train (pandas.Series or numpy.ndarray): Binary target.
        model_path (pathlib.Path): Directory where the checkpoint JSON
            will be written. Created if missing.
        cv (int): Number of cross-validation folds. Defaults to 5.
        scoring (str): Metric to optimize. Only ``"roc_auc"`` is currently
            supported. Defaults to ``"roc_auc"``.
        results_filename (str): Name of the checkpoint JSON file. Defaults
            to ``"grid_results.json"``.
        random_state (int): Seed for the stratified K-fold splitter.
            Defaults to 42.

    Returns:
        pandas.DataFrame: Sorted results, one row per combination, with
        columns ``signature``, ``params``, ``fold_scores``,
        ``fold_times``, ``mean_auc``, ``std_auc``, ``mean_fit_time`` and
        ``total_fit_time``.

    Raises:
        ValueError: If ``scoring`` is not ``"roc_auc"``.
    """
    if scoring != "roc_auc":
        raise ValueError(f"Only 'roc_auc' is supported, got '{scoring}'.")

    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    results_path = model_path / results_filename

    keys = list(param_grid.keys())
    combinations = [
        dict(zip(keys, combo))
        for combo in itertools.product(*param_grid.values())
    ]
    total = len(combinations)

    logger.info("Grid search: %d combinations, cv=%d", total, cv)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("results", [])
        done_signatures = {entry["signature"] for entry in data}
        logger.info(
            "Resuming from checkpoint: %d/%d already done",
            len(done_signatures), total,
        )
    else:
        data = []
        done_signatures = set()
        payload = {
            "metadata": {
                "grid_size": total,
                "cv": cv,
                "scoring": scoring,
                "total_wall_time": 0.0,
            },
            "results": data,
        }

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(skf.split(X_train, y_train))

    bar = tqdm(combinations, desc="GridSearch", unit="combo")
    grid_start = time.time()

    for params in bar:
        signature = str(sorted(params.items()))

        if signature in done_signatures:
            continue

        bar.set_postfix_str("training...")

        def _progress(fold, cv, auc):
            bar.set_postfix_str(f"fold {fold}/{cv} AUC={auc:.4f}")

        fold_scores, fold_times = _run_kfold(
            pipe=pipe,
            params=params,
            splits=splits,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            progress_hook=_progress,
        )

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        mean_fit_time = float(np.mean(fold_times))
        total_fit_time = float(np.sum(fold_times))

        params_json = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in params.items()
        }

        entry = {
            "signature": signature,
            "params": params_json,
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

    total_wall_time = time.time() - grid_start
    logger.info("Total wall time: %.2f min", total_wall_time / 60)

    for entry in data:
        hls_key = "mlpclassifier__hidden_layer_sizes"
        if hls_key in entry["params"]:
            entry["params"][hls_key] = tuple(entry["params"][hls_key])

    df_results = (
        pd.DataFrame(data)
        .sort_values("mean_auc", ascending=False)
        .reset_index(drop=True)
    )

    best_params = df_results.iloc[0]["params"]
    logger.info("Best AUC: %.4f", df_results.iloc[0]["mean_auc"])
    logger.info("Best params: %s", best_params)

    return df_results


def grid_results_table(
    results_path,
    top_n=5,
    auc_col="mean_auc",
    std_col="std_auc",
    auc_title="AUC",
    decimals_auc=4,
    decimals_std=4,
    param_order=None,
    param_display_names=None,
    strip_prefixes=True,
    render=True,
):
    """Render the top-N grid search results as a styled HTML table.

    Reads a JSON checkpoint produced by the grid search functions, sorts
    the combinations by mean AUC in descending order, and builds a table
    where the first columns are the hyperparameters and the last column is
    the AUC in ``mean ± std`` format. The number of rows shown is
    controlled by ``top_n``. Every cell is centered horizontally and
    vertically.

    Args:
        results_path (str or pathlib.Path): Path to the JSON checkpoint.
        top_n (int): Number of top configurations to display. Defaults to
            5.
        auc_col (str): Key in each result entry that holds the mean AUC.
            Defaults to ``"mean_auc"``.
        std_col (str): Key that holds the AUC standard deviation. Defaults
            to ``"std_auc"``.
        auc_title (str): Header for the AUC column. Defaults to ``"AUC"``.
        decimals_auc (int): Decimals for the mean AUC. Defaults to 4.
        decimals_std (int): Decimals for the AUC standard deviation.
            Defaults to 4.
        param_order (list of str or None): Explicit order of the
            hyperparameter keys. When ``None``, the order is taken from
            the first entry in the results. Defaults to ``None``.
        param_display_names (dict or None): Mapping from parameter key to
            display name. When ``None``, the key is used after optional
            prefix stripping. Defaults to ``None``.
        strip_prefixes (bool): If ``True``, remove the ``step__`` prefix
            from parameter keys (e.g. ``"mlpclassifier__alpha"`` becomes
            ``"alpha"``). Defaults to ``True``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) results table.

    Raises:
        FileNotFoundError: If ``results_path`` does not exist.
        ValueError: If the JSON contains no results, if ``top_n`` is not
            a positive integer, or if ``auc_col`` is not present in the
            entries.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    _validate_positive_int(top_n, "top_n")

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No results found in {results_path}")

    if auc_col not in results[0]:
        raise ValueError(
            f"'{auc_col}' not found in the results. Available keys: "
            f"{sorted(results[0].keys())}"
        )

    sorted_results = sorted(
        results, key=lambda r: r[auc_col], reverse=True,
    )
    top = sorted_results[:top_n]

    param_keys = (
        list(top[0]["params"].keys())
        if param_order is None
        else list(param_order)
    )

    rows = []
    for entry in top:
        row = {}
        for key in param_keys:
            value = entry["params"].get(key, "")
            display = _resolve_display_name(
                key, param_display_names, strip_prefixes
            )
            row[display] = _format_param_value(value)
        row[auc_title] = _format_auc_pm(
            entry[auc_col],
            entry.get(std_col),
            decimals_mean=decimals_auc,
            decimals_std=decimals_std,
        )
        rows.append(row)

    df_table = pd.DataFrame(rows)

    center_props = {
        "text-align": "center",
        "vertical-align": "middle",
    }

    styled = (
        df_table.style
        .hide(axis="index")
        .set_properties(**center_props) # type: ignore
        .set_table_styles(_table_style_rules(left_align_positions=()))
    )

    if render:
        _render_styled(styled)

    return df_table

def show_grid_results(
    base_path,
    subfolder=None,
    filename="grid_results.json",
    **kwargs,
):
    """Load and render a grid results table from a base path.

    Resolves the full path to the JSON checkpoint and delegates to
    :func:`grid_results_table`. Useful when the project follows the
    convention ``models/<framework>/grid_*.json``.

    Args:
        base_path (str or pathlib.Path): Base directory, typically
            ``MODEL_PATH``.
        subfolder (str or None): Optional subfolder such as ``"sklearn"``
            or ``"pyspark"``. When ``None``, the file is read directly
            from ``base_path``. Defaults to ``None``.
        filename (str): Name of the JSON checkpoint file. Defaults to
            ``"grid_results.json"``.
        **kwargs: Forwarded to :func:`grid_results_table`.

    Returns:
        pandas.DataFrame: The plain (unstyled) results table.
    """
    base_path = Path(base_path)
    results_path = (
        base_path / subfolder / filename if subfolder else base_path / filename
    )
    return grid_results_table(results_path, **kwargs)


def render_figure(fig, fmt="svg", dpi=100):
    """Render a matplotlib figure inline as centered HTML.

    Args:
        fig (matplotlib.figure.Figure): The figure to render. The figure
            is closed after encoding, so it will not also pop up as a
            separate matplotlib window.
        fmt ({"svg", "png"}): Output format. ``"svg"`` (default) embeds a
            resolution-independent vector image directly in the notebook
            output; ``dpi`` has no effect in this mode. ``"png"``
            rasterizes the figure and embeds it as a Base64-encoded
            image.
        dpi (int): Resolution (dots per inch) used only when
            ``fmt="png"``. Defaults to ``100``.

    Raises:
        ValueError: If ``fmt`` is not one of ``{"svg", "png"}``.
    """
    if fmt not in _VALID_FIGURE_FORMATS:
        raise ValueError(f"'fmt' must be 'svg' or 'png', got '{fmt}'.")

    buf = io.BytesIO()

    if fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_data = buf.getvalue().decode("utf-8")
        display(
            HTML(
                f'<div style="display: flex; justify-content: center; '
                f'width: 100%;">{svg_data}</div>'
            )
        )
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
        display(
            HTML(
                f'<div style="text-align: center; width: 100%;">'
                f'<img src="data:image/png;base64,{encoded}"></div>'
            )
        )


def plot_learning_curves_pair(
    df_history,
    titles=("Gradient Descent (Log Loss)", "ROC AUC Evolution"),
    xlabels=("Epoch", "Epoch"),
    ylabels=("Log Loss", "ROC AUC"),
    train_color="black",
    val_color="#555555",
    line_styles=("-", "--"),
    line_widths=(1.8, 1.8),
    legend_labels=("Training", "Validation"),
    show_legend=(True, False),
    legend_loc="upper left",
    legend_bbox=None,
    legend_fontsize=None,
    legend_frame=False,
    ylims=((0.39, 0.42), (0.74, 0.78)),
    y_steps=(0.005, 0.008),
    x_step=None,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(10, 10, 12),
    show_grid=False,
    figsize=(10, 4.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot learning curves side by side with fine legend placement controls.

    Creates a figure with two horizontal subplots displaying training and
    validation performance history across epochs, bounded by complete
    black rectangular frames on all four sides. The left panel shows the
    log-loss curves and the right panel the ROC-AUC curves; the right
    panel's y-axis is drawn on the right side so that both panels remain
    visually symmetric.

    Args:
        df_history (pandas.DataFrame): Per-epoch history with at least
            the columns ``train_loss``, ``val_loss``, ``train_auc`` and
            ``val_auc``. An optional ``epoch`` column drives the x-axis;
            when missing, epochs are numbered from 1.
        titles (tuple of 2 str): Panel titles. Defaults to
            ``("Gradient Descent (Log Loss)", "ROC AUC Evolution")``.
        xlabels (tuple of 2 str): Panel x-axis labels. Defaults to
            ``("Epoch", "Epoch")``.
        ylabels (tuple of 2 str): Panel y-axis labels. Defaults to
            ``("Log Loss", "ROC AUC")``.
        train_color (str or tuple of 2 str): Color of the training curve.
            Defaults to ``"black"``.
        val_color (str or tuple of 2 str): Color of the validation curve.
            Defaults to ``"#555555"``.
        line_styles (tuple of 2 str): Line styles. Defaults to
            ``("-", "--")``.
        line_widths (tuple of 2 float): Line widths. Defaults to
            ``(1.8, 1.8)``.
        legend_labels (tuple of 2 str): Legend entries. Defaults to
            ``("Training", "Validation")``.
        show_legend (bool or tuple of 2 bool): Per-panel legend toggle.
            Defaults to ``(True, False)``.
        legend_loc (str): Legend location. Defaults to ``"upper left"``.
        legend_bbox (tuple or None): Optional ``bbox_to_anchor`` for the
            legend. Defaults to ``None``.
        legend_fontsize (float or None): Legend font size. When ``None``,
            ``tick_size + 1`` is used.
        legend_frame (bool): If ``True``, draw a boxed legend frame.
            Defaults to ``False``.
        ylims (tuple of 2 (tuple or None)): Per-panel ``(ymin, ymax)``
            limits. Defaults to ``((0.39, 0.42), (0.74, 0.78))``.
        y_steps (tuple of 2 (float or None)): Per-panel y-tick spacing.
            Defaults to ``(0.005, 0.008)``.
        x_step (float or None): Spacing between x-axis ticks. When
            ``None``, matplotlib's default locator is used. Defaults to
            ``None``.
        label_size (float): Axis label font size. Defaults to 11.
        title_size (float): Panel title font size. Defaults to 12.
        tick_size (float): Tick label font size. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(10, 10, 12)``.
        show_grid (bool): If ``True``, draw a dotted grid behind the
            curves. Defaults to ``False``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(10, 4.5)``.
        render (bool): If ``True``, the figure is passed to
            :func:`render_figure` for inline rendering. Defaults to
            ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax_loss, ax_auc))``.

    Raises:
        TypeError: If ``df_history`` is not a ``pandas.DataFrame``, if
            any label argument is not a length-2 sequence of strings, or
            if ``render`` is not a boolean.
        KeyError: If ``df_history`` is missing any of the required
            columns.
    """
    if not isinstance(df_history, pd.DataFrame):
        raise TypeError("'df_history' must be a pandas DataFrame.")

    required_cols = {"train_loss", "val_loss", "train_auc", "val_auc"}
    missing = required_cols - set(df_history.columns)
    if missing:
        raise KeyError(
            f"'df_history' is missing required columns: {sorted(missing)}"
        )

    _validate_two_strings(titles, "titles")
    _validate_two_strings(xlabels, "xlabels")
    _validate_two_strings(ylabels, "ylabels")
    _validate_bool(render, "render")
    _validate_figsize(figsize)
    _validate_pads(pads)

    if isinstance(show_legend, bool):
        show_legend = (show_legend, show_legend)

    if ylims is None:
        ylims = (None, None)
    if y_steps is None:
        y_steps = (None, None)
    if legend_fontsize is None:
        legend_fontsize = tick_size + 1

    pad_x, pad_y, pad_title = pads

    if "epoch" in df_history.columns:
        epochs = df_history["epoch"].to_numpy()
    else:
        epochs = np.arange(1, len(df_history) + 1)

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=figsize)
    three_dec_formatter = ticker.FormatStrFormatter("%.3f")

    ax_loss.plot(
        epochs, df_history["train_loss"],
        label=legend_labels[0], color=train_color,
        linestyle=line_styles[0], linewidth=line_widths[0],
    )
    ax_loss.plot(
        epochs, df_history["val_loss"],
        label=legend_labels[1], color=val_color,
        linestyle=line_styles[1], linewidth=line_widths[1],
    )
    ax_loss.set_title(titles[0], fontsize=title_size, pad=pad_title, loc=title_loc)
    ax_loss.set_xlabel(xlabels[0], fontsize=label_size, labelpad=pad_x)
    ax_loss.set_ylabel(ylabels[0], fontsize=label_size, labelpad=pad_y)

    _apply_boxed_spines(ax_loss, tick_size=tick_size)
    _apply_yaxis_settings(ax_loss, ylim=ylims[0], y_step=y_steps[0])
    ax_loss.yaxis.set_major_formatter(three_dec_formatter)

    if x_step is not None:
        ax_loss.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    if show_grid:
        ax_loss.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax_loss.set_axisbelow(True)
    if show_legend[0]:
        ax_loss.legend(
            **_build_legend_kwargs(
                legend_loc, legend_frame, legend_bbox, legend_fontsize
            )
        )

    ax_auc.plot(
        epochs, df_history["train_auc"],
        label=legend_labels[0], color=train_color,
        linestyle=line_styles[0], linewidth=line_widths[0],
    )
    ax_auc.plot(
        epochs, df_history["val_auc"],
        label=legend_labels[1], color=val_color,
        linestyle=line_styles[1], linewidth=line_widths[1],
    )
    ax_auc.set_title(titles[1], fontsize=title_size, pad=pad_title, loc=title_loc)
    ax_auc.set_xlabel(xlabels[1], fontsize=label_size, labelpad=pad_x)

    ax_auc.yaxis.set_label_position("right")
    ax_auc.yaxis.tick_right()
    ax_auc.set_ylabel(ylabels[1], fontsize=label_size, labelpad=pad_y)

    _apply_boxed_spines(ax_auc, tick_size=tick_size)
    _apply_yaxis_settings(ax_auc, ylim=ylims[1], y_step=y_steps[1])
    ax_auc.yaxis.set_major_formatter(three_dec_formatter)

    if x_step is not None:
        ax_auc.xaxis.set_major_locator(ticker.MultipleLocator(x_step))
    if show_grid:
        ax_auc.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax_auc.set_axisbelow(True)
    if show_legend[1]:
        ax_auc.legend(
            **_build_legend_kwargs(
                legend_loc, legend_frame, legend_bbox, legend_fontsize
            )
        )

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax_loss, ax_auc)


def save_sklearn_pipeline(
    trainer,
    model_dir,
    pipeline_name="mlp_pipeline.joblib",
    history_name="training_history.json",
    pipeline_subdir=None,
    history_subdir=None,
    overwrite=True,
):
    """Persist a trained scikit-learn pipeline and its training history.

    Rebuilds a full ``Pipeline`` from the fitted preprocessor and the
    trained classifier stored in ``trainer``, serializes it with
    ``joblib`` so that it can be loaded and used directly with ``predict``
    and ``predict_proba``, and writes the per-epoch metric history to a
    JSON file. Both files are placed inside ``model_dir``, optionally
    under additional subdirectories.

    Args:
        trainer (object): Fitted trainer object exposing at least two
            attributes:

            - ``preprocessor``: a fitted ``sklearn.pipeline.Pipeline``
              whose last step is the encoding stage (for example
              ``AvazuPreprocessor`` or ``SelectiveScaler``).
            - ``mlp``: a fitted ``MLPClassifier`` instance.

            Optionally, the object may expose a ``history`` attribute
            with the per-epoch metrics. If it is missing, the history
            file is skipped and a warning is logged.
        model_dir (str or pathlib.Path): Base directory where the files
            are written. Created if it does not exist.
        pipeline_name (str): File name for the serialized pipeline. Must
            end in ``.joblib``. Defaults to ``"mlp_pipeline.joblib"``.
        history_name (str): File name for the JSON history. Must end in
            ``.json``. Defaults to ``"training_history.json"``.
        pipeline_subdir (str or None): Optional subdirectory inside
            ``model_dir`` for the pipeline file. When ``None``, the file
            is written directly in ``model_dir``. Defaults to ``None``.
        history_subdir (str or None): Optional subdirectory inside
            ``model_dir`` for the history file. When ``None``, the file
            is written directly in ``model_dir``. Defaults to ``None``.
        overwrite (bool): If ``False`` and the target pipeline file
            already exists, the function raises ``FileExistsError``
            instead of overwriting it. Defaults to ``True``.

    Returns:
        dict: Mapping with keys ``pipeline_path`` and ``history_path``,
        holding the resolved absolute paths of the written files.
        ``history_path`` is ``None`` when the trainer does not expose a
        ``history`` attribute.

    Raises:
        TypeError: If ``trainer`` does not expose the required
            ``preprocessor`` and ``mlp`` attributes, if ``model_dir`` is
            not a string or ``Path``, or if any file name is not a
            string.
        ValueError: If ``pipeline_name`` does not end in ``.joblib`` or
            if ``history_name`` does not end in ``.json``.
        FileExistsError: If ``overwrite`` is ``False`` and the pipeline
            file already exists.
    """
    if not hasattr(trainer, "preprocessor") or not hasattr(trainer, "mlp"):
        raise TypeError(
            "'trainer' must expose 'preprocessor' and 'mlp' attributes."
        )
    if not isinstance(model_dir, (str, Path)):
        raise TypeError("'model_dir' must be a string or a Path.")
    _validate_string(pipeline_name, "pipeline_name")
    _validate_string(history_name, "history_name")

    if not pipeline_name.endswith(".joblib"):
        raise ValueError("'pipeline_name' must end in '.joblib'.")
    if not history_name.endswith(".json"):
        raise ValueError("'history_name' must end in '.json'.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline_dir = model_dir / pipeline_subdir if pipeline_subdir else model_dir
    history_dir = model_dir / history_subdir if history_subdir else model_dir
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = pipeline_dir / pipeline_name
    history_path = history_dir / history_name

    if pipeline_path.exists() and not overwrite:
        raise FileExistsError(
            f"Pipeline already exists at {pipeline_path}. "
            "Pass overwrite=True to replace it."
        )

    preprocessor_steps = list(trainer.preprocessor.steps)
    fitted_pipeline = Pipeline(
        steps=preprocessor_steps + [("mlpclassifier", trainer.mlp)]
    )
    joblib.dump(fitted_pipeline, pipeline_path)
    logger.info("Pipeline saved to %s", pipeline_path)

    history_written = None
    if hasattr(trainer, "history"):
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(trainer.history, f, indent=2)
        history_written = history_path
        logger.info("History saved to %s", history_path)
    else:
        logger.warning(
            "Trainer does not expose a 'history' attribute. "
            "History file was not written."
        )

    return {
        "pipeline_path": pipeline_path,
        "history_path": history_written,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_string(value, name):
    """Validate that ``value`` is a string.

    Args:
        value: Value to validate.
        name (str): Parameter name used in error messages.

    Raises:
        TypeError: If ``value`` is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a string.")


def _validate_bool(value, name):
    """Validate that ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be a boolean.")


def _validate_positive_int(value, name):
    """Validate that ``value`` is a strictly positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{name}' must be a positive integer.")


def _validate_figsize(figsize):
    """Validate a ``(width, height)`` figure-size tuple."""
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise TypeError("'figsize' must be a tuple/list of length 2.")


def _validate_pads(pads):
    """Validate the ``(pad_x, pad_y, pad_title)`` padding tuple."""
    if not isinstance(pads, (tuple, list)):
        raise TypeError("'pads' must be a tuple or list of three numbers.")
    if len(pads) != 3:
        raise ValueError(
            f"'pads' must contain exactly 3 elements (pad_x, pad_y, "
            f"pad_title), got {len(pads)}."
        )


def _validate_two_strings(value, name):
    """Validate a length-2 tuple/list of strings."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"'{name}' must be a tuple or list of two strings.")
    if not all(isinstance(v, str) for v in value):
        raise TypeError(f"All elements of '{name}' must be strings.")


def _format_param_value(value):
    """Format a hyperparameter value for display in an HTML table.

    Args:
        value: Hyperparameter value (int, float, str, list, tuple, bool).

    Returns:
        str: A compact string representation.
    """
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_auc_pm(mean, std, decimals_mean=4, decimals_std=4):
    """Format an AUC value as ``mean ± std``.

    Args:
        mean (float): Mean AUC.
        std (float): Standard deviation of the AUC.
        decimals_mean (int): Number of decimals for the mean.
        decimals_std (int): Number of decimals for the standard deviation.

    Returns:
        str: String in the format ``"0.7451 ± 0.0011"``.
    """
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{decimals_mean}f}"
    return f"{mean:.{decimals_mean}f} \u00b1 {std:.{decimals_std}f}"


def _strip_step_prefix(key):
    """Remove the ``step__`` prefix from a scikit-learn parameter key.

    Args:
        key (str): Raw parameter key, e.g. ``"mlpclassifier__alpha"``.

    Returns:
        str: The key without the step prefix, e.g. ``"alpha"``. Returns
        the key unchanged when it has no ``"__"`` separator.
    """
    if "__" in key:
        return key.split("__", 1)[1]
    return key


def _resolve_display_name(key, display_names, strip_prefixes):
    """Resolve the display name for a parameter key.

    Args:
        key (str): Raw parameter key.
        display_names (dict or None): Explicit overrides.
        strip_prefixes (bool): If ``True``, strip the ``step__`` prefix
            when no explicit override is provided.

    Returns:
        str: The display name.
    """
    if display_names and key in display_names:
        return display_names[key]
    return _strip_step_prefix(key) if strip_prefixes else key


def _table_style_rules(left_align_positions=(1,)) -> list[CSSDict]:
    """Return the shared HTML style rules used by every styled table.

    Args:
        left_align_positions (tuple of int): 1-based positions of the
            columns whose cells should be left-aligned. All other columns
            are centered. Defaults to ``(1,)``.

    Returns:
        list[CSSDict]: Style rules compatible with
        ``pandas.Styler.set_table_styles``.
    """
    rules: list[CSSDict] = [
        {
            "selector": "",
            "props": [
                ("margin-left", "auto"),
                ("margin-right", "auto"),
                ("width", "auto"),
                ("border-collapse", "collapse"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("background-color", "#f2f2f2"),
                ("color", "black"),
                ("font-weight", "bold"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
    ]

    for pos in left_align_positions:
        rules.append({
            "selector": f"th:nth-child({pos})",
            "props": [("text-align", "left")],
        })
        rules.append({
            "selector": f"td:nth-child({pos})",
            "props": [("text-align", "left")],
        })

    return rules


def _render_styled(styler):
    """Render a pandas ``Styler`` as centered HTML via ``IPython.display``."""
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )


def _apply_boxed_spines(ax, tick_size=9, spine_width=0.8, spine_color="black"):
    """Apply a full four-side boxed frame to ``ax``.

    Unlike the open-spine style used in the EDA toolkit, this keeps the
    top and right spines visible so the axes look like a closed box,
    matching the C18 / C16 bar-chart aesthetic.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        tick_size (float): Font size for tick labels. Defaults to 9.
        spine_width (float): Spine line width. Defaults to 0.8.
        spine_color (str): Spine color. Defaults to ``"black"``.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
        spine.set_color(spine_color)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        labelsize=tick_size,
        colors=spine_color,
        top=False,
    )


def _apply_yaxis_settings(ax, ylim=None, y_step=None, y_format=None, yticks=None):
    """Apply optional y-axis limits, locator, formatter and ticks to ``ax``."""
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_step is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_step))
    if y_format is not None:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_format))
    if yticks is not None:
        ax.set_yticks(yticks)


def _build_legend_kwargs(loc, frame, bbox, fontsize):
    """Build the keyword arguments passed to ``ax.legend``.

    Args:
        loc (str): Legend location.
        frame (bool): Whether to draw a boxed legend frame.
        bbox (tuple or None): Optional ``bbox_to_anchor``.
        fontsize (float): Legend font size.

    Returns:
        dict: Keyword arguments suitable for ``ax.legend``.
    """
    kwargs = {
        "loc": loc,
        "frameon": frame,
        "facecolor": "white" if frame else "none",
        "edgecolor": "black" if frame else "none",
        "framealpha": 1.0 if frame else 0.0,
        "fontsize": fontsize,
    }
    if bbox is not None:
        kwargs["bbox_to_anchor"] = bbox
    return kwargs


def _run_kfold(
    pipe,
    params,
    splits,
    X_train,
    y_train,
    cv,
    progress_hook=None,
):
    """Run a single K-fold evaluation of one parameter combination.

    Clones ``pipe`` for every fold, applies ``params`` on the clone, fits
    it on the training split, scores it on the validation split, and
    returns the per-fold AUCs and fit times.

    Args:
        pipe (sklearn.pipeline.Pipeline): Base pipeline to clone.
        params (dict): Parameter overrides for this combination.
        splits (list of tuple): List of ``(train_idx, val_idx)`` integer
            index arrays.
        X_train (pandas.DataFrame): Raw training features.
        y_train (pandas.Series or numpy.ndarray): Training target.
        cv (int): Number of folds, forwarded to ``progress_hook``.
        progress_hook (callable or None): Optional callback invoked as
            ``progress_hook(fold, cv, auc)`` after each fold. Defaults
            to ``None``.

    Returns:
        tuple: ``(fold_scores, fold_times)`` as lists of floats.
    """
    fold_scores = []
    fold_times = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_tr = X_train.iloc[train_idx]
        y_tr = (
            y_train.iloc[train_idx]
            if hasattr(y_train, "iloc")
            else y_train[train_idx]
        )
        X_va = X_train.iloc[val_idx]
        y_va = (
            y_train.iloc[val_idx]
            if hasattr(y_train, "iloc")
            else y_train[val_idx]
        )

        model_fold = clone(pipe).set_params(**params)

        fold_start = time.time()
        model_fold.fit(X_tr, y_tr)
        fit_elapsed = time.time() - fold_start

        y_prob = model_fold.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_prob)

        fold_scores.append(float(auc))
        fold_times.append(float(fit_elapsed))

        if progress_hook is not None:
            progress_hook(fold, cv, auc)

    return fold_scores, fold_times

def _validate_string_tuple(value, name, length=None):
    """Validate a tuple/list of strings with an optional fixed length.

    Args:
        value: Value to validate.
        name (str): Parameter name used in error messages.
        length (int or None): Required length. When ``None``, any length
            is accepted. Defaults to ``None``.

    Raises:
        TypeError: If ``value`` is not a tuple/list or does not contain
            only strings.
        ValueError: If ``length`` is set and ``value`` does not have that
            length.
    """
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"'{name}' must be a tuple or list of strings.")
    if length is not None and len(value) != length:
        raise ValueError(
            f"'{name}' must contain exactly {length} elements, "
            f"got {len(value)}."
        )
    if not all(isinstance(v, str) for v in value):
        raise TypeError(f"All elements of '{name}' must be strings.")
    
def _format_apa7_no_zero(val, decimals=3, is_p_value=False):
    """Formatea valores según APA 7 (omite el cero inicial para valores en [0, 1])."""
    if pd.isna(val) or val is None:
        return "—"

    if is_p_value:
        if val < 0.001:
            return "< .001"
        formatted = f"{val:.{decimals}f}"
        return formatted.removeprefix("0")

    formatted = f"{val:.{decimals}f}"
    if formatted.startswith("0."):
        return formatted[1:]
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    return formatted
