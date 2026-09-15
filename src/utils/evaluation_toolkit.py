"""
evaluation_toolkit
==================

Statistical evaluation of binary classifiers on a shared test set,
inference utilities for DuckDB-backed test relations, and local
explanations for misclassified cases.

Provides the DeLong test for comparing two correlated ROC curves, a
side-by-side ROC plot, a Youden-index based threshold optimizer, a
Youden comparison table across multiple models, a classification
metrics comparison table, a persistence helper for predictions, a LIME
explainer builder, and an influence plot for local explanations,
together with APA 7-formatted summary tables and a shared figure
renderer.

The variance of each AUC and the covariance between two AUCs are
computed in O(N log N) using the U-statistic formulation of the DeLong
estimator. This avoids materializing the full m x n kernel matrix,
which makes the test practical on test sets with millions of rows.

All statistical outputs are rendered as styled HTML tables using the
same visual language applied by the other toolkits in this collection.

Main Functionalities
---------------------
Model Comparison
    delong_roc_table
    metrics_comparison_table
    plot_roc_curves
    youden_comparison_table
    youden_optimal_threshold

Inference
    predict_and_save

Local Explanations
    build_lime_explainer
    plot_lime_influence

Rendering
    render_figure

Dependencies
------------
duckdb, joblib, lime, numpy, pandas, scipy, scikit-learn, matplotlib,
IPython
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path

import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from pandas.io.formats.style_render import CSSDict
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve

__version__ = "1.4.0"

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

_DEFAULT_METRIC_TITLES = {
    "threshold": "Threshold",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC AUC",
}

__all__ = [
    "build_lime_explainer",
    "delong_roc_table",
    "metrics_comparison_table",
    "plot_lime_influence",
    "plot_roc_curves",
    "predict_and_save",
    "render_figure",
    "youden_comparison_table",
    "youden_optimal_threshold",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def delong_roc_table(
    y_true,
    y_pred1,
    y_pred2,
    model_names=("Model 1", "Model 2"),
    training_times=None,
    prediction_times=None,
    times_in_minutes=False,
    time_decimals=2,
    decimals_auc=3,
    decimals_p=3,
    column_titles=None,
    render=True,
):
    """Compare two ROC curves with the DeLong test.

    Computes the AUC of both models on the same observations, estimates
    the variance of each AUC and the covariance between them, and derives
    a two-sided z-statistic for the difference. The output table lists
    the training time, the prediction time, the AUC of the reference
    model, the AUC of the comparison model, the difference between the
    two, and the associated p-value, all formatted in APA 7 style. When
    the optional time tuples are provided, the table is extended with
    their corresponding columns. When both tuples are ``None``, only the
    comparison columns are shown. Column headers can be renamed through
    a translation dictionary.

    Args:
        y_true (array-like): Binary ground-truth labels.
        y_pred1 (array-like): Positive-class scores produced by the
            reference model.
        y_pred2 (array-like): Positive-class scores produced by the
            comparison model.
        model_names (tuple of 2 str): Display names for the reference
            and comparison models, in that order. Defaults to
            ``("Model 1", "Model 2")``.
        training_times (tuple of 2 float or None): Training wall-clock
            times for the reference and comparison models, in seconds.
            When ``None``, the ``Training Time`` column is omitted.
            Defaults to ``None``.
        prediction_times (tuple of 2 float or None): Prediction
            wall-clock times for the reference and comparison models, in
            seconds. When ``None``, the ``Prediction Time`` column is
            omitted. Defaults to ``None``.
        times_in_minutes (bool): If ``True``, the time values are
            divided by 60 and displayed with the suffix ``min``. If
            ``False``, the values are displayed with the suffix ``s``.
            Defaults to ``False``.
        time_decimals (int): Number of decimals used to format the time
            columns. Defaults to 2.
        decimals_auc (int): Number of decimals used for the AUC and the
            AUC difference. Defaults to 3.
        decimals_p (int): Number of decimals used for the p-value.
            Defaults to 3.
        column_titles (dict or None): Mapping from the default header to
            a replacement label. Keys are matched against the default
            headers ``"Model"``, ``"Training Time"``,
            ``"Prediction Time"``, ``"AUC"``, ``"ΔAUC"`` and ``"p"``.
            Unmapped columns keep their default name. When ``None``, all
            headers remain in their default form. Defaults to ``None``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        pandas.DataFrame: Numeric results with columns ``Model``,
        ``training_time``, ``prediction_time``, ``AUC``, ``ΔAUC``,
        ``z_stat`` and ``p_value``. Time columns are only present when
        their corresponding tuple is provided. The first row holds the
        reference model and leaves the difference, the z-statistic and
        the p-value as ``NaN``.

    Raises:
        TypeError: If ``model_names`` is not a length-2 tuple or list of
            strings, if ``column_titles`` is not a dict or ``None``, if
            ``render`` or ``times_in_minutes`` is not a boolean, if
            ``training_times`` or ``prediction_times`` is not ``None``
            nor a length-2 tuple/list of numbers, or if any input array
            cannot be cast to a numeric array.
        ValueError: If the two prediction vectors have different
            lengths, if they do not match the length of ``y_true``, if
            any time value is negative, or if any decimal argument is
            negative.
    """
    _validate_two_strings(model_names, "model_names")
    _validate_bool(render, "render")
    _validate_bool(times_in_minutes, "times_in_minutes")
    _validate_non_negative_int(time_decimals, "time_decimals")
    _validate_non_negative_int(decimals_auc, "decimals_auc")
    _validate_non_negative_int(decimals_p, "decimals_p")
    _validate_optional_time_pair(training_times, "training_times")
    _validate_optional_time_pair(prediction_times, "prediction_times")

    if column_titles is not None and not isinstance(column_titles, dict):
        raise TypeError("'column_titles' must be a dict or None.")

    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    if y_pred1.shape != y_pred2.shape:
        raise ValueError(
            "'y_pred1' and 'y_pred2' must have the same length."
        )
    if y_pred1.shape != y_true.shape:
        raise ValueError(
            "'y_pred1' and 'y_pred2' must match the length of 'y_true'."
        )

    auc1, V10_1, V01_1, var1 = _delong_roc_variance(y_true, y_pred1)
    auc2, V10_2, V01_2, var2 = _delong_roc_variance(y_true, y_pred2)

    m = len(V10_1)
    n = len(V01_1)

    cov_10 = np.cov(V10_1, V10_2)[0, 1] / m if m > 1 else 0.0
    cov_01 = np.cov(V01_1, V01_2)[0, 1] / n if n > 1 else 0.0
    var_diff = var1 + var2 - 2.0 * (cov_10 + cov_01)

    delta_auc = auc2 - auc1

    if var_diff <= 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = delta_auc / np.sqrt(var_diff)
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    display_values = [
        (
            model_names[0],
            _format_apa7_no_zero(auc1, decimals=decimals_auc),
            "—",
            "—",
        ),
        (
            model_names[1],
            _format_apa7_no_zero(auc2, decimals=decimals_auc),
            _format_apa7_no_zero(delta_auc, decimals=decimals_auc),
            _format_apa7_no_zero(
                p_value, decimals=decimals_p, is_p_value=True
            ),
        ),
    ]

    rows = []
    for i, (name, auc_str, delta_str, p_str) in enumerate(display_values):
        row = {"Model": name}
        if training_times is not None:
            row["Training Time"] = _format_time(
                training_times[i], times_in_minutes, time_decimals
            )
        if prediction_times is not None:
            row["Prediction Time"] = _format_time(
                prediction_times[i], times_in_minutes, time_decimals
            )
        row["AUC"] = auc_str
        row["ΔAUC"] = delta_str
        row["p"] = p_str
        rows.append(row)

    df_display = pd.DataFrame(rows)

    if column_titles is not None:
        df_display = df_display.rename(columns=column_titles)

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    df_raw = pd.DataFrame([
        {
            "Model": model_names[0],
            "AUC": auc1,
            "ΔAUC": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
        },
        {
            "Model": model_names[1],
            "AUC": auc2,
            "ΔAUC": delta_auc,
            "z_stat": z_stat,
            "p_value": p_value,
        },
    ])

    if training_times is not None:
        df_raw.insert(1, "training_time", list(training_times))
    if prediction_times is not None:
        insert_at = 2 if training_times is not None else 1
        df_raw.insert(insert_at, "prediction_time", list(prediction_times))

    return df_raw


def plot_roc_curves(
    pred1,
    pred2,
    model_names=("Model 1", "Model 2"),
    target_col="click",
    prediction_col="prediction",
    title=None,
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    colors=("black", "#555555"),
    line_styles=("-", "--"),
    line_widths=(1.4, 1.4),
    diagonal_color="#999999",
    diagonal_style=":",
    diagonal_width=0.8,
    show_auc=True,
    auc_decimals=4,
    axis_margin=0.01,
    legend_loc="lower right",
    legend_frame=False,
    legend_fontsize=None,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(9, 9, 12),
    show_grid=False,
    figsize=(5.5, 5.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the ROC curves of two binary classifiers on the same axes.

    Both inputs are resolved to a pair of vectors ``(y_true, y_score)``,
    either by reading a Parquet file or by using an in-memory DataFrame.
    The receiver operating characteristic curve of each model is drawn on
    a single panel, together with a diagonal reference line. The area
    under each curve can be shown in the legend. The figure follows the
    same visual language as the other toolkits in the collection: thin
    black spines, no gridlines by default, and vector output.

    Args:
        pred1 (str, pathlib.Path, or pandas.DataFrame): Reference source
            of predictions. When a path is given, the file is read with
            ``pandas.read_parquet``. When a DataFrame is given, it is
            used as-is.
        pred2 (str, pathlib.Path, or pandas.DataFrame): Comparison source
            of predictions, in the same format as ``pred1``.
        model_names (tuple of 2 str): Display names for the reference and
            comparison models. Defaults to ``("Model 1", "Model 2")``.
        target_col (str): Column that holds the binary ground-truth
            labels. Defaults to ``"click"``.
        prediction_col (str): Column that holds the positive-class
            scores. Defaults to ``"prediction"``.
        title (str or None): Chart title. When ``None``, no title is
            drawn. Defaults to ``None``.
        xlabel (str): X-axis label. Defaults to
            ``"False Positive Rate"``.
        ylabel (str): Y-axis label. Defaults to
            ``"True Positive Rate"``.
        colors (tuple of 2 str): Line colors for the two curves. Defaults
            to ``("black", "#555555")``.
        line_styles (tuple of 2 str): Line styles. Defaults to
            ``("-", "--")``.
        line_widths (tuple of 2 float): Line widths. Defaults to
            ``(1.4, 1.4)``.
        diagonal_color (str): Color of the diagonal reference line.
            Defaults to ``"#999999"``.
        diagonal_style (str): Line style of the diagonal reference line.
            Defaults to ``":"``.
        diagonal_width (float): Line width of the diagonal reference
            line. Defaults to 0.8.
        show_auc (bool): If ``True``, the AUC of each model is appended
            to its legend label. If ``False``, the legend shows only the
            model names. Defaults to ``True``.
        auc_decimals (int): Number of decimals used for the AUC when
            ``show_auc`` is ``True``. Defaults to 4.
        axis_margin (float): Symmetric margin added to both ends of both
            axes. ``0.01`` leaves a thin border, ``0.1`` leaves a visible
            gap around the curves. Defaults to ``0.01``.
        legend_loc (str): Legend location. Defaults to ``"lower right"``.
        legend_frame (bool): If ``True``, a boxed legend frame is drawn.
            Defaults to ``False``.
        legend_fontsize (float or None): Legend font size. When ``None``,
            ``tick_size + 1`` is used.
        label_size (float): Font size of the axis labels. Defaults to 11.
        title_size (float): Font size of the title. Defaults to 12.
        tick_size (float): Font size of the tick labels. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        show_grid (bool): If ``True``, a dotted grid is drawn behind the
            curves. Defaults to ``False``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(5.5, 5.5)``.
        render (bool): If ``True``, the figure is passed to
            :func:`render_figure` for inline rendering. Defaults to
            ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax, (auc1, auc2))`` where ``fig`` is the
        matplotlib figure, ``ax`` is the ``Axes``, and the tuple holds
        the AUC of the reference and comparison models.

    Raises:
        TypeError: If either source is not a path or a DataFrame, if
            ``model_names`` is not a length-2 sequence of strings, if
            ``show_auc`` is not a boolean, or if ``render`` is not a
            boolean.
        ValueError: If the input DataFrame is missing a required column,
            if the two target vectors differ, if ``auc_decimals`` is
            negative, or if ``axis_margin`` is negative.
    """
    _validate_two_strings(model_names, "model_names")
    _validate_bool(render, "render")
    _validate_bool(show_auc, "show_auc")
    _validate_non_negative_int(auc_decimals, "auc_decimals")
    _validate_figsize(figsize)
    _validate_pads(pads)

    if not isinstance(axis_margin, (int, float)) or isinstance(axis_margin, bool):
        raise TypeError("'axis_margin' must be a number.")
    if axis_margin < 0:
        raise ValueError("'axis_margin' must be non-negative.")

    y1, s1 = _resolve_predictions(pred1, target_col, prediction_col)
    y2, s2 = _resolve_predictions(pred2, target_col, prediction_col)

    if y1.shape != y2.shape or not np.array_equal(y1, y2):
        raise ValueError(
            "Both prediction sources must share the same target vector."
        )

    fpr1, tpr1, _ = roc_curve(y1, s1)
    fpr2, tpr2, _ = roc_curve(y2, s2)

    auc1 = float(auc(fpr1, tpr1))
    auc2 = float(auc(fpr2, tpr2))

    pad_x, pad_y, pad_title = pads

    if legend_fontsize is None:
        legend_fontsize = tick_size + 1

    if show_auc:
        label1 = f"{model_names[0]} (AUC = {auc1:.{auc_decimals}f})"
        label2 = f"{model_names[1]} (AUC = {auc2:.{auc_decimals}f})"
    else:
        label1 = model_names[0]
        label2 = model_names[1]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        [0, 1], [0, 1],
        color=diagonal_color,
        linestyle=diagonal_style,
        linewidth=diagonal_width,
        zorder=1,
    )

    ax.plot(
        fpr1, tpr1,
        color=colors[0],
        linestyle=line_styles[0],
        linewidth=line_widths[0],
        label=label1,
        zorder=2,
    )
    ax.plot(
        fpr2, tpr2,
        color=colors[1],
        linestyle=line_styles[1],
        linewidth=line_widths[1],
        label=label2,
        zorder=3,
    )

    ax.set_xlim(-axis_margin, 1 + axis_margin)
    ax.set_ylim(-axis_margin, 1 + axis_margin)
    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)
    ax.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)

    _style_axes(ax, tick_size)

    if show_grid:
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    ax.legend(
        loc=legend_loc,
        frameon=legend_frame,
        facecolor="white" if legend_frame else "none",
        edgecolor="black" if legend_frame else "none",
        framealpha=1.0 if legend_frame else 0.0,
        fontsize=legend_fontsize,
    )

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax, (auc1, auc2)


def youden_optimal_threshold(
    y_true,
    y_score,
    model_name="Model",
    column_titles=None,
    decimals=4,
    render=True,
):
    """Find the decision threshold that maximizes the Youden index.

    The Youden index is defined as ``J = Sensitivity + Specificity - 1``,
    which is equivalent to ``J = TPR - FPR``. The function sweeps the
    thresholds returned by ``sklearn.metrics.roc_curve``, selects the
    one that maximizes ``J``, and reports the threshold together with the
    sensitivity, specificity and index at that point.

    Args:
        y_true (array-like): Binary ground-truth labels.
        y_score (array-like): Positive-class scores.
        model_name (str): Display name shown in the first column of the
            output table. Defaults to ``"Model"``.
        column_titles (dict or None): Mapping from default header to a
            replacement label. Keys are matched against ``"Model"``,
            ``"Threshold"``, ``"Sensitivity"``, ``"Specificity"`` and
            ``"Youden Index"``. Unmapped columns keep their default
            header. Defaults to ``None``.
        decimals (int): Number of decimals used for all formatted values.
            Defaults to 4.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        dict: Numeric results with keys ``model``, ``threshold``,
        ``sensitivity``, ``specificity``, ``youden_index``, ``n_pos``
        and ``n_neg``.

    Raises:
        TypeError: If ``model_name`` is not a string, if
            ``column_titles`` is not a dict or ``None``, or if
            ``render`` is not a boolean.
        ValueError: If ``decimals`` is negative, if the two arrays have
            different lengths, or if either class is absent from
            ``y_true``.
    """
    _validate_string(model_name, "model_name")
    _validate_bool(render, "render")
    _validate_non_negative_int(decimals, "decimals")

    if column_titles is not None and not isinstance(column_titles, dict):
        raise TypeError("'column_titles' must be a dict or None.")

    scores = _youden_scores(y_true, y_score)

    row = {
        "Model": model_name,
        "Threshold": _format_apa7_no_zero(
            scores["threshold"], decimals=decimals
        ),
        "Sensitivity": _format_apa7_no_zero(
            scores["sensitivity"], decimals=decimals
        ),
        "Specificity": _format_apa7_no_zero(
            scores["specificity"], decimals=decimals
        ),
        "Youden Index": _format_apa7_no_zero(
            scores["youden_index"], decimals=decimals
        ),
    }

    df_display = pd.DataFrame([row])

    if column_titles is not None:
        df_display = df_display.rename(columns=column_titles)

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return {
        "model": model_name,
        "threshold": scores["threshold"],
        "sensitivity": scores["sensitivity"],
        "specificity": scores["specificity"],
        "youden_index": scores["youden_index"],
        "n_pos": scores["n_pos"],
        "n_neg": scores["n_neg"],
    }


def youden_comparison_table(
    sources,
    model_names=None,
    target_col="click",
    prediction_col="prediction",
    column_titles=None,
    decimals=4,
    render=True,
):
    """Build a comparison table of Youden thresholds across models.

    Computes the Youden optimal threshold for each prediction source and
    stacks the results into a table where each row corresponds to one
    model. The columns report the threshold, the sensitivity, the
    specificity and the Youden index at that threshold. The output is
    rendered with the same visual language as the other tables in the
    toolkit.

    Args:
        sources (list or tuple): Prediction sources, one per model. Each
            element is either a path to a Parquet file or a pandas
            DataFrame with the target and prediction columns.
        model_names (list or tuple of str or None): Display names for the
            models, in the same order as ``sources``. When ``None``,
            generic names ``("Model 1", "Model 2", ...)`` are used.
            Defaults to ``None``.
        target_col (str): Name of the column with the binary ground-truth
            labels. Defaults to ``"click"``.
        prediction_col (str): Name of the column with the positive-class
            scores. Defaults to ``"prediction"``.
        column_titles (dict or None): Mapping from default header to a
            replacement label. Keys are matched against ``"Model"``,
            ``"Threshold"``, ``"Sensitivity"``, ``"Specificity"`` and
            ``"Youden Index"``. Unmapped columns keep their default
            header. Defaults to ``None``.
        decimals (int): Number of decimals used for every formatted value.
            Defaults to 4.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The formatted comparison table with one row per
        model and the columns ``Model``, ``Threshold``, ``Sensitivity``,
        ``Specificity`` and ``Youden Index``.

    Raises:
        TypeError: If ``sources`` or ``model_names`` is not a list or
            tuple, if ``column_titles`` is not a dict or ``None``, if
            ``render`` is not a boolean, or if any element of
            ``model_names`` is not a string.
        ValueError: If ``model_names`` does not match the number of
            sources, if ``decimals`` is negative, or if any source lacks
            both classes in its target vector.
    """
    _validate_sequence(sources, "sources")
    _validate_bool(render, "render")
    _validate_non_negative_int(decimals, "decimals")

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(sources))]
    else:
        _validate_sequence(model_names, "model_names")
        if len(model_names) != len(sources):
            raise ValueError(
                "'model_names' must have one name per source."
            )
        if not all(isinstance(n, str) for n in model_names):
            raise TypeError("All elements of 'model_names' must be strings.")

    if column_titles is not None and not isinstance(column_titles, dict):
        raise TypeError("'column_titles' must be a dict or None.")

    rows = []
    for name, source in zip(model_names, sources):
        y_true, y_score = _resolve_predictions(
            source, target_col, prediction_col
        )
        scores = _youden_scores(y_true, y_score)

        rows.append({
            "Model": name,
            "Threshold": _format_apa7_no_zero(
                scores["threshold"], decimals=decimals
            ),
            "Sensitivity": _format_apa7_no_zero(
                scores["sensitivity"], decimals=decimals
            ),
            "Specificity": _format_apa7_no_zero(
                scores["specificity"], decimals=decimals
            ),
            "Youden Index": _format_apa7_no_zero(
                scores["youden_index"], decimals=decimals
            ),
        })

    df_display = pd.DataFrame(rows)

    if column_titles is not None:
        df_display = df_display.rename(columns=column_titles)

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return df_display


def metrics_comparison_table(
    sources,
    thresholds,
    model_names=None,
    metric_titles=None,
    metric_col_title="Metric",
    decimals=4,
    include_threshold=True,
    render=True,
):
    """Build a comparison table of classification metrics across models.

    Computes accuracy, precision, recall, F1-score and ROC AUC for each
    model passed in ``sources``, using the corresponding decision
    threshold from ``thresholds``. The result is a table whose rows are
    the metrics and whose columns are the models, formatted in APA 7
    style. Accuracy, precision, recall and F1-score depend on the
    decision threshold, while ROC AUC is invariant to it.

    Args:
        sources (list or tuple): Prediction sources, one per model. Each
            element is either a path to a Parquet file or a pandas
            DataFrame with the target and prediction columns.
        thresholds (list or tuple of float): Decision threshold for each
            model, in the same order as ``sources``. Each value must lie
            in ``(0, 1)``.
        model_names (list or tuple of str or None): Display names for the
            models, in the same order as ``sources``. When ``None``,
            generic names ``("Model 1", "Model 2", ...)`` are used.
            Defaults to ``None``.
        metric_titles (dict or None): Overrides for the metric labels.
            Supported keys are ``"threshold"``, ``"accuracy"``,
            ``"precision"``, ``"recall"``, ``"f1"`` and ``"roc_auc"``.
            When ``None``, English defaults are used. Defaults to
            ``None``.
        metric_col_title (str): Header for the first column, which lists
            the metric names. Defaults to ``"Metric"``.
        decimals (int): Number of decimals used for every formatted value.
            Defaults to 4.
        include_threshold (bool): If ``True``, the threshold of each model
            is shown as the first row of the table. Defaults to ``True``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The formatted comparison table with the metric
        names as rows and the model names as columns.

    Raises:
        TypeError: If ``sources``, ``thresholds`` or ``model_names`` is
            not a list or tuple, if ``metric_titles`` is not a dict or
            ``None``, if ``render`` or ``include_threshold`` is not a
            boolean, or if ``decimals`` is not a non-negative integer.
        ValueError: If ``sources`` and ``thresholds`` have different
            lengths, if ``model_names`` does not match the number of
            sources, if any threshold is outside ``(0, 1)``, or if any
            source lacks both classes in its target vector.
    """
    _validate_sequence(sources, "sources")
    _validate_sequence(thresholds, "thresholds")
    _validate_bool(render, "render")
    _validate_bool(include_threshold, "include_threshold")
    _validate_non_negative_int(decimals, "decimals")
    _validate_string(metric_col_title, "metric_col_title")

    if len(sources) != len(thresholds):
        raise ValueError(
            "'sources' and 'thresholds' must have the same length."
        )

    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(sources))]
    else:
        _validate_sequence(model_names, "model_names")
        if len(model_names) != len(sources):
            raise ValueError(
                "'model_names' must have one name per source."
            )
        if not all(isinstance(n, str) for n in model_names):
            raise TypeError("All elements of 'model_names' must be strings.")

    if metric_titles is not None and not isinstance(metric_titles, dict):
        raise TypeError("'metric_titles' must be a dict or None.")

    titles = dict(_DEFAULT_METRIC_TITLES)
    if metric_titles is not None:
        unknown = set(metric_titles) - set(titles)
        if unknown:
            raise ValueError(
                f"Unsupported metric title keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(titles)}."
            )
        titles.update(metric_titles)

    metrics_per_model = [
        _classification_metrics(source, threshold=thr)
        for source, thr in zip(sources, thresholds)
    ]

    metric_order = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []

    if include_threshold:
        row = {metric_col_title: titles["threshold"]}
        for name, thr in zip(model_names, thresholds):
            row[name] = _format_apa7_no_zero(thr, decimals=decimals)
        rows.append(row)

    for key in metric_order:
        row = {metric_col_title: titles[key]}
        for name, m in zip(model_names, metrics_per_model):
            row[name] = _format_apa7_no_zero(m[key], decimals=decimals)
        rows.append(row)

    df_display = pd.DataFrame(rows)

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return df_display


def predict_and_save(
    con,
    model_path,
    relation="test",
    target="click",
    output_dir=None,
    n_rows=200_000,
    threshold=0.5,
    extract=True,
    seed=42,
):
    """Run a serialized model over a DuckDB relation and persist predictions.

    Loads the pipeline, draws a reproducible sample from ``relation``,
    computes the positive-class probability for each row, measures the
    inference time, and writes the true label and the predicted
    probability to a Parquet file. The output file is always named after
    the model with the suffix ``_predictions.parquet``. When
    ``extract=True``, one false positive and one false negative are
    returned in memory along with their raw feature values, so they can
    be fed directly to an explainer.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        model_path (str or pathlib.Path): Path to the serialized
            pipeline in joblib format.
        relation (str): Name of the DuckDB view or table with the test
            data. Defaults to ``"test"``.
        target (str): Name of the binary target column. Defaults to
            ``"click"``.
        output_dir (str, pathlib.Path, or None): Directory where the
            Parquet file is written. The file is named after the model
            stem with the suffix ``_predictions.parquet``. When
            ``None``, the file is written next to the model. Defaults to
            ``None``.
        n_rows (int): Number of rows to draw from ``relation`` for the
            prediction set. When the value exceeds the total number of
            rows in the view, all rows are returned. Defaults to
            200,000.
        threshold (float): Decision threshold used to classify a row as
            a positive prediction. Defaults to 0.5.
        extract (bool): If ``True``, one false positive and one false
            negative are located and returned in memory. If ``False``,
            the extraction step is skipped and ``failing_cases`` is
            ``None``. Defaults to ``True``.
        seed (int): Seed for the reservoir sampler and for the failing
            case selection. Defaults to 42.

    Returns:
        dict: Dictionary with keys ``prediction_time`` (seconds),
        ``output_path`` (pathlib.Path), ``failing_cases``
        (pandas.DataFrame with two rows and the raw features, or
        ``None`` when ``extract=False``), ``n_failures``, ``n_fp`` and
        ``n_fn``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        TypeError: If ``extract`` is not a boolean.
        ValueError: If ``n_rows`` is not a positive integer, if
            ``threshold`` is not in ``(0, 1)``, or if the sample is
            empty.
    """

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    _validate_positive_int(n_rows, "n_rows")
    _validate_bool(extract, "extract")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) \
            or not (0 < threshold < 1):
        raise ValueError("'threshold' must be a number in (0, 1).")

    model = joblib.load(model_path)

    df = con.sql(f"""
        SELECT * FROM {relation}
        USING SAMPLE {n_rows} ROWS (reservoir, {seed})
    """).df()

    if df.empty:
        raise ValueError(f"The sample from '{relation}' returned no rows.")

    X = df.drop(columns=[target])
    y = df[target].astype("int8").to_numpy()

    start = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    prediction_time = time.perf_counter() - start

    target_dir = Path(output_dir) if output_dir is not None else model_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{model_path.stem}_predictions.parquet"

    pd.DataFrame({
        target: y,
        "prediction": proba.astype("float32"),
    }).to_parquet(output_path, index=False)

    y_pred = (proba >= threshold).astype("int8")

    n_failures = int((y != y_pred).sum())
    n_fp = int(((y_pred == 1) & (y == 0)).sum())
    n_fn = int(((y_pred == 0) & (y == 1)).sum())

    failing_cases = None
    if extract:
        df_full = df.copy()
        df_full["_pred"] = y_pred

        fp_pool = df_full[(df_full["_pred"] == 1) & (df_full[target] == 0)]
        fn_pool = df_full[(df_full["_pred"] == 0) & (df_full[target] == 1)]

        picks = []
        if not fp_pool.empty:
            picks.append(fp_pool.sample(1, random_state=seed))
        if not fn_pool.empty:
            picks.append(fn_pool.sample(1, random_state=seed + 1))

        if picks:
            failing_cases = pd.concat(picks, ignore_index=True)
        else:
            failing_cases = pd.DataFrame(columns=df_full.columns)

    return {
        "prediction_time": prediction_time,
        "output_path": output_path,
        "failing_cases": failing_cases,
        "n_failures": n_failures,
        "n_fp": n_fp,
        "n_fn": n_fn,
    }


def build_lime_explainer(
    df_features,
    categorical_cols,
    numeric_cols,
    random_state=42,
):
    """Build a LIME explainer for a mixed categorical/numeric feature set.

    Categorical columns are encoded as integer codes before being handed
    to LIME, and the original category values are stored so that any
    perturbed array can be decoded back to the DataFrame format the
    pipeline expects. Numeric columns are preserved as floats with no
    intermediate string conversion.

    Args:
        df_features (pandas.DataFrame): Reference sample of raw feature
            values.
        categorical_cols (list of str): Names of the categorical columns.
        numeric_cols (list of str): Names of the numeric columns.
        random_state (int): Seed for the LIME sampler. Defaults to 42.

    Returns:
        tuple: ``(explainer, decode, columns, encoders)`` where
        ``explainer`` is the fitted ``LimeTabularExplainer``, ``decode``
        is a function mapping a numpy array back to a DataFrame with the
        original column names and dtypes, ``columns`` is the feature name
        list, and ``encoders`` is a mapping from column name to the
        ordered list of observed categories.

    Raises:
        KeyError: If any name in ``categorical_cols`` or ``numeric_cols``
            is not a column of ``df_features``.
    """

    columns = list(df_features.columns)

    missing = [
        c for c in (categorical_cols + numeric_cols) if c not in columns
    ]
    if missing:
        raise KeyError(f"Columns not found in 'df_features': {missing}")

    original_dtypes = {col: df_features[col].dtype for col in columns}

    encoders = {}
    X_encoded = df_features.copy()

    for col in categorical_cols:
        cat = X_encoded[col].astype("category")
        encoders[col] = list(cat.cat.categories)
        X_encoded[col] = cat.cat.codes.astype(float)

    X_array = np.empty(X_encoded.shape, dtype=float)
    for i, col in enumerate(X_encoded.columns):
        X_array[:, i] = pd.to_numeric(
            X_encoded[col], errors="raise"
        ).to_numpy(dtype=float)

    cat_indices = [i for i, c in enumerate(columns) if c in encoders]
    cat_names = {i: encoders[columns[i]] for i in cat_indices}

    def decode(arr):
        """Map an encoded array back to a DataFrame matching the input.

        Categorical columns are restored to their original values and
        dtypes. Numeric columns keep their float values without any
        string conversion.

        Args:
            arr (numpy.ndarray): Array of shape ``(n, len(columns))``.

        Returns:
            pandas.DataFrame: DataFrame with the original column names
            and dtypes.
        """
        df = pd.DataFrame(arr, columns=columns)

        for col, cats in encoders.items():
            codes = df[col].round().astype(int).clip(0, len(cats) - 1)
            df[col] = pd.Series([cats[c] for c in codes], index=df.index)
            try:
                df[col] = df[col].astype(original_dtypes[col])
            except (ValueError, TypeError):
                pass

        for col in numeric_cols:
            try:
                df[col] = df[col].astype(original_dtypes[col])
            except (ValueError, TypeError):
                pass

        return df[columns]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_array,
        feature_names=columns,
        categorical_features=cat_indices,
        categorical_names=cat_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    return explainer, decode, columns, encoders


def plot_lime_influence(
    explanation,
    label=1,
    title=None,
    xlabel="Contribution to prediction",
    ylabel=None,
    positive_color="#1a3d6b",
    negative_color="#a32e2e",
    positive_label="For",
    negative_label="Against",
    bar_height=0.7,
    bar_edge_color="black",
    bar_edge_width=0.8,
    show_values=True,
    value_size=9,
    value_format="{:+.3f}",
    value_offset_ratio=0.02,
    show_legend=True,
    legend_loc="lower right",
    legend_frame=False,
    legend_fontsize=None,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(9, 9, 12),
    show_grid=False,
    xlim_padding=0.20,
    figsize=(8, 5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the per-feature contributions of a LIME explanation.

    Draws a horizontal bar chart of the weights returned by
    ``explanation.as_list(label)``. Positive weights push the prediction
    toward the target class and are drawn in one color; negative weights
    push the prediction away and are drawn in another. The bars are
    sorted by absolute weight, with the most influential feature at the
    top of the panel. The visual language matches the rest of the
    toolkit: thin black edges, no gridlines by default, vector output.

    Args:
        explanation (lime.explanation.Explanation): Fitted LIME
            explanation. Only the ``as_list(label)`` method is used, so
            any object exposing that method is accepted.
        label (int): Class index passed to ``explanation.as_list``. For
            binary classifiers, ``1`` corresponds to the positive class.
            Defaults to ``1``.
        title (str or None): Chart title. When ``None``, no title is
            drawn. Defaults to ``None``.
        xlabel (str): X-axis label. Defaults to
            ``"Contribution to prediction"``.
        ylabel (str or None): Y-axis label. When ``None``, no label is
            drawn. Defaults to ``None``.
        positive_color (str): Fill color for the bars with positive
            weight. Defaults to ``"#1a3d6b"``.
        negative_color (str): Fill color for the bars with negative
            weight. Defaults to ``"#a32e2e"``.
        positive_label (str): Legend label for positive contributions.
            Defaults to ``"For"``.
        negative_label (str): Legend label for negative contributions.
            Defaults to ``"Against"``.
        bar_height (float): Relative height of the bars. Defaults to 0.7.
        bar_edge_color (str): Edge color of the bars. Defaults to
            ``"black"``.
        bar_edge_width (float): Edge line width of the bars. Defaults to
            0.8.
        show_values (bool): If ``True``, the weight of each bar is
            annotated next to its tip. Defaults to ``True``.
        value_size (float): Font size of the annotation. Defaults to 9.
        value_format (str): Format string applied to each weight. Defaults
            to ``"{:+.3f}"``, which shows the sign and three decimals.
        value_offset_ratio (float): Offset of the annotation relative to
            the largest absolute weight, as a fraction. Defaults to 0.02.
        show_legend (bool): If ``True``, a legend with the two
            contribution directions is drawn. Defaults to ``True``.
        legend_loc (str): Legend location. Defaults to ``"lower right"``.
        legend_frame (bool): If ``True``, a boxed legend frame is drawn.
            Defaults to ``False``.
        legend_fontsize (float or None): Legend font size. When ``None``,
            ``tick_size`` is used.
        label_size (float): Font size of the axis labels. Defaults to 11.
        title_size (float): Font size of the title. Defaults to 12.
        tick_size (float): Font size of the tick labels. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        show_grid (bool): If ``True``, a dotted grid is drawn along the
            x-axis. Defaults to ``False``.
        xlim_padding (float): Fraction of the data range added on each
            side of the x-axis. Defaults to 0.20.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 5)``.
        render (bool): If ``True``, the figure is passed to
            :func:`render_figure` for inline rendering. Defaults to
            ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax)`` where ``fig`` is the matplotlib figure and
        ``ax`` is the ``Axes``.

    Raises:
        TypeError: If ``explanation`` does not expose ``as_list``, if any
            string argument is not a string, or if ``render`` is not a
            boolean.
        ValueError: If the explanation returns no weights, or if
            ``xlim_padding`` is negative.
    """
    _validate_bool(render, "render")
    _validate_bool(show_values, "show_values")
    _validate_bool(show_legend, "show_legend")
    _validate_bool(legend_frame, "legend_frame")
    _validate_bool(show_grid, "show_grid")
    _validate_figsize(figsize)
    _validate_pads(pads)
    _validate_string(xlabel, "xlabel")
    _validate_string(positive_color, "positive_color")
    _validate_string(negative_color, "negative_color")
    _validate_string(bar_edge_color, "bar_edge_color")

    if not isinstance(xlim_padding, (int, float)) or isinstance(xlim_padding, bool):
        raise TypeError("'xlim_padding' must be a number.")
    if xlim_padding < 0:
        raise ValueError("'xlim_padding' must be non-negative.")

    if not hasattr(explanation, "as_list"):
        raise TypeError(
            "'explanation' must expose an 'as_list' method "
            "(a LIME Explanation object is expected)."
        )

    pairs = explanation.as_list(label=label)
    if not pairs:
        raise ValueError(
            "The explanation returned no weights for the given label."
        )

    features = [str(p[0]) for p in pairs]
    values = [float(p[1]) for p in pairs]

    order = np.argsort(np.abs(values))[::-1]
    features = [features[i] for i in order]
    values = [values[i] for i in order]

    features = features[::-1]
    values = values[::-1]

    n_features = len(features)
    y_pos = np.arange(n_features)

    colors = [
        positive_color if v >= 0 else negative_color for v in values
    ]

    pad_x, pad_y, pad_title = pads

    if legend_fontsize is None:
        legend_fontsize = tick_size

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        y_pos,
        values,
        height=bar_height,
        color=colors,
        edgecolor=bar_edge_color,
        linewidth=bar_edge_width,
    )

    ax.axvline(0, color="black", linewidth=0.8, zorder=3)

    max_abs = max((abs(v) for v in values), default=1.0)
    offset = value_offset_ratio * max_abs

    if show_values:
        for y, v in zip(y_pos, values):
            ha = "left" if v >= 0 else "right"
            x_pos = v + (offset if v >= 0 else -offset)
            ax.text(
                x_pos,
                y,
                value_format.format(v),
                ha=ha,
                va="center",
                fontsize=value_size,
                color="black",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=tick_size)

    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)

    x_min = min(min(values), 0.0)
    x_max = max(max(values), 0.0)
    span = x_max - x_min
    if span == 0:
        span = 1.0

    ax.set_xlim(
        x_min - span * xlim_padding,
        x_max + span * xlim_padding,
    )

    _style_axes(ax, tick_size)

    if show_grid:
        ax.xaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    if show_legend:
        from matplotlib.patches import Patch

        legend_handles = [
            Patch(
                facecolor=positive_color,
                edgecolor=bar_edge_color,
                label=positive_label,
            ),
            Patch(
                facecolor=negative_color,
                edgecolor=bar_edge_color,
                label=negative_label,
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            frameon=legend_frame,
            facecolor="white" if legend_frame else "none",
            edgecolor="black" if legend_frame else "none",
            framealpha=1.0 if legend_frame else 0.0,
            fontsize=legend_fontsize,
        )

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax


def render_figure(fig, fmt="svg", dpi=100):
    """Render a matplotlib figure inline as centered HTML.

    Args:
        fig (matplotlib.figure.Figure): Figure to render. The figure is
            closed after encoding, so it is not displayed again by
            matplotlib.
        fmt ({"svg", "png"}): Output format. ``"svg"`` embeds a vector
            image; ``"png"`` rasterizes the figure. Defaults to
            ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Raises:
        ValueError: If ``fmt`` is not one of ``{"svg", "png"}``.
    """
    if fmt not in ("svg", "png"):
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classification_metrics(
    source,
    threshold=0.5,
    target_col="click",
    prediction_col="prediction",
):
    """Compute standard binary classification metrics from predictions.

    The metrics are computed from a prediction source that contains the
    ground-truth labels and the positive-class scores. Accuracy, precision,
    recall and F1-score are derived from the confusion matrix using the
    given decision threshold. The ROC AUC is computed from the scores
    directly and does not depend on the threshold.

    Args:
        source (str, pathlib.Path, or pandas.DataFrame): Prediction
            source. A path is read with ``pandas.read_parquet``. A
            DataFrame is used as-is.
        threshold (float): Decision threshold used to binarize the
            positive-class scores. Must be in ``(0, 1)``. Defaults to
            ``0.5``.
        target_col (str): Name of the column with the binary ground-truth
            labels. Defaults to ``"click"``.
        prediction_col (str): Name of the column with the positive-class
            scores. Defaults to ``"prediction"``.

    Returns:
        dict: Dictionary with keys ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``roc_auc``, ``threshold``, ``n``,
        ``n_pos``, ``n_neg``, ``tp``, ``fp``, ``fn`` and ``tn``.

    Raises:
        TypeError: If ``source`` is not a path, a ``Path``, or a
            DataFrame, if ``target_col`` or ``prediction_col`` is not a
            string, or if ``threshold`` is not a number.
        ValueError: If the required columns are missing, if ``threshold``
            is not in ``(0, 1)``, or if ``y_true`` contains only one
            class.
    """
    y_true, y_score = _resolve_predictions(
        source, target_col, prediction_col
    )

    _validate_string(target_col, "target_col")
    _validate_string(prediction_col, "prediction_col")

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError("'threshold' must be a number.")
    if not (0 < threshold < 1):
        raise ValueError("'threshold' must be a number in (0, 1).")

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "'y_true' must contain both classes to compute the metrics."
        )

    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "n": int(tp + tn + fp + fn),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _delong_roc_variance(ground_truth, predictions):
    """Compute the DeLong U-statistics for one ROC curve.

    Uses the rank-based formulation that avoids materializing the full
    m x n kernel matrix, so the cost is O(N log N) instead of O(m * n).

    Args:
        ground_truth (array-like): Binary ground-truth labels.
        predictions (array-like): Positive-class scores.

    Returns:
        tuple: ``(auc, V10, V01, variance)`` where ``V10`` and ``V01``
        are the per-observation placement values and ``variance`` is the
        estimated variance of the AUC.
    """
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)

    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]

    m = len(pos)
    n = len(neg)

    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    V10 = (
        np.searchsorted(neg_sorted, pos, side="left")
        + np.searchsorted(neg_sorted, pos, side="right")
    ) / (2.0 * n)

    V01 = 1.0 - (
        np.searchsorted(pos_sorted, neg, side="left")
        + np.searchsorted(pos_sorted, neg, side="right")
    ) / (2.0 * m)

    auc_value = float(np.mean(V10))
    var_10 = float(np.var(V10, ddof=1)) / m if m > 1 else 0.0
    var_01 = float(np.var(V01, ddof=1)) / n if n > 1 else 0.0

    return auc_value, V10, V01, var_10 + var_01


def _format_apa7_no_zero(val, decimals=3, is_p_value=False):
    """Format a value in APA 7 style, omitting the leading zero.

    APA 7 recommends dropping the leading zero for statistics bounded in
    the interval ``[-1, 1]``. When ``is_p_value`` is ``True``, the
    ``< .001`` convention is applied on top of the same rule.

    Args:
        val (float): Value to format.
        decimals (int): Number of decimals. Defaults to 3.
        is_p_value (bool): If ``True``, applies the ``< .001`` rule.
            Defaults to ``False``.

    Returns:
        str: The formatted value, or ``"—"`` when ``val`` is NaN.
    """
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


def _format_time(seconds, in_minutes=False, decimals=2):
    """Format a time value in seconds or minutes with its unit suffix.

    Args:
        seconds (float): Time value in seconds.
        in_minutes (bool): If ``True``, the value is divided by 60 and
            the suffix ``min`` is appended. Otherwise the value is kept
            in seconds and the suffix ``s`` is appended. Defaults to
            ``False``.
        decimals (int): Number of decimals. Defaults to 2.

    Returns:
        str: The formatted time, or ``"—"`` when ``seconds`` is NaN.
    """
    if pd.isna(seconds):
        return "—"
    if in_minutes:
        return f"{seconds / 60:.{decimals}f} min"
    return f"{seconds:.{decimals}f} s"


def _resolve_predictions(source, target_col, prediction_col):
    """Resolve a prediction source to a pair of numpy arrays.

    Args:
        source (str, pathlib.Path, or pandas.DataFrame): Prediction
            source. A path is read with ``pandas.read_parquet``. A
            DataFrame is used directly.
        target_col (str): Name of the target column.
        prediction_col (str): Name of the prediction column.

    Returns:
        tuple: ``(y_true, y_score)`` as numpy arrays.

    Raises:
        TypeError: If ``source`` is not a path or a DataFrame.
        ValueError: If the required columns are missing.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_parquet(source)
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        raise TypeError(
            "'source' must be a path, a Path, or a pandas DataFrame."
        )

    for col in (target_col, prediction_col):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in the prediction source. "
                f"Available columns: {list(df.columns)}"
            )

    return (
        df[target_col].to_numpy(),
        df[prediction_col].to_numpy(dtype=float),
    )


def _style_spines(ax, color="black", linewidth=0.8):
    """Apply the toolkit's spine style to every spine of ``ax``."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def _style_ticks(ax, tick_size):
    """Apply the toolkit's tick style to the x and y axes of ``ax``."""
    ax.tick_params(
        axis="x", which="both", bottom=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )
    ax.tick_params(
        axis="y", which="both", left=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )


def _style_axes(ax, tick_size):
    """Apply the toolkit's standard axis style (spines + ticks)."""
    _style_spines(ax)
    _style_ticks(ax, tick_size)


def _validate_bool(value, name):
    """Validate that ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be a boolean.")


def _validate_figsize(figsize):
    """Validate a ``(width, height)`` figure-size tuple."""
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise TypeError("'figsize' must be a tuple/list of length 2.")


def _validate_non_negative_int(value, name):
    """Validate that ``value`` is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative integer.")


def _validate_optional_time_pair(value, name):
    """Validate an optional length-2 sequence of non-negative numbers.

    Args:
        value: Value to validate. Either ``None`` or a tuple/list of two
            non-negative real numbers.
        name (str): Parameter name used in error messages.

    Raises:
        TypeError: If ``value`` is not ``None`` and not a length-2
            tuple/list.
        ValueError: If any element is not a non-negative number.
    """
    if value is None:
        return
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(
            f"'{name}' must be None or a tuple/list of two numbers."
        )
    for v in value:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ValueError(
                f"All elements of '{name}' must be non-negative numbers."
            )


def _validate_pads(pads):
    """Validate the ``(pad_x, pad_y, pad_title)`` padding tuple."""
    if not isinstance(pads, (tuple, list)):
        raise TypeError("'pads' must be a tuple or list of three numbers.")
    if len(pads) != 3:
        raise ValueError(
            f"'pads' must contain exactly 3 elements (pad_x, pad_y, "
            f"pad_title), got {len(pads)}."
        )


def _validate_positive_int(value, name):
    """Validate that ``value`` is a strictly positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{name}' must be a positive integer.")


def _validate_sequence(value, name):
    """Validate that ``value`` is a non-empty tuple or list."""
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"'{name}' must be a tuple or list.")
    if len(value) == 0:
        raise ValueError(f"'{name}' must contain at least one element.")


def _validate_string(value, name):
    """Validate that ``value`` is a string."""
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a string.")


def _validate_two_strings(value, name):
    """Validate a length-2 tuple/list of strings."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"'{name}' must be a tuple or list of two strings.")
    if not all(isinstance(v, str) for v in value):
        raise TypeError(f"All elements of '{name}' must be strings.")


def _youden_scores(y_true, y_score):
    """Compute the Youden optimal threshold and its associated metrics.

    Args:
        y_true (array-like): Binary ground-truth labels.
        y_score (array-like): Positive-class scores.

    Returns:
        dict: Dictionary with keys ``threshold``, ``sensitivity``,
        ``specificity``, ``youden_index``, ``n_pos`` and ``n_neg``.

    Raises:
        ValueError: If the two arrays have different lengths, or if
            either class is absent from ``y_true``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            "'y_true' and 'y_score' must have the same length."
        )

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "'y_true' must contain both classes to compute the Youden index."
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))

    return {
        "threshold": float(thresholds[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1.0 - fpr[best_idx]),
        "youden_index": float(youden[best_idx]),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


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