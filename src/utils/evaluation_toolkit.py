"""
evaluation_toolkit
==================

Statistical evaluation of binary classifiers on a shared test set.

Provides the DeLong test for comparing two correlated ROC curves,
together with an APA 7-formatted summary table. The test operates on
paired predictions, meaning both models are evaluated on the same
observations and their score vectors are compared element-wise.

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

Dependencies
------------
numpy, pandas, scipy, IPython
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from pandas.io.formats.style_render import CSSDict
from scipy.stats import norm

__version__ = "1.0.0"

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

__all__ = [
    "delong_roc_table",
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
    comparison columns are shown.

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
            strings, if ``render`` or ``times_in_minutes`` is not a
            boolean, if ``training_times`` or ``prediction_times`` is
            not ``None`` nor a length-2 tuple/list of numbers, or if any
            input array cannot be cast to a numeric array.
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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

    # Proportion of negatives ranked below each positive, ties split.
    V10 = (
        np.searchsorted(neg_sorted, pos, side="left")
        + np.searchsorted(neg_sorted, pos, side="right")
    ) / (2.0 * n)

    # Proportion of positives ranked above each negative, ties split.
    V01 = 1.0 - (
        np.searchsorted(pos_sorted, neg, side="left")
        + np.searchsorted(pos_sorted, neg, side="right")
    ) / (2.0 * m)

    auc = float(np.mean(V10))
    var_10 = float(np.var(V10, ddof=1)) / m if m > 1 else 0.0
    var_01 = float(np.var(V01, ddof=1)) / n if n > 1 else 0.0

    return auc, V10, V01, var_10 + var_01


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


def _validate_bool(value, name):
    """Validate that ``value`` is a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"'{name}' must be a boolean.")


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


def _validate_two_strings(value, name):
    """Validate a length-2 tuple/list of strings."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"'{name}' must be a tuple or list of two strings.")
    if not all(isinstance(v, str) for v in value):
        raise TypeError(f"All elements of '{name}' must be strings.")


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