"""
eda_toolkit
===========

Exploratory Data Analysis (EDA) Toolkit.

A curated collection of visualization and statistical analysis utilities
for structured and reproducible data exploration in Jupyter notebooks.

All functions generate publication-ready figures encoded as inline HTML/Base64
and styled statistical tables using a consistent visual language.

Visual Language
---------------
- Primary Blue    : #1f77b4  (baseline / negative class)
- Accent Red      : #d62728  (highlight / positive class)
- Secondary Green : #2ca02c
- Secondary Orange: #ff7f0e
- White background with minimal gridlines (seaborn "white" style)
- Bold axis labels, tight layout, 100–250 dpi output

Main Functionalities
--------------------
Univariate
    plot_categorical_distribution
    plot_numeric_diagnostics
    plot_descriptive_summary

Missing Data
    plot_missingness_map

Bivariate / Association
    plot_categorical_association
    plot_numeric_distribution
    analyze_chi2_association
    analyze_continuous_association

Multivariate
    plot_spearman_heatmap
    plot_mrmr_importance
    plot_pca_rank_2d
    plot_pca_3d_interactive

Dependencies
------------
matplotlib, seaborn, scipy, statsmodels, pandas, numpy,
mrmr-selection, scikit-learn, plotly
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

__author__ = (
    "Paula Andrea Gómez Vargas <apaulag@uninorte.edu.co>, "
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import io
import base64

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import warnings
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import norm, probplot, mannwhitneyu, levene, spearmanr
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Univariate
    "plot_categorical_distribution",
    "plot_numeric_diagnostics",
    "plot_descriptive_summary",

    # Missing Data
    "plot_missingness_map",

    # Bivariate / Association
    "plot_categorical_association",
    "plot_numeric_distribution",
    "analyze_chi2_association",
    "analyze_continuous_association",

    # Multivariate
    "plot_spearman_heatmap",
    "plot_mrmr_importance",
    "plot_pca_rank_2d",
    "plot_pca_3d_interactive",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PRIMARY_BLUE  = "#1f77b4"
_ACCENT_RED    = "#d62728"
_COLOR_PALETTE = [_PRIMARY_BLUE, _ACCENT_RED, "#2ca02c", "#ff7f0e"]

_TABLE_STYLES = [
    {"selector": "",   "props": [("margin", "20px auto"), ("border-collapse", "collapse")]},
    {"selector": "th", "props": [
        ("background-color", "#f2f2f2"), ("color", "black"),
        ("font-weight", "bold"), ("border", "1px solid black"), ("padding", "10px"),
    ]},
    {"selector": "td", "props": [
        ("border", "1px solid black"), ("padding", "10px"), ("text-align", "center"),
    ]},
]


def _apply_global_style() -> None:
    """Apply consistent rcParams and seaborn style before drawing any figure."""
    plt.rcParams.update({"axes.edgecolor": "black", "axes.linewidth": 1.5})
    sns.set_style("white")


def _render_figure(fig: plt.Figure, dpi: int = 100) -> None:
    """
    Encode *fig* as a Base64 PNG and display it as centered HTML.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    dpi : int, optional
        Resolution for the exported image (default 100).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
    display(HTML(
        f'<div style="text-align: center; width: 100%;">'
        f'<img src="data:image/png;base64,{encoded}"></div>'
    ))


def _significance_stars(p: float) -> str:
    """Return APA-style significance stars for a given p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ===========================================================================
# 1. UNIVARIATE — CATEGORICAL
# ===========================================================================

def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    display_name: str | None = None,
    level_mapping: dict | None = None,
    figsize: tuple[int, int] = (7, 5),
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Render a percentage bar chart for a single categorical variable.

    The height of each bar represents the relative frequency (%) of that
    category. The absolute count (N) is annotated above each bar.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    column : str
        Name of the categorical column to plot.
    display_name : str, optional
        Human-readable label for the x-axis. Falls back to *column*.
    level_mapping : dict, optional
        Dictionary mapping raw values to display labels,
        e.g. ``{0: "No", 1: "Yes"}``. Keys are coerced to ``str``
        before matching to handle mixed-type indices safely.
    figsize : tuple of (int, int), optional
        Width × height of the figure in inches (default ``(7, 5)``).
    title : str, optional
        Custom title for the plot. If ``None``, a default title is used.
    ylabel : str, optional
        Custom label for the y-axis. Defaults to ``"Percentage (%)"``.

    Examples
    --------
    >>> plot_categorical_distribution(
    ...     df,
    ...     "Sex",
    ...     display_name="Sex",
    ...     level_mapping={0: "Female", 1: "Male"},
    ...     title="Sex Distribution in Sample",
    ...     ylabel="Proportion (%)"
    ... )
    """
    counts = df[column].value_counts().sort_index()
    pcts   = df[column].value_counts(normalize=True).sort_index() * 100

    axis_label = display_name or column

    # Build display labels (robust to mixed-type index)
    if level_mapping:
        str_map = {str(k): v for k, v in level_mapping.items()}
        labels  = [str_map.get(str(x), x) for x in counts.index]
    else:
        labels = counts.index.astype(str).tolist()

    # Title and ylabel logic
    plot_title = title or f"Distribution of {axis_label} (%)"
    y_label    = ylabel or "Percentage (%)"

    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        labels,
        pcts.values,
        color=_COLOR_PALETTE[: len(counts)],
        edgecolor="black",
        alpha=0.8,
        width=0.6,
    )

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_xlabel(axis_label, fontsize=12, fontweight="bold", labelpad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2,
            f"{int(n):,}",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    plt.tight_layout()
    _render_figure(fig)


# ===========================================================================
# 2. MISSING DATA — MISSINGNESS MAP
# ===========================================================================

def plot_missingness_map(
    df: pd.DataFrame,
    title: str = "Information Density Map",
    missing_color: str = _ACCENT_RED,
    figsize: tuple[int, int] = (10, 5),
    title_pad: int = 15,
    label_pad: int = 10,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Display a heatmap that reveals missing-value patterns across the data frame.

    Each cell is coloured **white** if the value is present and
    ``missing_color`` if it is ``NaN``. Row tick labels are hidden by
    default to keep the chart readable for large data sets.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to inspect for missing values.
    title : str, optional
        Main chart title (default ``"Information Density Map"``).
    missing_color : str, optional
        Hex colour used to highlight missing cells (default red ``#d62728``).
    figsize : tuple of (int, int), optional
        Width × height in inches (default ``(10, 5)``).
    title_pad : int, optional
        Padding in points between the title and the top of the axes (default 15).
    label_pad : int, optional
        Padding in points for axis labels (default 10).
    xlabel : str, optional
        Custom label for the x-axis. Defaults to ``"Attributes (Variables)"``.
    ylabel : str, optional
        Custom label for the y-axis. Defaults to ``"Records (Observations)"``.

    Examples
    --------
    >>> plot_missingness_map(
    ...     df,
    ...     title='Heart Dataset — Missing Values',
    ...     xlabel='Features',
    ...     ylabel='Patients'
    ... )
    """
    _apply_global_style()
    cmap_custom = ListedColormap(["white", missing_color])

    # Default labels
    x_label = xlabel or "Attributes (Variables)"
    y_label = ylabel or "Records (Observations)"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df.isnull(),
        ax=ax,
        yticklabels=False,
        cbar=False,
        cmap=cmap_custom,
        edgecolor=None,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=title_pad)
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold", labelpad=label_pad)
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold", labelpad=label_pad)

    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig)


# ===========================================================================
# 3. UNIVARIATE — NUMERIC DIAGNOSTICS (Hampel / QQ)
# ===========================================================================

def plot_numeric_diagnostics(
    df: pd.DataFrame,
    column: str,
    display_name: str,
    k: int = 3,
    show_qq: bool = False,
    show_hampel: bool = True,
    figsize: tuple[float, float] = (11, 4.5),
    ylim: tuple[float, float] | None = None,
    spanish: bool = True,
) -> None:
    """
    Two-panel diagnostic plot for a numeric variable.

    **Left panel** — Histogram with a fitted normal curve and the
    Lilliefors normality test statistic.

    **Right panel** — Either a boxplot with optional Hampel outlier bounds
    (``show_qq=False``, default) or a normal Q-Q plot (``show_qq=True``).

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    column : str
        Name of the numeric column.
    display_name : str
        Human-readable axis label for the variable.
    k : int, optional
        Hampel multiplier: bounds are ``median ± k × 1.4826 × MAD``
        (default ``3``).
    show_qq : bool, optional
        If ``True``, replace the boxplot with a normal Q-Q plot
        (default ``False``).
    show_hampel : bool, optional
        Draw Hampel fence lines on the boxplot when ``show_qq=False``
        (default ``True``).
    figsize : tuple of (float, float), optional
        Width × height in inches (default ``(11, 4.5)``).
    ylim : tuple of (float, float), optional
        Force a fixed y-axis limit on the histogram panel,
        e.g. ``(0, 0.05)``. Useful when comparing distributions across
        variables with very different scales.
    spanish : bool, optional
        If ``True``, translate labels to Spanish:
        ``Density → Densidad``,
        ``Theoretical Quantiles → Cuantiles Teóricos``,
        ``Observed Quantiles → Cuantiles Empíricos`` (default ``True``).

    Notes
    -----
    The Lilliefors test is a modified Kolmogorov-Smirnov test suitable
    when the population mean and variance are unknown. Significance stars
    follow the convention ``* p<0.05``, ``** p<0.01``, ``*** p<0.001``.

    Examples
    --------
    >>> plot_numeric_diagnostics(df, "Cholesterol", "Total Cholesterol (mg/dL)",
    ...                          k=3, show_qq=True, spanish=True)
    >>> plot_numeric_diagnostics(df, "Cholesterol", "Total Cholesterol (mg/dL)",
    ...                          show_qq=True, spanish=False)
    """
    _apply_global_style()

    data  = df[column].dropna()
    n_obs = len(data)

    # Lilliefors normality test
    stat, p_val = lilliefors(data)
    stars        = _significance_stars(p_val)
    lilliefors_text = f"Lilliefors' D({n_obs}): {stat:.3f}{stars}"

    # Hampel fence
    median     = data.median()
    mad_scaled = 1.4826 * np.median(np.abs(data - median))
    lower_h    = median - k * mad_scaled
    upper_h    = median + k * mad_scaled

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Labels depending on language
    density_label = "Densidad" if spanish else "Density"
    theo_label = "Cuantiles Teóricos" if spanish else "Theoretical Quantiles"
    obs_label = "Cuantiles Empíricos" if spanish else "Observed Quantiles"

    # ------------------------------------------------------------------
    # Left panel: Histogram + normal curve
    # ------------------------------------------------------------------
    sns.histplot(data, kde=False, stat="density",
                 color=_PRIMARY_BLUE, edgecolor="black", alpha=0.7, ax=axes[0])

    mu, std = data.mean(), data.std()
    x_range = np.linspace(data.min(), data.max(), 100)
    axes[0].plot(x_range, norm.pdf(x_range, mu, std),
                 color=_ACCENT_RED, linestyle="--", linewidth=2)

    if ylim is not None:
        axes[0].set_ylim(ylim)

    axes[0].text(0.04, 0.95, lilliefors_text, transform=axes[0].transAxes,
                 fontsize=12, fontweight="bold", verticalalignment="top")
    axes[0].set_xlabel(display_name, fontsize=11, fontweight="bold", labelpad=15)
    axes[0].set_ylabel(density_label, fontsize=11, fontweight="bold", labelpad=15)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.3)

    # ------------------------------------------------------------------
    # Right panel: Boxplot (default) or Q-Q plot
    # ------------------------------------------------------------------
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    if show_qq:
        (osm, osr), (slope, intercept, _) = probplot(data, dist="norm", plot=None)
        axes[1].scatter(osm, osr, color=_PRIMARY_BLUE, alpha=1,
                        edgecolor="none", s=15)
        axes[1].plot(osm, slope * osm + intercept,
                     color=_ACCENT_RED, linestyle="--", linewidth=2)
        axes[1].set_xlabel(theo_label, fontsize=12,
                           fontweight="bold", labelpad=15)
        axes[1].set_ylabel(obs_label, fontsize=12,
                           fontweight="bold", labelpad=15)
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        sns.boxplot(
            x=data, color=_PRIMARY_BLUE, ax=axes[1],
            flierprops={
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "alpha": 0.8,
                "markersize": 4,
            },
        )
        if show_hampel:
            axes[1].axvline(lower_h, color=_ACCENT_RED, linestyle=":", linewidth=2.5)
            axes[1].axvline(upper_h, color=_ACCENT_RED, linestyle=":", linewidth=2.5)
        axes[1].set_xlabel(display_name, fontsize=11, fontweight="bold", labelpad=15)
        axes[1].set_ylabel("", visible=False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig, dpi=110)


# ===========================================================================
# 4. MISSING DATA — CHI-SQUARE ASSOCIATION (categorical)
# ===========================================================================

def analyze_chi2_association(
    df: pd.DataFrame,
    target: str,
    feature_list: list[str],
    correction_method: str | None = "holm",
    spanish: bool = False,
    title: str | None = None,
) -> None:
    """
    Chi-square tests of independence between categorical features and a binary target.

    For each variable in *feature_list* a contingency table is built, the
    Pearson Chi-square statistic is computed, and Cramér's V is derived as a
    standardised effect-size measure. Multiple-testing correction is applied
    to the raw p-values before significance stars are assigned.

    Results are displayed as a styled HTML table sorted by Cramér's V
    (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    target : str
        Binary outcome / grouping variable.
    feature_list : list of str
        Categorical columns to test against *target*.
    correction_method : {"holm", "bonferroni", None}, optional
        Multiple-testing correction.
    spanish : bool, optional
        If True, display results in Spanish.
    title : str, optional
        Custom title for the results table. If ``None``, no title is displayed.

    Examples
    --------
    >>> analyze_chi2_association(
    ...     df_missing,
    ...     target="Cholesterol_Missing",
    ...     feature_list=["Sex", "ChestPainType"],
    ...     spanish=True,
    ...     title="Asociación con valores faltantes"
    ... )
    """
    from scipy.stats import chi2_contingency, chi2
    from statsmodels.stats.multitest import multipletests

    # Auxiliares que faltaban en el original para que no falle al ejecutar
    def _significance_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    _TABLE_STYLES = [
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f2f2f2'),
                                     ('color', 'black'), ('font-weight', 'bold'),
                                     ('border', '1px solid black'), ('padding', '10px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid black'),
                                     ('padding', '10px')]},
        {'selector': '', 'props': [('margin-left', 'auto'), ('margin-right', 'auto')]}
    ]

    if df[target].nunique(dropna=True) < 2:
        print(f"[ERROR] Target '{target}' has less than 2 categories.")
        return

    raw_results = []
    skipped = []

    for col in feature_list:
        temp = df[[col, target]].dropna()

        if temp.empty:
            skipped.append(col)
            continue

        table = pd.crosstab(temp[col], temp[target])
        r, k = table.shape

        if r < 2 or k < 2:
            skipped.append(col)
            continue

        chi2_obs, p_raw, dof, _ = chi2_contingency(table)
        crit_val = chi2.ppf(0.95, dof)

        n = len(temp)
        phi2 = chi2_obs / n
        v_cramer = np.sqrt(phi2 / min(k - 1, r - 1))

        raw_results.append({
            "Feature": col,
            "Chi2_Obs": chi2_obs,
            "DF": int(dof),
            "Critical Val": crit_val,
            "P_Raw": p_raw,
            "V_Cramer": v_cramer,
        })

    if not raw_results:
        print("[WARNING] No variables could be processed.")
        print(f"Skipped variables: {skipped}")
        return

    df_res = pd.DataFrame(raw_results)

    if correction_method:
        _, p_corr, _, _ = multipletests(df_res["P_Raw"], method=correction_method)
        df_res["P_Final"] = p_corr
    else:
        df_res["P_Final"] = df_res["P_Raw"]

    def _interpret_v(v: float) -> str:
        if v < 0.1:  return "Despreciable" if spanish else "Negligible"
        if v < 0.3:  return "Pequeño" if spanish else "Small"
        if v < 0.5:  return "Mediano" if spanish else "Medium"
        return "Grande" if spanish else "Large"

    df_res["Chi-square"] = df_res.apply(
        lambda r: f"{r['Chi2_Obs']:.2f}{_significance_stars(r['P_Final'])}", axis=1
    )
    df_res["Cramér's V"] = df_res["V_Cramer"].map("{:.2f}".format)
    df_res["Effect"] = df_res["V_Cramer"].apply(_interpret_v)
    df_res["Critical Val"] = df_res["Critical Val"].map("{:.2f}".format)

    summary = df_res[
        ["Feature", "Chi-square", "DF", "Critical Val", "Cramér's V", "Effect"]
    ].sort_values("Cramér's V", ascending=False)

    if spanish:
        summary.columns = [
            "Variable",
            "Chi-cuadrado",
            "GL",
            "Valor Crítico",
            "V de Cramér",
            "Efecto",
        ]

    # AJUSTE DE CENTRADO AQUÍ
    styled = (
        summary.style
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
        .set_properties(
            subset=[summary.columns[0]],
            **{"text-align": "center", "font-weight": "bold"} # Forzamos center
        )
    )

    html_title = f"<h3>{title}</h3>" if title else ""

    display(HTML(
        f"<div style='text-align: center;'>"
        f"{html_title}"
        + styled.to_html()
        + "</div>"
    ))
    
# ===========================================================================
# 5. MISSING DATA — MANN-WHITNEY ASSOCIATION (continuous)
# ===========================================================================

def analyze_continuous_association(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    correction_method: str | None = "holm",
    spanish: bool = True,
    group_names: tuple[str, str] = ("Data Present", "Data Absent"),
) -> None:
    """
    Mann-Whitney U and Brown-Forsythe tests for continuous features vs. a binary target.

    For each numeric variable the two groups defined by *target* are compared
    using the non-parametric Mann-Whitney U test for location and the
    Brown-Forsythe variant of Levene's test for variance homogeneity.
    Rosenthal's *r* is computed as an effect-size estimate.

    Results are displayed as a styled HTML table sorted by *r* (descending).

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    target : str
        Binary grouping variable (values 0 and 1).
    numeric_features : list of str
        Names of numeric columns to test.
    correction_method : {"holm", "bonferroni", None}, optional
        Multiple-testing correction applied separately to U-test p-values and
        Brown-Forsythe p-values (default ``"holm"``).
    spanish : bool, optional
        If True, labels are displayed in Spanish.
    group_names : tuple of str, optional
        Custom names for the two groups shown in the table
        (default ``("Data Present", "Data Absent")``).

    Examples
    --------
    >>> analyze_continuous_association(
    ...     df, "Cholesterol_Missing",
    ...     ["Age", "RestingBP"],
    ...     spanish=True,
    ...     group_names=("Dato Presente", "Dato Ausente"),
    ... )
    """
    raw_results = []

    for col in numeric_features:
        g0 = df[df[target] == 0][col].dropna()
        g1 = df[df[target] == 1][col].dropna()

        if len(g0) < 2 or len(g1) < 2:
            continue

        g0_med, g0_iqr = g0.median(), g0.quantile(0.75) - g0.quantile(0.25)
        g1_med, g1_iqr = g1.median(), g1.quantile(0.75) - g1.quantile(0.25)

        u_stat, p_u  = mannwhitneyu(g0, g1, alternative="two-sided")
        _,      p_bf = levene(g0, g1, center="median")

        n1, n2 = len(g0), len(g1)
        N      = n1 + n2
        mu_u   = (n1 * n2) / 2
        sigma_u = np.sqrt((n1 * n2 * (N + 1)) / 12)
        z       = (u_stat - mu_u) / sigma_u
        r_rosenthal = abs(z) / np.sqrt(N)

        raw_results.append({
            "Feature":        col,
            "Present_Med_IQR": f"{g0_med:.1f} ({g0_iqr:.1f})",
            "Absent_Med_IQR":  f"{g1_med:.1f} ({g1_iqr:.1f})",
            "U_Obs":           u_stat,
            "P_U_Raw":         p_u,
            "P_BF_Raw":        p_bf,
            "r_Rosenthal":     r_rosenthal,
        })

    if not raw_results:
        print("Insufficient data for the specified numeric variables.")
        return

    df_res = pd.DataFrame(raw_results)

    if correction_method:
        _, p_u_corr,  _, _ = multipletests(df_res["P_U_Raw"],  method=correction_method)
        _, p_bf_corr, _, _ = multipletests(df_res["P_BF_Raw"], method=correction_method)
        df_res["P_U_Final"]  = p_u_corr
        df_res["P_BF_Final"] = p_bf_corr
    else:
        df_res["P_U_Final"]  = df_res["P_U_Raw"]
        df_res["P_BF_Final"] = df_res["P_BF_Raw"]

    def _interpret_r(r: float) -> str:
        if r < 0.1:  return "Despreciable" if spanish else "Negligible"
        if r < 0.3:  return "Pequeño"      if spanish else "Small"
        if r < 0.5:  return "Mediano"      if spanish else "Medium"
        return "Grande"                    if spanish else "Large"

    g0_name, g1_name = group_names

    df_res["Data Present"] = df_res["Present_Med_IQR"]
    df_res["Data Absent"]  = df_res["Absent_Med_IQR"]

    df_res["U"] = df_res.apply(
        lambda r: f"{r['U_Obs']:.2f}{_significance_stars(r['P_U_Final'])}", axis=1
    )

    df_res["Homoced."] = df_res["P_BF_Final"].apply(
        lambda p: (
            "Sí" if p >= 0.05 else "No"
        ) if spanish else (
            "Yes" if p >= 0.05 else "No"
        )
    )

    df_res["r"] = df_res["r_Rosenthal"].map("{:.2f}".format)
    df_res["Effect"]   = df_res["r_Rosenthal"].apply(_interpret_r)

    summary = df_res[[
        "Feature", "Data Present", "Data Absent",
        "Homoced.", "U", "r", "Effect",
    ]].copy()

    if spanish:
        summary = summary.rename(columns={
            "Feature": "Variable",
            "Data Present": g0_name,
            "Data Absent": g1_name,
            "Effect": "Efecto",
        })
        sort_col = "r"
    else:
        summary = summary.rename(columns={
            "Data Present": g0_name,
            "Data Absent": g1_name,
        })
        sort_col = "r"

    summary = summary.sort_values(sort_col, ascending=False)

    styled = (
        summary.style
        .hide(axis="index")
        .set_table_styles(
            _TABLE_STYLES + [
                {"selector": "th", "props": [("text-align", "center")]},
            ]
        )
        .set_properties(
            subset=summary.columns,
            **{"text-align": "center"}
        )
    )

    display(HTML(
        "<div style='text-align: center;'>"
        + styled.to_html()
        + "</div>"
    ))

# ===========================================================================
# 6. UNIVARIATE — DESCRIPTIVE SUMMARY TABLE
# ===========================================================================

def plot_descriptive_summary(
    df: pd.DataFrame,
    numeric_features: list[str],
    spanish: bool = True,
) -> None:
    """
    Render a styled HTML table of descriptive statistics for numeric variables.

    Reports: **Mean (SD)**, **Median (IQR)**, **Min**, **Max**,
    **Skewness**, and **Kurtosis** (excess, Fisher's definition).

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    numeric_features : list of str
        Columns to include in the summary.
    spanish : bool, optional
        If True, labels are displayed in Spanish:
        'Mean (SD)' → 'Media (SD)',
        'Median (IQR)' → 'Mediana (IQR)',
        'Minimum' → 'Mínimo',
        'Maximum' → 'Máximo',
        'Skewness' → 'Asimetría',
        'Kurtosis' → 'Curtosis'.

    Examples
    --------
    >>> plot_descriptive_summary(df, ["Age", "RestingBP", "Cholesterol",
    ...                               "MaxHR", "Oldpeak"], spanish=True)
    >>> plot_descriptive_summary(df, ["Age", "RestingBP"], spanish=False)
    """
    desc = df[numeric_features].describe().T
    desc["IQR"]      = desc["75%"] - desc["25%"]
    desc["Skewness"] = df[numeric_features].skew()
    desc["Kurtosis"] = df[numeric_features].kurt()

    desc["Mean (SD)"]    = desc.apply(lambda r: f"{r['mean']:.2f} ({r['std']:.2f})", axis=1)
    desc["Median (IQR)"] = desc.apply(lambda r: f"{r['50%']:.2f} ({r['IQR']:.2f})", axis=1)

    summary = (
        desc[["Mean (SD)", "Median (IQR)", "min", "max", "Skewness", "Kurtosis"]]
        .copy()
        .rename(columns={"min": "Minimum", "max": "Maximum"})
        .reset_index()
        .rename(columns={"index": "Variable"})
    )

    if spanish:
        summary = summary.rename(columns={
            "Variable": "Variable",
            "Mean (SD)": "Media (SD)",
            "Median (IQR)": "Mediana (IQR)",
            "Minimum": "Mínimo",
            "Maximum": "Máximo",
            "Skewness": "Asimetría",
            "Kurtosis": "Curtosis",
        })

    styled = (
        summary.style
        .hide(axis="index")
        .set_table_styles(_TABLE_STYLES)
        .format({
            summary.columns[3]: "{:.2f}",
            summary.columns[4]: "{:.2f}",
            summary.columns[5]: "{:.2f}",
            summary.columns[6]: "{:.2f}",
        })
        .set_properties(subset=[summary.columns[0]],
                        **{"text-align": "left", "font-weight": "bold"})
    )

    display(HTML(
        "<div style='text-align: center; width: 100%;'>"
        + styled.to_html()
        + "</div>"
    ))


# ===========================================================================
# 7. BIVARIATE — CATEGORICAL vs. BINARY TARGET
# ===========================================================================

def plot_categorical_association(
    df: pd.DataFrame,
    column: str,
    display_name: str | None = None,
    level_mapping: dict | None = None,
    target: str = "HeartDisease",
    target_mapping: dict | None = None,
    chart_title: str | None = None,
    ylabel: str | None = None,
    legend_title: str = "Category",
    legend_position: str = "upper left",
    label_spacing: float = 0.9,
    spanish: bool = False,
) -> None:
    """
    Grouped bar chart showing the proportion of each target class within
    every level of a categorical predictor.

    Each bar group corresponds to one level of *column*. Within the group,
    bars are coloured by *target* class (blue = first class, red = second).
    Absolute counts are annotated above every bar.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    column : str
        Categorical predictor column.
    display_name : str, optional
        Human-readable name for *column* (axis label).
    level_mapping : dict, optional
        Maps raw predictor values to display labels.
    target : str, optional
        Binary outcome column (default ``"HeartDisease"``).
    target_mapping : dict, optional
        Maps raw target values to display labels.
    chart_title : str, optional
        Custom figure title. If ``None``, no title is displayed.
    ylabel : str, optional
        Custom label for the y-axis. If ``None``, a default label is used.
    legend_title : str, optional
        Title for the legend.
    legend_position : str, optional
        Matplotlib legend location.
    label_spacing : float, optional
        Vertical spacing between legend entries.
    spanish : bool, optional
        If True, labels are displayed in Spanish.

    Examples
    --------
    >>> plot_categorical_association(
    ...     df,
    ...     column="ChestPainType",
    ...     display_name="Chest Pain Type",
    ...     target_mapping={"0": "No Disease", "1": "Disease"},
    ...     ylabel="Percentage (%)",
    ...     chart_title="Association with Heart Disease",
    ... )
    """
    axis_label = display_name or column

    default_ylabel = (
        "Proporción dentro del nivel (%)"
        if spanish else "Proportion within Level (%)"
    )
    final_ylabel = ylabel or default_ylabel

    df_plot = df[[column, target]].copy()
    df_plot[column] = df_plot[column].astype(str)
    df_plot[target] = df_plot[target].astype(str)

    if level_mapping:
        df_plot[column] = df_plot[column].map(level_mapping)
    if target_mapping:
        df_plot[target] = df_plot[target].map(target_mapping)

    counts = df_plot.groupby([column, target]).size().reset_index(name="Count")
    totals = df_plot.groupby(column).size().reset_index(name="Total")
    plot_df = pd.merge(counts, totals, on=column)
    plot_df["Percentage"] = (plot_df["Count"] / plot_df["Total"]) * 100

    _apply_global_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=plot_df,
        x=column,
        y="Percentage",
        hue=target,
        palette=[_PRIMARY_BLUE, _ACCENT_RED],
        edgecolor="black",
        alpha=0.8,
        ax=ax,
    )

    ax.set_ylim(0, 115)
    ax.set_yticks(np.arange(0, 101, 10))

    if chart_title:
        ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=18)

    ax.set_ylabel(final_ylabel, fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlabel(axis_label, fontsize=12, fontweight="bold", labelpad=10)

    ax.legend(
        title=legend_title,
        loc=legend_position,
        frameon=False,
        fontsize=13,
        title_fontsize=13,
        labelspacing=label_spacing,
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    cat_order = [t.get_text() for t in ax.get_xticklabels()]
    hue_order = [t.get_text() for t in ax.get_legend().get_texts()]

    for i, container in enumerate(ax.containers):
        current_hue = hue_order[i]
        labels_n = []
        for cat in cat_order:
            row = plot_df[
                (plot_df[column] == cat) & (plot_df[target] == current_hue)
            ]
            labels_n.append(
                f"{int(row['Count'].values[0]):,}" if not row.empty else "0"
            )

        ax.bar_label(
            container,
            labels=labels_n,
            padding=4,
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()
    _render_figure(fig)

# ===========================================================================
# 8. BIVARIATE — NUMERIC vs. BINARY TARGET (KDE)
# ===========================================================================

def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    display_name: str | None = None,
    target: str = "HeartDisease",
    target_mapping: dict | None = None,
    chart_title: str | None = None,
    ylabel: str | None = None,
    legend_title: str = "Category",
    legend_position: str = "upper left",
    label_spacing: float = 1,
    figsize: tuple[int, int] = (8, 5),
    spanish: bool = False,
) -> None:
    """
    Kernel density plot comparing the distribution of a numeric variable
    across two target classes.

    Each class is rendered with a filled kernel density estimate. The
    ``common_norm=False`` option ensures that each density is normalised
    independently, preserving the shape regardless of class imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.
    column : str
        Numeric column to visualise.
    display_name : str, optional
        Human-readable name for *column* (axis label).
    target : str, optional
        Binary grouping column (default ``"HeartDisease"``).
    target_mapping : dict, optional
        Maps raw target values to display labels.
    chart_title : str, optional
        Custom figure title. If ``None``, no title is displayed.
    ylabel : str, optional
        Custom label for the y-axis. If ``None``, a default label is used.
    legend_title : str, optional
        Title for the legend.
    legend_position : str, optional
        Matplotlib legend location.
    label_spacing : float, optional
        Vertical spacing between legend entries.
    figsize : tuple of (int, int), optional
        Width × height in inches (default ``(8, 5)``).
    spanish : bool, optional
        If True, labels are displayed in Spanish.

    Examples
    --------
    >>> plot_numeric_distribution(
    ...     df,
    ...     column="Age",
    ...     display_name="Age",
    ...     target_mapping={"0": "No CAD", "1": "CAD"},
    ...     chart_title="Age Distribution by Diagnosis",
    ...     ylabel="Probability Density",
    ...     legend_title="Health Status",
    ... )
    """
    axis_label = display_name or column

    default_ylabel = (
        "Densidad de probabilidad"
        if spanish else "Probability Density"
    )
    final_ylabel = ylabel or default_ylabel

    df_temp = df[[column, target]].copy().dropna()
    df_temp[column] = df_temp[column].astype(float)
    df_temp[target] = df_temp[target].astype(str)

    if target_mapping:
        df_temp[target] = df_temp[target].map(target_mapping)

    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        data=df_temp,
        x=column,
        hue=target,
        palette=[_PRIMARY_BLUE, _ACCENT_RED],
        fill=True,
        alpha=0.4,
        linewidth=2.5,
        common_norm=False,
        ax=ax,
    )

    if chart_title:
        ax.set_title(chart_title, fontsize=13, fontweight="bold", pad=18)

    ax.set_ylabel(final_ylabel, fontsize=12, fontweight="bold", labelpad=10)
    ax.set_xlabel(axis_label, fontsize=12, fontweight="bold", labelpad=10)

    if ax.get_legend():
        leg = ax.get_legend()
        handles = leg.legend_handles
        
        for h in handles:
            if hasattr(h, "set_edgecolor"):
                h.set_edgecolor("none")
        
        ax.legend(
            handles=leg.legend_handles,
            labels=[t.get_text() for t in leg.get_texts()],
            title=legend_title,
            loc=legend_position,
            frameon=False,
            fontsize=13,
            title_fontsize=13,
            labelspacing=label_spacing,
        )

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig)


# ===========================================================================
# 9. MULTIVARIATE — SPEARMAN CORRELATION HEATMAP
# ===========================================================================

def plot_spearman_heatmap(
    df: pd.DataFrame,
    features: list[str],
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    annot_size: int = 8,
) -> None:
    """
    Lower-triangle Spearman rank-correlation heatmap with annotated coefficients.

    Coefficients are rounded to two decimal places. The colour scale runs
    from **−1** (strong negative, blue) through **0** (white) to
    **+1** (strong positive, red) using the ``RdBu_r`` diverging palette.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame (only *features* columns are used).
    features : list of str
        Numeric columns to include in the correlation matrix.
    figsize : tuple of (int, int), optional
        Width × height in inches (default ``(8, 5)``).
    title : str, optional
        Custom chart title. If ``None``, no title is displayed.
    annot_size : int, optional
        Font size of the correlation values inside the cells (default ``8``).

    Examples
    --------
    >>> plot_spearman_heatmap(
    ...     df,
    ...     ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"],
    ...     title="Correlation Matrix",
    ...     annot_size=10
    ... )
    """
    corr_matrix = df[features].corr(method="spearman")
    mask        = np.triu(np.ones_like(corr_matrix, dtype=bool))

    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    res = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        linewidths=0.5, linecolor="white",
        xticklabels=True, yticklabels=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
        annot_kws={"size": annot_size},
    )

    cbar = res.collections[0].colorbar
    cbar.outline.set_linewidth(1.2)
    cbar.set_label("Spearman's ρ", labelpad=13, weight="bold", fontsize=12)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=14)

    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig, dpi=100)

# ===========================================================================
# 10. MULTIVARIATE — mRMR + MUTUAL INFORMATION BAR CHART
# ===========================================================================

def plot_mrmr_importance(
    df: pd.DataFrame,
    target_col: str = "HeartDisease",
    top_k: int = 10,
    figsize: tuple[int, int] = (8, 5),
    spanish: bool = True,
) -> None:
    """
    Horizontal bar chart of feature importance using mRMR + Mutual Information.

    The **mRMR** (minimum Redundancy Maximum Relevance) algorithm selects the
    top-*k* features that maximise relevance to the target while minimising
    redundancy among themselves. Bar length encodes the raw **Mutual
    Information** (MI) score between each feature and *target_col*.

    Features are ranked from highest (top) to lowest (bottom) mRMR relevance.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame containing numeric features and *target_col*.
    target_col : str, optional
        Binary target column (default ``"HeartDisease"``).
    top_k : int, optional
        Number of top features to select and display (default ``10``).
    figsize : tuple of (int, int), optional
        Width × height in inches (default ``(8, 5)``).
    spanish : bool, optional
        If True, axis labels are displayed in Spanish.

    Notes
    -----
    Binary features (≤ 2 unique values) are treated as discrete when
    computing MI via ``sklearn.feature_selection.mutual_info_classif``.

    Requires: ``mrmr-selection`` (``pip install mrmr-selection``).

    Examples
    --------
    >>> plot_mrmr_importance(df, target_col="HeartDisease", top_k=10, spanish=True)
    """
    from mrmr import mrmr_classif
    from sklearn.feature_selection import mutual_info_classif

    X = (
        df.drop(columns=[target_col])
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    y = df[target_col]
    k_actual = min(top_k, X.shape[1])

    selected  = mrmr_classif(X=X, y=y, K=k_actual, show_progress=False)
    X_top     = X[selected]
    discrete  = [X_top[col].nunique() <= 2 for col in X_top.columns]
    mi_scores = mutual_info_classif(X_top, y, discrete_features=discrete, random_state=42)

    df_plot = pd.DataFrame({"Feature": selected, "MI": mi_scores}).iloc[::-1]

    _apply_global_style()
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    bars = ax.barh(
        df_plot["Feature"], df_plot["MI"],
        color=_PRIMARY_BLUE, edgecolor="black", alpha=0.8, linewidth=1.2,
    )

    # Labels depending on language
    xlabel = "Información Mutua (MI — bits)" if spanish else "Mutual Information (MI — bits)"
    ylabel = "Variables" if spanish else "Features"

    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", labelpad=12)

    ax.set_xlim(0, df_plot["MI"].max() * 1.2)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            ha="left", va="center",
            fontsize=11, fontweight="bold",
        )

    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig, dpi=120)

# ===========================================================================
# 11. MULTIVARIATE — RANK-BASED PCA (2-D)
# ===========================================================================

def plot_pca_rank_2d(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "HeartDisease",
    target_mapping: dict | None = None,
    legend_title: str = "Category",
    legend_position: str = "best",
    spanish: bool = True,
) -> None:
    """
    Two-panel rank-based PCA: scree plot and 2-D projection scatter.

    Standard PCA is sensitive to outliers and non-normal distributions.
    This function implements a Robust PCA approach using Spearman's rank 
    correlation matrix as the fundamental coordinate system:

    1. **Robust Pre-scaling**: Uses RobustScaler (median/IQR) to handle 
       outliers in the raw feature space.
    2. **Rank Transformation**: Converts data to ranks to capture monotonic 
       relationships and ensure distribution-free properties.
    3. **Spearman Eigen-decomposition**: Computes the PCA over the Spearman 
       correlation matrix (equivalent to Pearson on ranks).
    4. **Standardized Projection**: Standardizes the ranks before projecting 
       onto eigenvectors to ensure variance consistency.

    **Left panel** — Cumulative explained-variance scree plot with a
    90 % reference line.

    **Right panel** — 2-D scatter of PC1 vs PC2, coloured by *target_col*.
    """
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from scipy.stats import spearmanr

    df_temp = df[features + [target_col]].dropna()
    X = df_temp[features]
    y = df_temp[target_col]

    scaler_robust = RobustScaler()
    X_scaled = scaler_robust.fit_transform(X)
    
    # Rank conversion
    df_ranks = pd.DataFrame(X_scaled, columns=features).rank()

    corr_sp = df_ranks.corr(method='pearson')
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(corr_sp)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    scaler_std = StandardScaler()
    ranks_standardized = scaler_std.fit_transform(df_ranks)
    X_pca = np.dot(ranks_standardized, eigenvectors)

    exp_var = eigenvalues / np.sum(eigenvalues)
    cum_var = np.cumsum(exp_var)
    n_comp  = len(cum_var)

    # --- Label handling ---
    y_str = y.astype(str)
    if target_mapping:
        mapping_str = {str(k): v for k, v in target_mapping.items()}
        labels = y_str.map(mapping_str)
    else:
        labels = y_str

    unique_vals = sorted(y_str.unique())
    palette = {
        unique_vals[0]: _PRIMARY_BLUE,
        unique_vals[1]: _ACCENT_RED if len(unique_vals) > 1 else _PRIMARY_BLUE,
    }

    # --- Language ---
    if spanish:
        scree_title = "Varianza Acumulada (Basada en Rangos)"
        x_label_scree = "Número de Componentes"
        y_label_scree = "Varianza Explicada (%)"
        scatter_title = "Proyección PCA (Spearman)"
    else:
        scree_title = "Cumulative Variance (Rank-Based)"
        x_label_scree = "Number of Components"
        y_label_scree = "Explained Variance (%)"
        scatter_title = "PCA Projection (Spearman)"

    _apply_global_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor="white")

    # --- Left: Scree plot ---
    x_axis = np.arange(1, n_comp + 1)
    axes[0].plot(
        x_axis, cum_var * 100,
        color=_PRIMARY_BLUE, marker="o",
        markersize=6, linewidth=2,
        markerfacecolor="white", markeredgewidth=2
    )
    axes[0].axhline(y=90, color=_ACCENT_RED, linestyle="--", alpha=0.7, linewidth=1.5)

    axes[0].set_title(scree_title, fontsize=12, fontweight="bold", pad=15)
    axes[0].set_xlabel(x_label_scree, fontsize=12, fontweight="bold", labelpad=12)
    axes[0].set_ylabel(y_label_scree, fontsize=12, fontweight="bold", labelpad=12)

    axes[0].set_xticks(x_axis)
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.3)

    # --- Right: Scatter ---
    for val in unique_vals:
        mask = y_str == val
        label_name = labels[mask].iloc[0]
        color = palette[val]

        axes[1].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=color,
            label=label_name,
            alpha=0.6,
            edgecolor="black",
            s=45,
            linewidth=0.6,
        )

    axes[1].set_title(scatter_title, fontsize=12, fontweight="bold", pad=15)
    axes[1].set_xlabel(f"PC1 ({exp_var[0]*100:.1f}%)", fontsize=12, fontweight="bold", labelpad=12)
    axes[1].set_ylabel(f"PC2 ({exp_var[1]*100:.1f}%)", fontsize=12, fontweight="bold", labelpad=12)

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()

    axes[1].legend(
        title=legend_title,
        loc=legend_position,
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )

    axes[1].yaxis.grid(True, linestyle="--", alpha=0.3)
    
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=12)
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _render_figure(fig, dpi=120)

# ===========================================================================
# 12. MULTIVARIATE — INTERACTIVE 3-D PCA (Plotly)
# ===========================================================================

def plot_pca_3d_interactive(
    df: pd.DataFrame,
    features: list[str],
    target: str = "HeartDisease",
    target_mapping: dict | None = None,
    title: str | None = None,
    legend_title: str = "Class",
    figsize: tuple[int, int] = (600, 500),
):
    """
    Interactive 3-D PCA scatter plot optimized for MyST Markdown.

    This function implements a Robust PCA approach by performing the 
    eigendecomposition on the Spearman rank correlation matrix to mitigate 
    the influence of outliers and non-normal distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    features : list of str
        Continuous variables to include in the dimensionality reduction.
    target : str
        Column name for class coloring.
    target_mapping : dict, optional
        Works with both str and int keys to remap class labels.
    title : str, optional
        No automatic title; text for the main figure title.
    legend_title : str, optional
        Custom legend title displayed above the horizontal labels.
    figsize : tuple
        Width × height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure object for native site rendering.
    """
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler, StandardScaler

    # --- 1. Data Preprocessing & Robust PCA Engine ---
    df_work = df.dropna(subset=features + [target]).copy()
    X = df_work[features]

    scaler_robust = RobustScaler()
    X_scaled = scaler_robust.fit_transform(X)
    df_ranks = pd.DataFrame(X_scaled, columns=features).rank()

    corr_matrix = df_ranks.corr(method='pearson')
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    scaler_std = StandardScaler()
    ranks_std = scaler_std.fit_transform(df_ranks)
    pca_results = np.dot(ranks_std, eigenvectors[:, :3])
    exp_var = (eigenvalues / np.sum(eigenvalues)) * 100

    df_work["PC1"], df_work["PC2"], df_work["PC3"] = pca_results.T

    # --- 2. Labeling and Color Mapping ---
    df_work["_target_str"] = df_work[target].astype(str)
    if target_mapping:
        mapping_str = {str(k): v for k, v in target_mapping.items()}
        df_work["_label"] = df_work["_target_str"].map(mapping_str)
    else:
        df_work["_label"] = df_work["_target_str"]

    palette = {"0": "#1f77b4", "1": "#d62728"}

    # --- 3. Figure Construction ---
    fig = go.Figure()

    for val in sorted(df_work["_target_str"].unique()):
        mask = df_work["_target_str"] == val
        name = df_work.loc[mask, "_label"].iloc[0]
        color = palette.get(val, "#7f8c8d")

        fig.add_trace(go.Scatter3d(
            x=df_work.loc[mask, 'PC1'], 
            y=df_work.loc[mask, 'PC2'], 
            z=df_work.loc[mask, 'PC3'],
            mode='markers',
            marker=dict(
                size=4.0, 
                color=color, 
                opacity=0.75, 
                line=dict(width=0.3, color="black")
            ),
            name=name,
            customdata=df_work.loc[mask, ["_label", "PC1", "PC2", "PC3"]].values,
            hovertemplate=(
                "<b>Class:</b> %{customdata[0]}<br>"
                "<b>PC1:</b> %{customdata[1]:.2f}<br>"
                "<b>PC2:</b> %{customdata[2]:.2f}<br>"
                "<b>PC3:</b> %{customdata[3]:.2f}"
                "<extra></extra>"
            )
        ))

    # --- 4. Scene and Layout Configuration ---
    axis_style = dict(
        gridcolor="#d1d1d1", 
        showbackground=True, 
        backgroundcolor="#ebebeb",
        zerolinecolor="#999999",
        tickfont=dict(size=10, color="#444444")
    )

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        margin=dict(l=0, r=0, b=0, t=70),
        template="plotly_white",
        
        
        annotations=[dict(
            text=f"<b>{legend_title}</b>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=1.06,         
            xanchor="center", yanchor="bottom",
            font=dict(size=14, color="#2c3e50")
        )],
        
        scene=dict(
            xaxis_title=f"PC1 ({exp_var[0]:.1f}%)",
            yaxis_title=f"PC2 ({exp_var[1]:.1f}%)",
            zaxis_title=f"PC3 ({exp_var[2]:.1f}%)",
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.6))
        ),
        
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.0,              
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            font=dict(size=15),
            bgcolor="rgba(255, 255, 255, 0)"
        )
    )

    return fig