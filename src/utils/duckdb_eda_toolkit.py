"""
duckdb_eda_toolkit
===================

DuckDB-Backed Exploratory Data Analysis & Visualization Toolkit.

A curated collection of visualization and statistical-summary utilities
for structured, reproducible exploratory data analysis (EDA) of
click-through-rate (CTR) style datasets in Jupyter notebooks.

Every function that touches the underlying data receives an active
``duckdb.DuckDBPyConnection`` plus a relation name (a table, a view, or
an inline DuckDB expression such as ``read_parquet('path.parquet')``).
All filtering, grouping and aggregation is pushed down to DuckDB via
SQL, so only the small, already-summarized result (a histogram's bin
values, a contingency table, a preview of a few rows, a cardinality
count, ...) is ever materialized into a pandas DataFrame. This design
assumes the source relation is too large to comfortably fit in memory
and lets the toolkit scale to multi-million-row datasets without ever
loading the full relation into Python.

All figures are rendered inline as vector SVG by default (resolution
independent, ideal for reports and papers), with an optional raster
PNG mode for cases where file size or renderer compatibility matters.
All statistical outputs are rendered as styled HTML tables using a
consistent visual language.

Visual Language
----------------
- Black bars / lines / edges, white fills
- Thin (0.8-1.2 pt) black spines on every axis
- White background, minimal or no gridlines
- Bold axis labels and titles, tight layout
- Default output format: SVG (vector, dpi-independent)

Main Functionalities
---------------------
Rendering
    render_figure

Numeric & Categorical Distributions
    plot_distribution_pair
    plot_categorical_distribution
    plot_categorical_distribution_pair

Temporal Patterns
    plot_temporal_overview
    plot_ctr_by_hour

Association & Correlation
    categorical_association_table
    plot_spearman_heatmap

Cardinality & Encoding Strategy
    categorical_cardinality_table
    categorical_coverage_table
    show_encoding_strategy_table

Data Quality / Summary
    relation_integrity_table
    show_preview_table

Modeling Prep
    create_split_column
    split_summary_table

Dependencies
------------
duckdb, numpy, pandas, matplotlib, scipy, IPython
"""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import base64
import io

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------
__all__ = [
    "categorical_association_table",
    "categorical_cardinality_table",
    "categorical_coverage_table",
    "create_split_column",
    "plot_categorical_distribution",
    "plot_categorical_distribution_pair",
    "plot_ctr_by_hour",
    "plot_distribution_pair",
    "plot_spearman_heatmap",
    "plot_temporal_overview",
    "relation_integrity_table",
    "render_figure",
    "show_encoding_strategy_table",
    "show_preview_table",
    "split_summary_table",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_VALID_P_ADJUST_METHODS = {"bonferroni", "holm"}


# ===========================================================================
# PUBLIC API
# ===========================================================================

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

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
            ``fmt="png"``. Defaults to ``100``, a reasonable balance
            between on-screen clarity and output size for inline
            notebook display.

    Returns:
        None: The figure is displayed as a side effect; nothing is
        returned.

    Raises:
        ValueError: If ``fmt`` is not one of ``{"svg", "png"}``.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> render_figure(fig)
        >>> render_figure(fig, fmt="png", dpi=150)
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
# Numeric & Categorical Distributions
# ---------------------------------------------------------------------------

def plot_distribution_pair(
    con,
    relation="device_features",
    columns=("freq_device_id", "freq_device_ip"),
    titles=("Device ID frequency", "Device IP frequency"),
    xlabels=("Number of impressions", "Number of impressions"),
    ylabel="Frequency",
    bins=50,
    log_x=True,
    log_y=True,
    bar_color="black",
    bar_edge_color="black",
    bar_edge_width=0.5,
    bar_alpha=0.85,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(9, 9, 12),
    sharex=False,
    sharey=False,
    figsize=(10, 3.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot histograms of two numerical columns side by side.

    Creates a figure with two horizontal panels, each showing the
    histogram of one column from ``columns``. Supports logarithmic scaling
    on either axis, which is useful for long-tailed distributions such as
    device or IP frequencies. Both panels are drawn from data retrieved
    with a single query per column, so the underlying relation is never
    loaded entirely into memory.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of a table or view containing the requested
            columns. Defaults to ``"device_features"``.
        columns (tuple or list of 2 str): Names of the numerical columns
            to plot in the left and right panels. Defaults to
            ``("freq_device_id", "freq_device_ip")``.
        titles (tuple or list of 2 str): Titles for the two panels.
            Defaults to ``("Device ID frequency", "Device IP frequency")``.
        xlabels (str or tuple of 2 str): X-axis labels for both panels.
            Defaults to ``("Number of impressions", "Number of
            impressions")``.
        ylabel (str): Y-axis label, drawn only on the left panel.
            Defaults to ``"Frequency"``.
        bins (int): Number of histogram bins. Defaults to 50.
        log_x (bool): If ``True``, the x-axis is drawn in log scale.
            Defaults to ``True``.
        log_y (bool): If ``True``, the y-axis is drawn in log scale.
            Defaults to ``True``.
        bar_color (str): Fill color of the bars. Defaults to ``"black"``.
        bar_edge_color (str): Edge color of the bars. Defaults to
            ``"black"``.
        bar_edge_width (float): Edge line width of the bars. Defaults to
            0.5.
        bar_alpha (float): Opacity of the bars. Defaults to 0.85.
        label_size (float): Font size for the axis labels. Defaults to 11.
        title_size (float): Font size for the panel titles. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        sharex (bool): If ``True``, both panels share the x-axis. Defaults
            to ``False``.
        sharey (bool): If ``True``, both panels share the y-axis. Defaults
            to ``False``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(10, 3.5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax_left, ax_right), (df_left, df_right))`` where
        each DataFrame contains the raw values retrieved for the
        corresponding column.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if any string
            argument has the wrong type, if ``columns`` is not a tuple or
            list of two strings, if ``titles`` is not a tuple or list of
            two strings, or if ``render`` is not a boolean.
        ValueError: If ``bins`` is not a positive integer, or if the
            queries return no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(columns, (tuple, list)) or len(columns) != 2:
        raise TypeError("'columns' must be a tuple or list of two strings.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    if not isinstance(titles, (tuple, list)) or len(titles) != 2:
        raise TypeError("'titles' must be a tuple or list of two strings.")

    if not all(isinstance(t, str) for t in titles):
        raise TypeError("All elements of 'titles' must be strings.")

    if not isinstance(xlabels, (tuple, list)) or len(xlabels) != 2:
        raise TypeError("'xlabels' must be a tuple or list of two strings.")

    if (
        not isinstance(bins, int)
        or isinstance(bins, bool)
        or bins <= 0
    ):
        raise ValueError("'bins' must be a positive integer.")

    _validate_pads(pads)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    df_left = con.sql(
        f'SELECT "{columns[0]}" AS value FROM {relation}'
    ).df()

    df_right = con.sql(
        f'SELECT "{columns[1]}" AS value FROM {relation}'
    ).df()

    if df_left.empty or df_right.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    pad_x, pad_y, pad_title = pads

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, sharex=sharex, sharey=sharey,
    )

    for ax, df, title, xlabel in [
        (ax_left, df_left, titles[0], xlabels[0]),
        (ax_right, df_right, titles[1], xlabels[1]),
    ]:
        values = df["value"].to_numpy()

        if log_x:
            values = values[values > 0]

        ax.hist(
            values,
            bins=bins,
            color=bar_color,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_width,
            alpha=bar_alpha,
        )

        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        _style_axes(ax, tick_size)

    ax_left.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)
    _hide_yaxis(ax_right)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax_left, ax_right), (df_left, df_right)


def plot_categorical_distribution(
    con,
    relation,
    column,
    display_name=None,
    level_mapping=None,
    title=None,
    xlabel=None,
    ylabel="Porcentaje (%)",
    sort_descending=True,
    show_counts=True,
    n=None,
    ylim=(0, 100),
    y_step=10,
    x_step=None,
    ax=None,
    bar_colors=None,
    bar_color="black",
    bar_edge_color="black",
    bar_edge_width=1.0,
    bar_width=0.6,
    value_size=10,
    value_offset=3.0,
    value_color="black",
    label_size=10,
    title_size=12,
    tick_size=9,
    title_loc="center",
    pads=(9, 9, 12),
    show_grid=False,
    figsize=(5, 5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the distribution of a categorical column from a DuckDB relation.

    Executes a single aggregate query against ``relation`` to compute the
    absolute count and the relative frequency of each level of ``column``.
    The relative frequency is computed as ``count * 100 / n`` when ``n``
    is provided, or as ``count * 100 / total`` when ``n`` is ``None``.
    The custom denominator is useful when the relation passed has been
    filtered (for example, to the top-N levels of a high-cardinality
    column) but the percentages should still be expressed relative to a
    broader reference population. Levels are optionally sorted by
    descending percentage, and the percentages are drawn as vertical bars.
    The absolute counts are annotated as text just above each bar. Bar
    fills can be a single color or a per-bar list of colors. Level labels
    are taken from ``level_mapping`` when provided, otherwise the raw
    level values are used. Applies a consistent, publication-ready style
    (thin black spines). The figure is optionally passed to
    ``render_figure`` for inline rendering.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table, view, or a fully qualified
            relation (e.g. ``"train"`` or ``"read_parquet('path.parquet')"``).
        column (str): Name of the categorical column to summarize.
        display_name (str or None): Human-readable name used as the
            x-axis label when ``xlabel`` is ``None``. Defaults to
            ``None``, in which case the raw ``column`` name is used.
        level_mapping (dict or None): Optional mapping from raw level
            values to display labels (e.g. ``{0: "No hubo clic (0)",
            1: "Hubo clic (1)"}``). When ``None``, the raw level values
            are shown as strings. Defaults to ``None``.
        title (str or None): Title of the plot. When ``None``, no title
            is drawn. Defaults to ``None``.
        xlabel (str or None): X-axis label. When ``None``, ``display_name``
            (or ``column`` if ``display_name`` is also ``None``) is used.
            Defaults to ``None``.
        ylabel (str): Y-axis label for the primary axis (percentages).
            Defaults to ``"Porcentaje (%)"``.
        sort_descending (bool): If ``True``, levels are sorted by
            descending percentage. Defaults to ``True``.
        show_counts (bool): If ``True``, the absolute counts are
            annotated as text just above each bar. Defaults to ``True``.
        n (float or None): Custom denominator used to compute the
            relative frequency as ``count * 100 / n``. When ``None``, the
            denominator is the sum of all counts in ``relation``, so the
            bars sum to 100%. When provided, the bars sum to
            ``100 * sum(counts) / n``, which is useful when ``relation``
            has been filtered. Must be positive when provided. Defaults
            to ``None``.
        ylim (tuple of float): Y-axis limits for the primary axis.
            Defaults to ``(0, 100)`` so that percentages are always read
            on the same scale across figures.
        y_step (float or None): Spacing between major ticks on the
            y-axis. When ``None``, matplotlib's default locator is used.
            Defaults to 10.
        x_step (float or None): Spacing between major ticks on the
            x-axis. When ``None``, one tick per bar is used. Defaults to
            ``None``.
        ax (matplotlib.axes.Axes or None): Existing axes to draw on. When
            ``None``, a new figure and axes are created using ``figsize``.
            Defaults to ``None``.
        bar_colors (list of str or None): Optional list of colors, one
            per bar in the order the bars are drawn. Must have the same
            length as the number of levels in ``column``. When ``None``,
            ``bar_color`` is used for every bar. Defaults to ``None``.
        bar_color (str): Fill color of the bars when ``bar_colors`` is
            ``None``. Defaults to ``"black"``.
        bar_edge_color (str): Edge color of the bars. Defaults to
            ``"black"``.
        bar_edge_width (float): Edge line width of the bars. Defaults to
            1.0.
        bar_width (float): Relative width of the bars in ``(0, 1]``.
            Defaults to 0.6.
        value_size (float): Font size of the annotation above each bar.
            Defaults to 10.
        value_offset (float): Vertical offset, in percentage points, of
            the annotation above the bar top. Defaults to 3.0.
        value_color (str): Color of the count annotation text. Defaults
            to ``"black"``.
        label_size (float): Font size for the axis labels. Defaults to
            10.
        title_size (float): Font size for the title. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 9.
        title_loc (str): Title alignment: ``"left"``, ``"center"`` or
            ``"right"``. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis label, the y-axis label and the title,
            respectively. Defaults to ``(9, 9, 12)``.
        show_grid (bool): If ``True``, a dotted horizontal grid is drawn
            behind the bars. Defaults to ``False``.
        figsize (tuple of float): Figure size in inches. Only used when
            ``ax`` is ``None``. Defaults to ``(5, 5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax, df_dist)`` where ``fig`` is the matplotlib
        figure, ``ax`` is the primary ``Axes``, and ``df_dist`` is the
        summary DataFrame with columns ``[column, "count", "percentage",
        "label"]``, already sorted according to ``sort_descending``.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if ``relation``,
            ``column`` or ``display_name`` is not a string, if
            ``level_mapping`` is not a dict or ``None``, if ``ax`` is
            provided but is not an ``Axes``, if ``bar_colors`` is not a
            list or ``None``, or if ``figsize`` is not a tuple/list of
            length 2.
        ValueError: If ``bar_width`` is not in ``(0, 1]``, if ``n`` is not
            ``None`` and not positive, if ``pads`` does not contain three
            elements, if ``ylim`` is not a length-2 sequence, if
            ``y_step`` or ``x_step`` is not a positive number, if
            ``bar_colors`` length does not match the number of levels, or
            if the query returns no rows.
    """
    import duckdb
    from matplotlib.axes import Axes

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(column, str):
        raise TypeError("'column' must be a string.")

    if display_name is not None and not isinstance(display_name, str):
        raise TypeError("'display_name' must be a string or None.")

    if level_mapping is not None and not isinstance(level_mapping, dict):
        raise TypeError("'level_mapping' must be a dict or None.")

    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("'ax' must be a matplotlib Axes or None.")

    if bar_colors is not None and not isinstance(bar_colors, list):
        raise TypeError("'bar_colors' must be a list of color strings or None.")

    if not isinstance(bar_width, (int, float)) or not (0 < bar_width <= 1):
        raise ValueError("'bar_width' must be a number in the interval (0, 1].")

    if n is not None and (
        not isinstance(n, (int, float))
        or isinstance(n, bool)
        or n <= 0
    ):
        raise ValueError("'n' must be None or a positive number.")

    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_x_step(x_step)
    _validate_pads(pads)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    if n is None:
        denominator_expr = "SUM(COUNT(*)) OVER ()"
    else:
        denominator_expr = str(n)

    query = f"""
        SELECT
            "{column}" AS level,
            COUNT(*) AS count,
            COUNT(*) * 100.0 / {denominator_expr} AS percentage
        FROM {relation}
        GROUP BY "{column}"
    """
    df_dist = con.sql(query).df()

    if df_dist.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    if sort_descending:
        df_dist = (
            df_dist
            .sort_values("percentage", ascending=False)
            .reset_index(drop=True)
        )

    if level_mapping is not None:
        df_dist["label"] = (
            df_dist["level"]
            .map(level_mapping)
            .fillna(df_dist["level"].astype(str))
        )
    else:
        df_dist["label"] = df_dist["level"].astype(str)

    n_bars = len(df_dist)

    if bar_colors is not None and len(bar_colors) != n_bars:
        raise ValueError(
            f"'bar_colors' must have one color per bar: expected "
            f"{n_bars}, got {len(bar_colors)}."
        )

    colors = bar_colors if bar_colors is not None else [bar_color] * n_bars

    pad_x, pad_y, pad_title = pads

    if xlabel is None:
        xlabel = display_name if display_name is not None else column

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)

    x_positions = np.arange(n_bars)
    heights = df_dist["percentage"].to_numpy()
    counts = df_dist["count"].to_numpy()

    ax.bar(
        x_positions,
        heights,
        width=bar_width,
        color=colors,
        edgecolor=bar_edge_color,
        linewidth=bar_edge_width,
    )

    if show_counts:
        for x, h, c in zip(x_positions, heights, counts):
            ax.text(
                x, h + value_offset, f"{int(c):,}",
                ha="center", va="bottom",
                fontsize=value_size, color=value_color,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_dist["label"], fontsize=tick_size)

    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)
    ax.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)

    ax.set_ylim(*ylim)

    _style_axes(ax, tick_size)
    _apply_yaxis_settings(ax, ylim=ylim, y_step=y_step)

    if x_step is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(x_step))

    if show_grid:
        ax.yaxis.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

    if created_fig:
        plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax, df_dist


def plot_categorical_distribution_pair(
    con,
    relation,
    columns,
    display_names=None,
    level_mappings=None,
    titles=("", ""),
    xlabels=None,
    ylabel="Porcentaje (%)",
    sort_descending=True,
    show_counts=True,
    n=None,
    ylim=(0, 100),
    y_step=10,
    x_step=None,
    bar_colors=None,
    bar_color="black",
    bar_edge_color="black",
    bar_edge_width=1.0,
    bar_width=0.6,
    value_size=10,
    value_offset=3.0,
    value_color="black",
    label_size=10,
    title_size=12,
    tick_size=9,
    title_loc="center",
    pads=(9, 9, 12),
    show_grid=False,
    sharey=True,
    figsize=(10, 5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot two categorical distributions side by side.

    Creates a figure with two horizontal subplots and delegates each one
    to :func:`plot_categorical_distribution`. Per-panel parameters can be
    passed as a single value (applied to both panels) or as a 2-element
    tuple (one value per panel). The y-axis is shared by default so that
    bar heights can be compared directly across panels, and the y-axis of
    the right panel is hidden (ticks and labels) so that only the left
    panel carries the shared scale.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str or tuple of 2 str): Relation name, shared by both
            panels when a single string is passed, or specified per panel
            when a 2-element tuple is used.
        columns (tuple or list of 2 str): Names of the categorical columns
            to plot in the left and right panels, respectively.
        display_names (str, tuple of 2 str, or None): Human-readable names
            used as x-axis labels when ``xlabels`` is ``None``.
        level_mappings (dict, tuple of 2 dict, or None): Optional mapping
            from raw level values to display labels, per panel.
        titles (tuple or list of 2 str): Titles for the left and right
            panels. Defaults to ``("", "")``.
        xlabels (str, tuple of 2 str, or None): X-axis labels for the
            panels. When ``None``, ``display_names`` is used.
        ylabel (str): Y-axis label, drawn only on the left panel.
            Defaults to ``"Porcentaje (%)"``.
        sort_descending (bool): Passed to each panel. Defaults to ``True``.
        show_counts (bool): Passed to each panel. Defaults to ``True``.
        n (float, tuple of 2 float, or None): Custom denominator for the
            relative frequency, per panel. Defaults to ``None``.
        ylim (tuple of float): Shared y-axis limits. Defaults to
            ``(0, 100)``.
        y_step (float or None): Spacing between major y-axis ticks.
            Defaults to 10.
        x_step (float or None): Spacing between major x-axis ticks.
            Defaults to ``None``.
        bar_colors (tuple of 2 (list or None), or None): Per-panel list of
            colors, one per bar. When ``None``, ``bar_color`` is used.
        bar_color (str or tuple of 2 str): Fill color of the bars, per
            panel. Defaults to ``"black"``.
        bar_edge_color (str): Edge color of the bars. Defaults to
            ``"black"``.
        bar_edge_width (float): Edge line width of the bars. Defaults to
            1.0.
        bar_width (float): Relative width of the bars in ``(0, 1]``.
            Defaults to 0.6.
        value_size (float): Font size of the annotation above each bar.
            Defaults to 10.
        value_offset (float): Vertical offset of the annotation above the
            bar top. Defaults to 3.0.
        value_color (str): Color of the count annotation text. Defaults to
            ``"black"``.
        label_size (float): Font size for the axis labels. Defaults to 10.
        title_size (float): Font size for the panel titles. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 9.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        show_grid (bool): If ``True``, a dotted horizontal grid is drawn
            behind the bars in both panels. Defaults to ``False``.
        sharey (bool): If ``True``, both panels share the same y-axis.
            Defaults to ``True``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(10, 5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax_left, ax_right), [df_left, df_right])`` where
        ``fig`` is the matplotlib figure, the two axes correspond to the
        left and right panels, and the two DataFrames are the summaries
        returned by each call to ``plot_categorical_distribution``.

    Raises:
        TypeError: If ``columns`` is not a tuple or list of two strings,
            or if ``titles`` is not a tuple or list of two strings.
        ValueError: If any per-panel parameter is a tuple or list whose
            length is not 2.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(columns, (tuple, list)) or len(columns) != 2:
        raise TypeError("'columns' must be a tuple or list of two strings.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    if not isinstance(titles, (tuple, list)) or len(titles) != 2:
        raise TypeError("'titles' must be a tuple or list of two strings.")

    if not all(isinstance(t, str) for t in titles):
        raise TypeError("All elements of 'titles' must be strings.")

    relations = _to_pair(relation, "relation")
    display_names_pair = _to_pair(display_names, "display_names")
    level_mappings_pair = _to_pair(level_mappings, "level_mappings")
    xlabels_pair = _to_pair(xlabels, "xlabels")
    n_pair = _to_pair(n, "n")
    bar_colors_pair = _to_pair(bar_colors, "bar_colors")
    bar_color_pair = _to_pair(bar_color, "bar_color")

    _validate_figsize(figsize)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, sharey=sharey,
    )

    _, _, df_left = plot_categorical_distribution(
        con,
        relations[0],
        columns[0],
        display_name=display_names_pair[0],
        level_mapping=level_mappings_pair[0],
        title=titles[0],
        xlabel=xlabels_pair[0],
        ylabel=ylabel,
        sort_descending=sort_descending,
        show_counts=show_counts,
        n=n_pair[0],
        ylim=ylim,
        y_step=y_step,
        x_step=x_step,
        ax=ax_left,
        bar_colors=bar_colors_pair[0],
        bar_color=bar_color_pair[0],
        bar_edge_color=bar_edge_color,
        bar_edge_width=bar_edge_width,
        bar_width=bar_width,
        value_size=value_size,
        value_offset=value_offset,
        value_color=value_color,
        label_size=label_size,
        title_size=title_size,
        tick_size=tick_size,
        title_loc=title_loc,
        pads=pads,
        show_grid=show_grid,
        render=False,
    )

    _, _, df_right = plot_categorical_distribution(
        con,
        relations[1],
        columns[1],
        display_name=display_names_pair[1],
        level_mapping=level_mappings_pair[1],
        title=titles[1],
        xlabel=xlabels_pair[1],
        ylabel="",
        sort_descending=sort_descending,
        show_counts=show_counts,
        n=n_pair[1],
        ylim=ylim,
        y_step=y_step,
        x_step=x_step,
        ax=ax_right,
        bar_colors=bar_colors_pair[1],
        bar_color=bar_color_pair[1],
        bar_edge_color=bar_edge_color,
        bar_edge_width=bar_edge_width,
        bar_width=bar_width,
        value_size=value_size,
        value_offset=value_offset,
        value_color=value_color,
        label_size=label_size,
        title_size=title_size,
        tick_size=tick_size,
        title_loc=title_loc,
        pads=pads,
        show_grid=show_grid,
        render=False,
    )

    _hide_yaxis(ax_right)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax_left, ax_right), [df_left, df_right]


# ---------------------------------------------------------------------------
# Temporal Patterns
# ---------------------------------------------------------------------------

def plot_temporal_overview(
    con,
    relation="temporal_features",
    target="click",
    day_column="day",
    band_column="time_band",
    metric="ctr",
    band_order=None,
    titles=("By day of month", "By time band"),
    xlabels=("Day", "Time band"),
    ylabel=None,
    bar_color="black",
    bar_edge_color="black",
    bar_edge_width=1.0,
    bar_width=0.6,
    value_size=9,
    value_color="black",
    show_values=True,
    ylim=None,
    y_step=None,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(9, 9, 12),
    figsize=(10, 3.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a two-panel temporal overview of clicks, impressions or CTR.

    Creates a figure with two horizontal panels. The left panel shows the
    requested metric aggregated by day of the month. The right panel
    shows the same metric aggregated by the time band. Both panels use
    bar charts with a shared y-axis so that magnitudes can be compared
    directly. The y-axis of the right panel is hidden.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of a table or view containing the day, the
            time band and the target. Defaults to ``"temporal_features"``.
        target (str): Name of the binary target column. Defaults to
            ``"click"``.
        day_column (str): Column with the day of the month. Defaults to
            ``"day"``.
        band_column (str): Column with the time band. Defaults to
            ``"time_band"``.
        metric ({"clicks", "impressions", "ctr"}): Metric to compute in
            both panels. Defaults to ``"ctr"``.
        band_order (tuple of str or None): Order in which the time bands
            are plotted. When ``None``, the order returned by the query is
            preserved. Defaults to ``None``.
        titles (tuple of 2 str): Titles for the left and right panels.
            Defaults to ``("By day of month", "By time band")``.
        xlabels (tuple of 2 str): X-axis labels for both panels. Defaults
            to ``("Day", "Time band")``.
        ylabel (str or None): Y-axis label for the left panel. When
            ``None``, a default label is derived from ``metric``.
        bar_color (str): Fill color of the bars. Defaults to ``"black"``.
        bar_edge_color (str): Edge color of the bars. Defaults to
            ``"black"``.
        bar_edge_width (float): Edge line width of the bars. Defaults to
            1.0.
        bar_width (float): Relative width of the bars in ``(0, 1]``.
            Defaults to 0.6.
        value_size (float): Font size of the annotation above each bar.
            Defaults to 9.
        value_color (str): Color of the annotation text. Defaults to
            ``"black"``.
        show_values (bool): If ``True``, the value of each bar is
            annotated above it. Defaults to ``True``.
        ylim (tuple of float or None): Shared y-axis limits. When
            ``None``, matplotlib chooses the range. Defaults to ``None``.
        y_step (float or None): Spacing between major y-axis ticks.
            Defaults to ``None``.
        label_size (float): Font size for the axis labels. Defaults to 11.
        title_size (float): Font size for the panel titles. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(10, 3.5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax_left, ax_right), (df_day, df_band))`` where
        each DataFrame is the summary of the corresponding panel.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if any string
            argument has the wrong type, or if ``render`` is not a
            boolean.
        ValueError: If ``metric`` is not supported, if ``titles`` or
            ``xlabels`` does not have length 2, or if the queries return
            no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    valid_metrics = {"clicks", "impressions", "ctr"}
    if metric not in valid_metrics:
        raise ValueError(
            f"'metric' must be one of {sorted(valid_metrics)}, "
            f"got '{metric}'."
        )

    if not isinstance(titles, (tuple, list)) or len(titles) != 2:
        raise TypeError("'titles' must be a tuple or list of two strings.")

    if not isinstance(xlabels, (tuple, list)) or len(xlabels) != 2:
        raise TypeError("'xlabels' must be a tuple or list of two strings.")

    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_pads(pads)
    _validate_figsize(figsize)

    if not isinstance(show_values, bool):
        raise TypeError("'show_values' must be a boolean.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    if metric == "clicks":
        value_expr = f'SUM(CASE WHEN "{target}" = 1 THEN 1 ELSE 0 END)'
        default_ylabel = "Number of clicks"
        fmt_value = lambda v: f"{int(v):,}"
    elif metric == "impressions":
        value_expr = "COUNT(*)"
        default_ylabel = "Number of impressions"
        fmt_value = lambda v: f"{int(v):,}"
    else:
        value_expr = f'AVG("{target}") * 100.0'
        default_ylabel = "Click-through rate (%)"
        fmt_value = lambda v: f"{v:.2f}"

    if ylabel is None:
        ylabel = default_ylabel

    df_day = con.sql(f"""
        SELECT
            "{day_column}" AS level,
            {value_expr} AS value
        FROM {relation}
        GROUP BY "{day_column}"
        ORDER BY "{day_column}"
    """).df()

    df_band = con.sql(f"""
        SELECT
            "{band_column}" AS level,
            {value_expr} AS value
        FROM {relation}
        GROUP BY "{band_column}"
    """).df()

    if df_day.empty or df_band.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    if band_order is not None:
        df_band["order"] = df_band["level"].apply(
            lambda x: band_order.index(x) if x in band_order else len(band_order)
        )
        df_band = (
            df_band
            .sort_values("order")
            .drop(columns="order")
            .reset_index(drop=True)
        )

    all_values = np.concatenate([
        df_day["value"].to_numpy(),
        df_band["value"].to_numpy(),
    ])
    span = all_values.max() - all_values.min()
    value_offset = span * 0.02 if span > 0 else all_values.max() * 0.02

    pad_x, pad_y, pad_title = pads

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
    )

    for ax, df, title, xlabel in [
        (ax_left, df_day, titles[0], xlabels[0]),
        (ax_right, df_band, titles[1], xlabels[1]),
    ]:
        x_positions = np.arange(len(df))
        heights = df["value"].to_numpy()

        ax.bar(
            x_positions,
            heights,
            width=bar_width,
            color=bar_color,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_width,
        )

        if show_values:
            for x, h in zip(x_positions, heights):
                ax.text(
                    x, h + value_offset, fmt_value(h),
                    ha="center", va="bottom",
                    fontsize=value_size, color=value_color,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(df["level"].astype(str), fontsize=tick_size)
        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)
        _style_axes(ax, tick_size)

    ax_left.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)
    _apply_yaxis_settings(ax_left, ylim=ylim, y_step=y_step)
    _hide_yaxis(ax_right)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax_left, ax_right), (df_day, df_band)


def plot_ctr_by_hour(
    con,
    relation="temporal_features",
    target="click",
    hour_column="hour_of_day",
    title="Click-through rate by hour of day",
    xlabel="Hour of day",
    ylabel="Click-through rate (%)",
    line_color="black",
    line_width=1.2,
    marker="o",
    marker_size=5,
    ylim=None,
    y_step=None,
    x_step=2,
    label_size=11,
    title_size=12,
    tick_size=10,
    title_loc="center",
    pads=(9, 9, 12),
    figsize=(8, 3.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the click-through rate as a function of the hour of the day.

    Executes a single aggregate query against ``relation`` to compute the
    click-through rate for each value of ``hour_column``, expressed as a
    percentage. The result is drawn as a line with markers, preserving the
    cyclic order of the hours. Applies a consistent, publication-ready
    style (thin black spines). The figure is optionally passed to
    ``render_figure`` for inline rendering.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table or view containing both the hour
            column and the target. Defaults to ``"temporal_features"``.
        target (str): Name of the binary target column. Defaults to
            ``"click"``.
        hour_column (str): Name of the column containing the hour of the
            day, with integer values between 0 and 23. Defaults to
            ``"hour_of_day"``.
        title (str or None): Title of the plot. Defaults to
            ``"Click-through rate by hour of day"``.
        xlabel (str): X-axis label. Defaults to ``"Hour of day"``.
        ylabel (str): Y-axis label. Defaults to
            ``"Click-through rate (%)"``.
        line_color (str): Color of the line. Defaults to ``"black"``.
        line_width (float): Width of the line. Defaults to 1.2.
        marker (str): Marker style for each point. Defaults to ``"o"``.
        marker_size (float): Marker size. Defaults to 5.
        ylim (tuple of float or None): Y-axis limits. When ``None``,
            matplotlib chooses the range. Defaults to ``None``.
        y_step (float or None): Spacing between major y-axis ticks. When
            ``None``, matplotlib's default locator is used. Defaults to
            ``None``.
        x_step (int): Spacing between major x-axis ticks, in hours.
            Defaults to 2.
        label_size (float): Font size for the axis labels. Defaults to 11.
        title_size (float): Font size for the title. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 10.
        title_loc (str): Title alignment. Defaults to ``"center"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 12)``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 3.5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax, df_ctr)`` where ``fig`` is the matplotlib
        figure, ``ax`` is the ``Axes``, and ``df_ctr`` is the summary
        DataFrame with columns ``[hour_column, "ctr", "n"]``.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if any string
            argument has the wrong type, or if ``render`` is not a
            boolean.
        ValueError: If ``ylim`` is not ``None`` and not a length-2
            sequence, if ``y_step`` is not ``None`` and not positive, if
            ``x_step`` is not a positive integer, or if the query returns
            no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    for name, value in [
        ("relation", relation),
        ("target", target),
        ("hour_column", hour_column),
    ]:
        if not isinstance(value, str):
            raise TypeError(f"'{name}' must be a string.")

    if (
        not isinstance(x_step, int)
        or isinstance(x_step, bool)
        or x_step <= 0
    ):
        raise ValueError("'x_step' must be a positive integer.")

    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_pads(pads)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    query = f"""
        SELECT
            "{hour_column}" AS hour,
            AVG("{target}") * 100.0 AS ctr,
            COUNT(*) AS n
        FROM {relation}
        GROUP BY "{hour_column}"
        ORDER BY "{hour_column}"
    """
    df_ctr = con.sql(query).df()

    if df_ctr.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    pad_x, pad_y, pad_title = pads

    fig, ax = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)

    ax.plot(
        df_ctr["hour"],
        df_ctr["ctr"],
        color=line_color,
        linewidth=line_width,
        marker=marker,
        markersize=marker_size,
    )

    ax.set_xticks(range(0, 24, x_step))
    ax.set_xlabel(xlabel, fontsize=label_size, labelpad=pad_x)
    ax.set_ylabel(ylabel, fontsize=label_size, labelpad=pad_y)

    _style_axes(ax, tick_size)
    _apply_yaxis_settings(ax, ylim=ylim, y_step=y_step)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax, df_ctr


# ---------------------------------------------------------------------------
# Association & Correlation
# ---------------------------------------------------------------------------

def categorical_association_table(
    con,
    relation,
    columns,
    target="click",
    alpha=0.05,
    p_adjust_method=None,
    add_p_adj=False,
    min_effect_size=None,
    min_expected=1,
    max_frac_below_5=0.20,
    hide_empty_rows=False,
    column_titles=("Variable", "gl", "Chi-cuadrado", "p-valor", "Cramér's V"),
    p_adj_title="p-valor ajustado",
    decimals_chi2=3,
    decimals_v=3,
    sort_descending=True,
    render=True,
):
    """Test association between categorical columns and a binary target.

    For each column in ``columns``, builds a contingency table against
    ``target`` using DuckDB and computes the Pearson chi-squared statistic,
    the degrees of freedom, the p-value and Cramér's V effect size using
    SciPy. An optional multiple-testing correction is applied to the
    p-values when ``p_adjust_method`` is not ``None``. The significance
    filter uses the corrected p-values when a correction is active, and an
    optional effect-size threshold can further restrict the output. Rows
    are sorted by effect size and the resulting table is formatted in APA
    style and optionally rendered as centered HTML.

    Cochran's rule for the validity of the chi-squared approximation is
    verified in its original formulation. The assumption is considered
    satisfied when both conditions hold: no cell has an expected frequency
    below ``min_expected``, and the fraction of cells with expected
    frequency below 5 does not exceed ``max_frac_below_5``. Columns that
    violate either condition are flagged, and their chi-squared, p-value
    and Cramér's V are displayed as ``NA`` to avoid reporting statistically
    unreliable quantities. Flagged rows are kept in the output regardless
    of the significance and effect-size filters, so the assumption
    violation remains visible, unless ``hide_empty_rows`` is ``True``.
    Pass ``min_expected=None`` to disable the check entirely.

    Degrees of freedom follow the standard formula ``(r - 1)(c - 1)``,
    where ``r`` is the cardinality of the tested column and ``c`` is the
    number of levels of the target. For a binary target, ``c = 2`` and
    ``dof = r - 1``. Cramér's V is computed as ``sqrt(chi2 / (n * 1))``,
    which is the simplification of the general formula when one of the
    table dimensions equals 2.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of the source table or view.
        columns (list or tuple of str): Names of the categorical columns
            to test. Must contain at least one string.
        target (str): Name of the binary target column. Defaults to
            ``"click"``.
        alpha (float): Significance threshold. Columns with p-value
            greater than or equal to ``alpha`` are dropped. Defaults to
            0.05.
        p_adjust_method (str or None): Multiple-testing correction to
            apply to the p-values. Supported methods are ``"bonferroni"``
            and ``"holm"``. When ``None``, no correction is applied.
            Defaults to ``None``.
        add_p_adj (bool): If ``True`` and ``p_adjust_method`` is not
            ``None``, both the raw and the adjusted p-value columns are
            shown. If ``False``, only the adjusted p-value column is
            shown, relabeled with ``p_adj_title``. Ignored when
            ``p_adjust_method`` is ``None``. Defaults to ``False``.
        min_effect_size (float or None): Minimum Cramér's V value required
            for a column to be kept in the output. When ``None``, no
            effect-size filter is applied. Must be in ``[0, 1]`` when
            provided. Defaults to ``None``.
        min_expected (float or None): Minimum expected frequency allowed
            in any cell of the contingency table, following Cochran's
            first condition. When ``None``, the check is disabled. Must be
            positive when provided. Defaults to 1.
        max_frac_below_5 (float or None): Maximum fraction of cells
            allowed to have an expected frequency below 5, following
            Cochran's second condition. When ``None``, the fraction check
            is disabled. Must be in ``[0, 1]`` when provided. Defaults to
            0.20.
        hide_empty_rows (bool): If ``True``, rows whose contingency table
            does not satisfy Cochran's assumption are removed from the
            output. If ``False``, those rows are kept and their statistics
            are displayed as ``NA``. Defaults to ``False``.
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(variable, dof, chi2, p, v)``. Must
            contain exactly five strings. Defaults to
            ``("Variable", "gl", "Chi-cuadrado", "p-valor", "Cramér's V")``.
        p_adj_title (str): Header for the adjusted p-value column when a
            correction is applied. Defaults to ``"p-valor ajustado"``.
        decimals_chi2 (int): Number of decimals used to format the
            chi-squared column. Defaults to 3.
        decimals_v (int): Number of decimals used to format Cramér's V.
            Defaults to 3.
        sort_descending (bool): If ``True``, rows are sorted by Cramér's V
            in descending order. Defaults to ``True``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The filtered and sorted table with numeric
        columns, plus the internal ``_min_expected``, ``_frac_below_5``
        and ``assumption_ok`` columns. When a correction is applied, an
        additional ``"p_adjusted"`` column is included.

    Raises:
        TypeError: If any argument has the wrong type.
        ValueError: If ``columns`` is empty, if ``alpha`` is not in
            ``(0, 1)``, if ``min_effect_size`` is not ``None`` and not in
            ``[0, 1]``, if ``min_expected`` is not ``None`` and not
            positive, if ``max_frac_below_5`` is not ``None`` and not in
            ``[0, 1]``, if ``column_titles`` does not contain exactly
            five elements, if ``p_adjust_method`` is not a supported
            string, if ``add_p_adj`` is ``True`` while
            ``p_adjust_method`` is ``None``, or if any of the decimal
            counts is negative.
    """
    import duckdb
    from scipy.stats import chi2_contingency

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(columns, (list, tuple)):
        raise TypeError("'columns' must be a list or tuple of strings.")

    if len(columns) == 0:
        raise ValueError("'columns' must contain at least one column name.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    if not isinstance(target, str):
        raise TypeError("'target' must be a string.")

    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not (0 < alpha < 1)
    ):
        raise ValueError("'alpha' must be a number in (0, 1).")

    if p_adjust_method is not None:
        if not isinstance(p_adjust_method, str):
            raise TypeError("'p_adjust_method' must be a string or None.")
        if p_adjust_method not in _VALID_P_ADJUST_METHODS:
            raise ValueError(
                f"Unsupported p-adjustment method: '{p_adjust_method}'. "
                f"Valid methods: {sorted(_VALID_P_ADJUST_METHODS)}."
            )

    if not isinstance(add_p_adj, bool):
        raise TypeError("'add_p_adj' must be a boolean.")

    if add_p_adj and p_adjust_method is None:
        raise ValueError(
            "'add_p_adj' requires a non-None 'p_adjust_method'."
        )

    if min_effect_size is not None and (
        not isinstance(min_effect_size, (int, float))
        or isinstance(min_effect_size, bool)
        or not (0 <= min_effect_size <= 1)
    ):
        raise ValueError(
            "'min_effect_size' must be None or a number in [0, 1]."
        )

    if min_expected is not None and (
        not isinstance(min_expected, (int, float))
        or isinstance(min_expected, bool)
        or min_expected <= 0
    ):
        raise ValueError(
            "'min_expected' must be None or a positive number."
        )

    if max_frac_below_5 is not None and (
        not isinstance(max_frac_below_5, (int, float))
        or isinstance(max_frac_below_5, bool)
        or not (0 <= max_frac_below_5 <= 1)
    ):
        raise ValueError(
            "'max_frac_below_5' must be None or a number in [0, 1]."
        )

    if not isinstance(hide_empty_rows, bool):
        raise TypeError("'hide_empty_rows' must be a boolean.")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of five strings."
        )

    if len(column_titles) != 5:
        raise ValueError(
            "'column_titles' must contain exactly 5 elements "
            "(variable, dof, chi2, p, v)."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(p_adj_title, str):
        raise TypeError("'p_adj_title' must be a string.")

    for name, value in [
        ("decimals_chi2", decimals_chi2),
        ("decimals_v", decimals_v),
    ]:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var, col_dof, col_chi2, col_p, col_v = column_titles

    n_total = int(
        con.sql(f"SELECT COUNT(*) AS n FROM {relation}").df()["n"].iloc[0]
    )

    results = []
    for col in columns:
        query = f"""
            SELECT
                "{col}" AS level,
                "{target}" AS target,
                COUNT(*) AS n
            FROM {relation}
            GROUP BY "{col}", "{target}"
        """
        df_long = con.sql(query).df()

        if df_long.empty:
            continue

        contingency = (
            df_long
            .pivot(index="level", columns="target", values="n")
            .fillna(0)
            .to_numpy()
        )

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2_stat, p_value, dof, _ = chi2_contingency(
            contingency, correction=False
        )

        v_cramer = float(np.sqrt(chi2_stat / n_total))

        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / n_total

        min_expected_value = float(expected.min())
        frac_below_5 = float((expected < 5).mean())

        results.append({
            col_var: col,
            col_dof: int(dof),
            col_chi2: float(chi2_stat),
            col_p: float(p_value),
            col_v: v_cramer,
            "_min_expected": min_expected_value,
            "_frac_below_5": frac_below_5,
        })

    df_assoc = pd.DataFrame(results)

    if df_assoc.empty:
        if render:
            print("No columns produced a valid contingency table.")
        return df_assoc

    if min_expected is None and max_frac_below_5 is None:
        df_assoc["assumption_ok"] = True
    else:
        cond_min = (
            df_assoc["_min_expected"] >= min_expected
            if min_expected is not None
            else True
        )
        cond_frac = (
            df_assoc["_frac_below_5"] <= max_frac_below_5
            if max_frac_below_5 is not None
            else True
        )
        df_assoc["assumption_ok"] = cond_min & cond_frac

    if hide_empty_rows:
        df_assoc = df_assoc[df_assoc["assumption_ok"]].reset_index(drop=True)

    if df_assoc.empty:
        if render:
            print("No columns satisfied Cochran's assumption.")
        return df_assoc

    if p_adjust_method == "bonferroni":
        n_tests = len(df_assoc)
        df_assoc["p_adjusted"] = (df_assoc[col_p] * n_tests).clip(upper=1.0)
        p_filter_col = "p_adjusted"
    elif p_adjust_method == "holm":
        df_assoc["p_adjusted"] = _holm_bonferroni(
            df_assoc[col_p].to_numpy()
        )
        p_filter_col = "p_adjusted"
    else:
        p_filter_col = col_p

    keep_mask = df_assoc[p_filter_col] < alpha
    keep_mask = keep_mask | ~df_assoc["assumption_ok"]
    df_assoc = df_assoc[keep_mask].reset_index(drop=True)

    if min_effect_size is not None:
        keep_mask = df_assoc[col_v] >= min_effect_size
        keep_mask = keep_mask | ~df_assoc["assumption_ok"]
        df_assoc = df_assoc[keep_mask].reset_index(drop=True)

    if df_assoc.empty:
        if render:
            print("No columns passed the significance and effect-size filters.")
        return df_assoc

    if sort_descending:
        df_assoc = (
            df_assoc
            .sort_values(
                ["assumption_ok", col_v],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

    def _fmt_or_na(value, formatter, ok):
        return formatter(value) if ok else "NA"

    chi2_str = df_assoc.apply(
        lambda r: _fmt_or_na(
            r[col_chi2],
            lambda x: f"{x:,.{decimals_chi2}f}",
            r["assumption_ok"],
        ),
        axis=1,
    )
    v_str = df_assoc.apply(
        lambda r: _fmt_or_na(
            r[col_v],
            lambda x: _format_apa_effect(x, decimals_v),
            r["assumption_ok"],
        ),
        axis=1,
    )
    p_raw_str = df_assoc.apply(
        lambda r: _fmt_or_na(
            r[col_p],
            _format_apa_pvalue,
            r["assumption_ok"],
        ),
        axis=1,
    )

    if p_adjust_method is None:
        df_display = pd.DataFrame({
            col_var: df_assoc[col_var],
            col_dof: df_assoc[col_dof],
            col_chi2: chi2_str,
            col_p: p_raw_str,
            col_v: v_str,
        })
    elif add_p_adj:
        p_adj_str = df_assoc.apply(
            lambda r: _fmt_or_na(
                r["p_adjusted"],
                _format_apa_pvalue,
                r["assumption_ok"],
            ),
            axis=1,
        )
        df_display = pd.DataFrame({
            col_var: df_assoc[col_var],
            col_dof: df_assoc[col_dof],
            col_chi2: chi2_str,
            col_p: p_raw_str,
            p_adj_title: p_adj_str,
            col_v: v_str,
        })
    else:
        p_adj_str = df_assoc.apply(
            lambda r: _fmt_or_na(
                r["p_adjusted"],
                _format_apa_pvalue,
                r["assumption_ok"],
            ),
            axis=1,
        )
        df_display = pd.DataFrame({
            col_var: df_assoc[col_var],
            col_dof: df_assoc[col_dof],
            col_chi2: chi2_str,
            p_adj_title: p_adj_str,
            col_v: v_str,
        })

    styled = (
        df_display.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return df_assoc


def plot_spearman_heatmap(
    con,
    relation="numeric_features",
    columns=None,
    figsize=(8, 5),
    title=None,
    annot_size=8,
    cmap="RdBu_r",
    decimals=2,
    label_size=10,
    title_size=11,
    tick_size=10,
    pads=(9, 9, 14),
    cbar_shrink=0.7,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a lower-triangle Spearman correlation heatmap from a DuckDB relation.

    Draws the Spearman rank-correlation matrix of the given numeric columns
    as a lower-triangle heatmap with annotated coefficients. The rank
    transformation is computed entirely inside DuckDB using window
    functions, and the correlations are computed with DuckDB's ``CORR``
    aggregate. Only the resulting matrix is transferred to pandas, so the
    underlying relation is never loaded into memory.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of a table or view containing the numeric
            columns. Defaults to ``"numeric_features"``.
        columns (list of str or None): Names of the numeric columns to
            include in the correlation matrix. When ``None``, a default
            set of five columns is used. Defaults to ``None``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 5)``.
        title (str or None): Title of the chart. When ``None``, no title
            is drawn. Defaults to ``None``.
        annot_size (float): Font size of the coefficients inside the
            cells. Defaults to 8.
        cmap (str): Colormap used for the heatmap. Defaults to
            ``"RdBu_r"``.
        decimals (int): Number of decimals used for the coefficients.
            Defaults to 2.
        label_size (float): Font size of the colorbar label. Defaults to
            10.
        title_size (float): Font size of the title. Defaults to 11.
        tick_size (float): Font size of the axis tick labels. Defaults
            to 10.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``.
            Defaults to ``(9, 9, 14)``.
        cbar_shrink (float): Fraction by which to shrink the colorbar.
            Defaults to 0.7.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format. Defaults to ``"svg"``.
        dpi (int): Resolution when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax, corr_matrix)`` where ``fig`` is the matplotlib
        figure, ``ax`` is the ``Axes``, and ``corr_matrix`` is the
        correlation matrix as a pandas DataFrame.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if ``relation``
            is not a string, if ``columns`` is not a list of strings or
            ``None``, or if ``render`` is not a boolean.
        ValueError: If ``columns`` is empty after resolution or if the
            query returns no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if columns is None:
        columns = [
            "day",
            "hour_sin",
            "hour_cos",
            "log_freq_device_id",
            "log_freq_device_ip",
        ]

    if not isinstance(columns, (list, tuple)) or len(columns) == 0:
        raise ValueError("'columns' must be a non-empty list of strings.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    _validate_pads(pads)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    n = len(columns)

    rank_exprs = ",\n            ".join(
        f'RANK() OVER (ORDER BY "{c}") AS r{i}'
        for i, c in enumerate(columns)
    )

    corr_exprs = []
    for i in range(n):
        for j in range(i + 1):
            corr_exprs.append(f"CORR(r{i}, r{j}) AS c{i}_{j}")
    corr_exprs = ",\n            ".join(corr_exprs)

    query = f"""
        WITH ranked AS (
            SELECT
            {rank_exprs}
            FROM {relation}
        )
        SELECT
            {corr_exprs}
        FROM ranked
    """
    row = con.sql(query).df().iloc[0]

    if row.isna().all():
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1):
            corr[i, j] = row[f"c{i}_{j}"]
            corr[j, i] = row[f"c{i}_{j}"]

    corr_matrix = pd.DataFrame(corr, index=columns, columns=columns)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    _pad_x, pad_y, pad_title = pads

    fig, ax = plt.subplots(figsize=figsize)

    masked = np.ma.array(corr_matrix.to_numpy(), mask=mask)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    im = ax.imshow(
        masked,
        cmap=cmap_obj,
        vmin=-1, vmax=1,
        aspect="auto",
    )

    for i in range(n):
        for j in range(n):
            if i > j:
                ax.text(
                    j, i, f"{corr_matrix.iloc[i, j]:.{decimals}f}",
                    ha="center", va="center",
                    fontsize=annot_size, color="black",
                )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=tick_size)
    ax.set_yticklabels(columns, fontsize=tick_size)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    if title is not None:
        ax.set_title(title, fontsize=title_size, pad=pad_title)

    cbar = fig.colorbar(im, ax=ax, shrink=cbar_shrink)
    cbar.outline.set_linewidth(1.2)
    cbar.set_label(
        "Spearman's ρ",
        labelpad=pad_y,
        fontsize=label_size,
    )

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax, corr_matrix


# ---------------------------------------------------------------------------
# Cardinality & Encoding Strategy
# ---------------------------------------------------------------------------

def categorical_cardinality_table(
    con,
    relation,
    columns,
    column_titles=("Variable", "Unique Values"),
    sort_descending=True,
    decimals=0,
    render=True,
):
    """Summarize the cardinality of categorical columns from a DuckDB relation.

    Builds a small table with one row per column in ``columns``, reporting
    the column name and the number of distinct values it contains. The
    distinct counts are computed entirely inside DuckDB with a single
    aggregate query, so the underlying data is never loaded into memory.
    Rows are sorted by cardinality and the resulting table is split
    vertically into two halves that are placed side by side, yielding a
    compact two-panel layout with four columns and ``ceil(n/2)`` rows.
    If the number of columns is odd, the last row of the second panel is
    padded with ``None``. Applies a consistent, publication-ready style
    (centered cells, thin black borders, left-aligned variable columns).
    Optionally renders the styled table as centered HTML (useful in
    Jupyter notebooks).

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table, view, or a fully qualified
            relation (e.g. ``"train"`` or ``"read_parquet('path.parquet')"``).
        columns (list or tuple of str): Names of the categorical columns
            whose cardinality will be computed. Must contain at least one
            string.
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(variable, cardinality)``. Must contain
            exactly two strings. Defaults to
            ``("Variable", "Unique Values")``.
        sort_descending (bool): If ``True``, rows are sorted by cardinality
            in descending order. Defaults to ``True``.
        decimals (int): Number of decimals used to format the cardinality
            column. Since counts are integers, the default of ``0`` is
            usually what you want. Defaults to 0.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) two-panel results table
        with four columns (the two column titles duplicated) and
        ``ceil(n/2)`` rows. The styled version is built internally and,
        when ``render=True``, rendered as HTML — it is not returned.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if ``relation``
            is not a string, if ``columns`` is not a list or tuple of
            strings, if ``column_titles`` is not a tuple/list of two
            strings, if ``decimals`` is not an int, or if ``render`` is
            not a boolean.
        ValueError: If ``columns`` is empty, if ``column_titles`` does not
            contain exactly two elements, or if ``decimals`` is negative.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(columns, (list, tuple)):
        raise TypeError("'columns' must be a list or tuple of strings.")

    if len(columns) == 0:
        raise ValueError("'columns' must contain at least one column name.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of two strings."
        )

    if len(column_titles) != 2:
        raise ValueError(
            "'column_titles' must contain exactly 2 elements "
            "(variable, cardinality)."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("'decimals' must be a non-negative integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var, col_card = column_titles

    cardinality_exprs = ", ".join(
        f'COUNT(DISTINCT "{name}") AS "{name}"' for name in columns
    )
    row = con.sql(f"SELECT {cardinality_exprs} FROM {relation}").df().iloc[0]

    df_card = pd.DataFrame(
        {
            col_var: list(columns),
            col_card: [int(row[name]) for name in columns],
        }
    )

    if sort_descending:
        df_card = (
            df_card
            .sort_values(col_card, ascending=False)
            .reset_index(drop=True)
        )

    n_rows = len(df_card)
    half = (n_rows + 1) // 2

    first_half = df_card.iloc[:half].reset_index(drop=True)
    second_half = df_card.iloc[half:].reset_index(drop=True)

    if len(second_half) < half:
        padding = pd.DataFrame(
            [[""] * df_card.shape[1]] * (half - len(second_half)),
            columns=df_card.columns,
        )
        second_half = pd.concat([second_half, padding], ignore_index=True)

    combined = pd.concat([first_half, second_half], axis=1)

    styled = (
        combined.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1, 3)))
        .format(
            lambda x: _smart_format(x, f"{{:,.{decimals}f}}"),
            subset=[col_card],
        )
    )

    if render:
        _render_styled(styled)

    return combined


def categorical_coverage_table(
    con,
    relation,
    columns,
    thresholds=(0.90, 0.80, 0.70, 0.60),
    column_titles=("Variable", "90%", "80%", "70%", "60%"),
    sort_ascending=True,
    render=True,
):
    """Summarize the number of categories needed to reach coverage thresholds.

    For each column in ``columns``, sorts the categories by descending
    frequency and computes the minimum number of categories required to
    cover each of the given thresholds. The result is a compact table
    where a small value indicates a concentrated variable (few categories
    cover most of the data) and a large value indicates a spread-out
    variable. All aggregates are computed inside DuckDB with a single
    query per column, so the underlying data is never loaded into memory.

    The output is intended as a diagnostic to decide the encoding
    strategy for categorical variables. A variable that reaches 90% with
    one or two categories is a candidate for dichotomization. A variable
    that requires dozens of categories to reach 80% has a long-tailed
    distribution.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table, view, or a fully qualified
            relation (e.g. ``"train"`` or ``"read_parquet('path.parquet')"``).
        columns (list or tuple of str): Names of the categorical columns
            to summarize. Must contain at least one string.
        thresholds (tuple of float): Coverage thresholds in the interval
            ``(0, 1)``, ordered from highest to lowest. Each becomes a
            column in the output. Defaults to ``(0.90, 0.80, 0.70, 0.60)``.
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(variable, ...thresholds)``. Must contain
            one string per threshold plus one for the variable. Defaults
            to ``("Variable", "90%", "80%", "70%", "60%")``.
        sort_ascending (bool): If ``True``, rows are sorted by the first
            threshold column in ascending order, so the most concentrated
            variables appear first. If ``False``, the order is descending.
            Defaults to ``True``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The summary table with numeric columns
        ``[variable, n90, n80, n70, n60]``.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if ``relation``
            is not a string, if ``columns`` is not a list or tuple of
            strings, if ``column_titles`` is not a tuple/list of strings,
            or if ``render`` is not a boolean.
        ValueError: If ``columns`` is empty, if ``thresholds`` is empty,
            if any threshold is not in ``(0, 1)``, if ``column_titles``
            length does not match ``1 + len(thresholds)``, or if the
            query returns no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(columns, (list, tuple)):
        raise TypeError("'columns' must be a list or tuple of strings.")

    if len(columns) == 0:
        raise ValueError("'columns' must contain at least one column name.")

    if not all(isinstance(c, str) for c in columns):
        raise TypeError("All elements of 'columns' must be strings.")

    if not isinstance(thresholds, (tuple, list)) or len(thresholds) == 0:
        raise ValueError("'thresholds' must be a non-empty tuple or list.")

    if not all(
        isinstance(t, (int, float))
        and not isinstance(t, bool)
        and 0 < t < 1
        for t in thresholds
    ):
        raise ValueError("All thresholds must be numbers in (0, 1).")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of strings."
        )

    if len(column_titles) != 1 + len(thresholds):
        raise ValueError(
            "'column_titles' must contain one string per threshold plus "
            "one for the variable."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var = column_titles[0]
    col_thresholds = list(column_titles[1:])

    threshold_exprs = ",\n                ".join(
        f"MIN(rn) FILTER (WHERE cum_pct >= {t}) AS t{i}"
        for i, t in enumerate(thresholds)
    )

    results = []
    for col in columns:
        query = f"""
            WITH counts AS (
                SELECT "{col}" AS level, COUNT(*) AS n
                FROM {relation}
                GROUP BY "{col}"
            ),
            ranked AS (
                SELECT
                    ROW_NUMBER() OVER (ORDER BY n DESC) AS rn,
                    SUM(n) OVER (ORDER BY n DESC) * 1.0
                        / SUM(n) OVER () AS cum_pct
                FROM counts
            )
            SELECT
                {threshold_exprs}
            FROM ranked
        """
        row = con.sql(query).df().iloc[0]

        result_row = {col_var: col}
        for i, title in enumerate(col_thresholds):
            val = row[f"t{i}"]
            result_row[title] = int(val) if pd.notna(val) else None
        results.append(result_row)

    df_cov = pd.DataFrame(results)

    if df_cov.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    sort_col = col_thresholds[0]
    df_cov = (
        df_cov
        .sort_values(sort_col, ascending=sort_ascending, na_position="last")
        .reset_index(drop=True)
    )

    styled = (
        df_cov.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return df_cov


def show_encoding_strategy_table(render=True):
    """Display the suggested encoding strategy summary table.

    Consolidates the encoding decisions derived from the cardinality,
    concentration and association analyses into a single table. Within
    each strategy group, variables are ordered alphabetically. The rows
    are split vertically into two halves that are placed side by side,
    yielding a compact two-panel layout. If the number of rows is odd,
    the last row of the second panel is padded with empty strings.

    Args:
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The two-panel summary table with six columns.
    """
    specifications = [
        # Dicotomizar
        {"variable": "C1",               "strategy": "Dicotomizar",            "n": None},
        {"variable": "C15",              "strategy": "Dicotomizar",            "n": None},
        {"variable": "C16",              "strategy": "Dicotomizar",            "n": None},
        {"variable": "device_conn_type", "strategy": "Dicotomizar",            "n": None},
        {"variable": "device_type",      "strategy": "Dicotomizar",            "n": None},
        # Top-K + Other
        {"variable": "app_category",     "strategy": "Top-K + Other",          "n": 3},
        {"variable": "app_domain",       "strategy": "Top-K + Other",          "n": 6},
        {"variable": "banner_pos",       "strategy": "Top-K + Other",          "n": 2},
        {"variable": "C18",              "strategy": "Top-K + Other",          "n": 3},
        {"variable": "C19",              "strategy": "Top-K + Other",          "n": 10},
        {"variable": "C20",              "strategy": "Top-K + Other",          "n": 10},
        {"variable": "C21",              "strategy": "Top-K + Other",          "n": 10},
        {"variable": "site_category",    "strategy": "Top-K + Other",          "n": 3},
        # Híbrida
        {"variable": "app_id",           "strategy": "Híbrida (binaria + TE)", "n": None},
        {"variable": "site_domain",      "strategy": "Híbrida (binaria + TE)", "n": None},
        {"variable": "site_id",          "strategy": "Híbrida (binaria + TE)", "n": None},
        # Target encoding
        {"variable": "C14",              "strategy": "Target encoding",        "n": None},
        {"variable": "C17",              "strategy": "Target encoding",        "n": None},
        {"variable": "device_model",     "strategy": "Target encoding",        "n": None},
        # Feature derivada
        {"variable": "device_id",        "strategy": "Feature derivada",       "n": None},
        {"variable": "device_ip",        "strategy": "Feature derivada",       "n": None},
    ]

    column_titles = ("Variable", "Estrategia", "K")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var, col_strategy, col_k = column_titles

    rows = []
    for spec in specifications:
        rows.append({
            col_var: spec["variable"],
            col_strategy: spec["strategy"],
            col_k: "—" if spec["n"] is None else str(spec["n"]),
        })

    df_strategy = pd.DataFrame(rows)

    n_rows = len(df_strategy)
    half = (n_rows + 1) // 2

    first_half = df_strategy.iloc[:half].reset_index(drop=True)
    second_half = df_strategy.iloc[half:].reset_index(drop=True)

    if len(second_half) < half:
        padding = pd.DataFrame(
            [[""] * df_strategy.shape[1]] * (half - len(second_half)),
            columns=df_strategy.columns,
        )
        second_half = pd.concat([second_half, padding], ignore_index=True)

    combined = pd.concat([first_half, second_half], axis=1)

    styled = (
        combined.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1, 2, 4, 5)))
    )

    if render:
        _render_styled(styled)

    return combined


# ---------------------------------------------------------------------------
# Data Quality / Summary
# ---------------------------------------------------------------------------

def relation_integrity_table(
    con,
    relation,
    column_titles=("Variable", "Dtype", "Missing Values"),
    decimals=0,
    render=True,
):
    """Summarize a DuckDB relation's structure, dtypes and missing values.

    Builds a small integrity table with one row per column in the given
    DuckDB relation, reporting the column name, its DuckDB type as a
    string, and the number of missing values. The null counts are
    computed entirely inside DuckDB with a single aggregate query, so
    the underlying data is never loaded into memory. The resulting table
    is then split vertically into two halves that are placed side by
    side, yielding a compact two-panel layout with six columns and
    ``ceil(n/2)`` rows. If the number of source columns is odd, the
    last row of the second panel is padded with ``None``. Applies a
    consistent, publication-ready style (centered cells, thin black
    borders, left-aligned variable columns). Optionally renders the
    styled table as centered HTML (useful in Jupyter notebooks).

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table, view, or a fully qualified
            relation (e.g. ``"train"`` or ``"read_parquet('path.parquet')"``).
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(variable, dtype, missing)``. Must
            contain exactly three strings. Defaults to
            ``("Variable", "Dtype", "Missing Values")``.
        decimals (int): Number of decimals used to format the missing
            values column. Since counts are integers, the default of
            ``0`` is usually what you want. Defaults to 0.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) two-panel results table
        with six columns (the three column titles duplicated) and
        ``ceil(n/2)`` rows. The styled version is built internally and,
        when ``render=True``, rendered as HTML — it is not returned.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if
            ``relation`` is not a string, if ``column_titles`` is not a
            tuple/list of three strings, if ``decimals`` is not an int,
            or if ``render`` is not a boolean.
        ValueError: If ``column_titles`` does not contain exactly three
            elements, if any element of ``column_titles`` is not a
            string, or if ``decimals`` is negative.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of three strings."
        )

    if len(column_titles) != 3:
        raise ValueError(
            "'column_titles' must contain exactly 3 elements "
            "(variable, dtype, missing)."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("'decimals' must be a non-negative integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var, col_dtype, col_missing = column_titles

    schema_df = con.sql(f"DESCRIBE {relation}").df()
    column_names = schema_df["column_name"].tolist()
    column_types = schema_df["column_type"].tolist()

    null_exprs = ", ".join(
        f'COUNT(*) - COUNT("{name}") AS "{name}"' for name in column_names
    )
    null_counts = (
        con.sql(f"SELECT {null_exprs} FROM {relation}").df().iloc[0].tolist()
    )

    df_integrity = pd.DataFrame(
        {
            col_var: column_names,
            col_dtype: column_types,
            col_missing: null_counts,
        }
    )

    n_rows = len(df_integrity)
    half = (n_rows + 1) // 2

    first_half = df_integrity.iloc[:half].reset_index(drop=True)
    second_half = df_integrity.iloc[half:].reset_index(drop=True)

    if len(second_half) < half:
        padding = pd.DataFrame(
            [[""] * df_integrity.shape[1]] * (half - len(second_half)),
            columns=df_integrity.columns,
        )
        second_half = pd.concat([second_half, padding], ignore_index=True)

    combined = pd.concat([first_half, second_half], axis=1)

    styled_integrity = (
        combined.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1, 4)))
        .format(
            lambda x: _smart_format(x, f"{{:,.{decimals}f}}"),
            subset=[col_missing],
        )
    )

    if render:
        _render_styled(styled_integrity)

    return combined


def show_preview_table(con, relation, limit=5, render=True):
    """Display the first rows of a DuckDB relation in a styled table.

    Executes a ``SELECT *`` with a ``LIMIT`` against the given relation
    and renders the result using the same publication-ready style applied
    by the other summary tables in the toolkit. The first column is
    left-aligned and the remaining columns are centered.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection that
            can resolve ``relation``.
        relation (str): Name of a table, view, or a fully qualified
            relation (e.g. ``"temporal_features"`` or
            ``"read_parquet('path.parquet')"``).
        limit (int): Number of rows to preview. Must be a positive
            integer. Defaults to 5.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The preview table with the first ``limit`` rows.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if ``relation``
            is not a string, or if ``render`` is not a boolean.
        ValueError: If ``limit`` is not a positive integer, or if the
            query returns no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    if not isinstance(relation, str):
        raise TypeError("'relation' must be a string.")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        raise ValueError("'limit' must be a positive integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    df_preview = con.sql(f"SELECT * FROM {relation} LIMIT {limit}").df()

    if df_preview.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    styled = (
        df_preview.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules(left_align_positions=(1,)))
    )

    if render:
        _render_styled(styled)

    return df_preview


# ---------------------------------------------------------------------------
# Modeling Prep
# ---------------------------------------------------------------------------

def create_split_column(
    con,
    relation,
    id_column="id",
    train_ratio=0.70,
    split_column="split",
    train_label="Entrenamiento",
    holdout_label="Prueba",
    view_name=None,
):
    """Create a DuckDB view with a deterministic train/holdout split column.

    Assigns each row of ``relation`` to either the training or the holdout
    subset using a hash-based bucketing scheme. The assignment is a
    deterministic function of ``id_column``, which provides three
    properties simultaneously:

    1. Reproducibility: the same row always lands in the same subset.
    2. Shuffling: hashing destroys any ordering present in the source
       data, so the resulting split is free of order effects.
    3. Practical stratification: because the hash is uniform and
       independent of the target, each subset preserves the target
       distribution of the original dataset up to negligible sampling
       noise.

    A new view is registered in the connection under ``view_name``. The
    original relation is not modified.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of the source table or view.
        id_column (str): Column used as the hash key. Must uniquely
            identify each row. Defaults to ``"id"``.
        train_ratio (float): Fraction of rows assigned to the training
            subset. Must be in ``(0, 1)``. Defaults to 0.70.
        split_column (str): Name of the new partition column. Defaults to
            ``"split"``.
        train_label (str): Label used for the training subset. Defaults
            to ``"Entrenamiento"``.
        holdout_label (str): Label used for the holdout subset. Defaults
            to ``"Prueba"``.
        view_name (str or None): Name of the view to register. When
            ``None``, ``f"{relation}_split"`` is used.

    Returns:
        str: The name of the registered view.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection or any string
            argument has the wrong type.
        ValueError: If ``train_ratio`` is not in ``(0, 1)``.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    for name, value in [
        ("relation", relation),
        ("id_column", id_column),
        ("split_column", split_column),
        ("train_label", train_label),
        ("holdout_label", holdout_label),
    ]:
        if not isinstance(value, str):
            raise TypeError(f"'{name}' must be a string.")

    if view_name is not None and not isinstance(view_name, str):
        raise TypeError("'view_name' must be a string or None.")

    if (
        not isinstance(train_ratio, (int, float))
        or isinstance(train_ratio, bool)
        or not (0 < train_ratio < 1)
    ):
        raise ValueError("'train_ratio' must be a number in (0, 1).")

    threshold = round(train_ratio * 100)
    view_name = view_name or f"{relation}_split"

    con.execute(f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            *,
            CASE
                WHEN hash({id_column}) % 100 < {threshold}
                THEN '{train_label}'
                ELSE '{holdout_label}'
            END AS {split_column}
        FROM {relation}
    """)

    return view_name


def split_summary_table(
    con,
    relation,
    split_column="split",
    target_column="click",
    target_labels=None,
    split_labels=None,
    total_label="Total",
    count_format="{count:,} ({pct:.2f}%)",
    total_format="{count:,} ({pct:.2f}%)",
    render=True,
):
    """Build a summary table of counts per target level and per split.

    Produces a table with one row per level of ``target_column`` plus a
    final row for the totals, and one column per distinct value of
    ``split_column``. Cells in the target rows display the absolute count
    followed by the percentage of that level within its split. Cells in
    the totals row display the absolute count followed by the percentage
    of that split relative to the full dataset.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection.
        relation (str): Name of the source table or view.
        split_column (str): Column that holds the partition label.
            Defaults to ``"split"``.
        target_column (str): Column that holds the binary target.
            Defaults to ``"click"``.
        target_labels (dict or None): Mapping from raw target values to
            display labels. When provided, its key order defines the row
            order. When ``None``, raw values are used and sorted.
        split_labels (dict or None): Mapping from raw split values to
            display labels used as column headers. When provided, its
            key order defines the column order. When ``None``, raw
            values are used and sorted.
        total_label (str): Row label for the totals row. Defaults to
            ``"Total"``.
        count_format (str): Python format string for cells in the target
            rows. Must accept ``{count}`` and ``{pct}``. Defaults to
            ``"{count:,} ({pct:.2f}%)"``.
        total_format (str): Python format string for cells in the totals
            row. Must accept ``{count}`` and ``{pct}``. Defaults to
            ``"{count:,} ({pct:.2f}%)"``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain summary table with formatted strings.

    Raises:
        TypeError: If ``con`` is not a DuckDB connection, if any string
            argument has the wrong type, if ``target_labels`` or
            ``split_labels`` is not a dict or ``None``, or if ``render``
            is not a boolean.
        ValueError: If the underlying query returns no rows.
    """
    import duckdb

    if not isinstance(con, duckdb.DuckDBPyConnection):
        raise TypeError("'con' must be a DuckDB connection.")

    for name, value in [
        ("relation", relation),
        ("split_column", split_column),
        ("target_column", target_column),
        ("total_label", total_label),
        ("count_format", count_format),
        ("total_format", total_format),
    ]:
        if not isinstance(value, str):
            raise TypeError(f"'{name}' must be a string.")

    if target_labels is not None and not isinstance(target_labels, dict):
        raise TypeError("'target_labels' must be a dict or None.")

    if split_labels is not None and not isinstance(split_labels, dict):
        raise TypeError("'split_labels' must be a dict or None.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    query = f"""
        SELECT
            "{target_column}" AS target,
            "{split_column}" AS split,
            COUNT(*) AS n
        FROM {relation}
        GROUP BY target, split
    """
    df_long = con.sql(query).df()

    if df_long.empty:
        raise ValueError(
            f"The query on relation '{relation}' returned no rows."
        )

    targets = (
        list(target_labels.keys()) if target_labels is not None
        else sorted(df_long["target"].unique())
    )
    splits = (
        list(split_labels.keys()) if split_labels is not None
        else sorted(df_long["split"].unique())
    )

    split_totals = df_long.groupby("split")["n"].sum().to_dict()
    grand_total = int(df_long["n"].sum())

    columns = {}
    for split in splits:
        split_df = df_long[df_long["split"] == split]
        split_total = int(split_totals.get(split, 0))

        cells = []
        for target in targets:
            row = split_df[split_df["target"] == target]
            n = int(row["n"].iloc[0]) if not row.empty else 0
            pct = 100.0 * n / split_total if split_total > 0 else 0.0
            cells.append(count_format.format(count=n, pct=pct))

        pct_orig = 100.0 * split_total / grand_total if grand_total > 0 else 0.0
        cells.append(total_format.format(count=split_total, pct=pct_orig))

        col_label = split_labels[split] if split_labels else split
        columns[col_label] = cells

    row_labels = [
        target_labels[t] if target_labels else str(t) for t in targets
    ] + [total_label]

    df_summary = pd.DataFrame(columns, index=row_labels)
    df_summary.index.name = None

    styled = df_summary.style.set_table_styles(_table_style_rules())

    if render:
        _render_styled(styled)

    return df_summary


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _to_pair(value, name):
    """Normalize a per-panel parameter to a 2-element tuple.

    If ``value`` is a tuple or list of length 2, it is returned unchanged.
    Otherwise, it is duplicated to form a 2-element tuple. This is used by
    paired-plot functions to accept parameters that can be specified
    either as a single value (applied to both panels) or as a 2-element
    sequence (one per panel).

    Args:
        value: Value to normalize.
        name (str): Parameter name used in error messages.

    Returns:
        tuple: A 2-element tuple.

    Raises:
        ValueError: If ``value`` is a tuple or list whose length is not 2.
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"'{name}' must be a single value or a length-2 sequence."
            )
        return tuple(value)
    return (value, value)


def _holm_bonferroni(p_values):
    """Apply the Holm-Bonferroni step-down correction to p-values.

    Sorts the p-values in ascending order and applies the correction
    ``p_adj_(i) = max(p_adj_(i-1), (m - i) * p_(i))`` for each rank ``i``,
    clipping the result at 1.0. The output preserves the original order of
    the input.

    Args:
        p_values (array-like): Raw p-values.

    Returns:
        numpy.ndarray: Adjusted p-values in the same order as the input.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        adj = (m - i) * sorted_p[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(running_max, 1.0)

    result = np.empty(m, dtype=float)
    result[order] = adjusted
    return result


def _smart_format(x, fmt):
    """Apply a format string to a value, leaving strings and NaNs untouched.

    Args:
        x: Value to format.
        fmt (str): Python format string (e.g. ``"{:,.4f}"``).

    Returns:
        The formatted value if ``x`` is numeric, otherwise ``x`` unchanged.
    """
    if pd.isna(x) or isinstance(x, str):
        return x
    try:
        return fmt.format(x)
    except (ValueError, TypeError):
        return x


def _format_apa_pvalue(val):
    """Format a p-value in APA style (no leading zero, ``< .001`` rule).

    Args:
        val (float): p-value.

    Returns:
        str: Formatted p-value, or ``""`` when ``val`` is NaN.
    """
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "< .001"
    formatted = f"{val:.3f}"
    return formatted[1:] if formatted.startswith("0.") else formatted


def _format_apa_effect(val, decimals=4):
    """Format an effect size in APA style (no leading zero).

    Effect sizes such as Cramér's V are bounded in ``[0, 1]``. APA 7
    recommends omitting the leading zero for statistics that cannot
    exceed 1 in absolute value.

    Args:
        val (float): Effect size value.
        decimals (int): Number of decimals. Defaults to 4.

    Returns:
        str: Formatted effect size without leading zero, or ``""`` when
        ``val`` is NaN.
    """
    if pd.isna(val):
        return ""
    formatted = f"{val:.{decimals}f}"
    return formatted[1:] if formatted.startswith("0.") else formatted


def _is_number(x):
    """Return ``True`` if ``x`` is a real number (``bool`` is excluded)."""
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float, np.integer, np.floating))


def _is_two_numbers(x):
    """Return ``True`` if ``x`` is a length-2 sequence of real numbers."""
    return (
        isinstance(x, (tuple, list, np.ndarray))
        and len(x) == 2
        and all(_is_number(v) for v in x)
    )


def _broadcast_param(value, n_panels, single_test, name):
    """Broadcast a per-panel parameter to a list of ``n_panels`` entries.

    Args:
        value: Value to broadcast. ``None`` becomes a list of ``None``.
        n_panels (int): Number of panels.
        single_test (callable): Predicate returning ``True`` when
            ``value`` should be applied to every panel.
        name (str): Parameter name used in error messages.

    Returns:
        list: A list of length ``n_panels``.

    Raises:
        ValueError: If ``value`` is neither ``None``, a single value
            accepted by ``single_test``, nor a sequence of length
            ``n_panels``.
    """
    if value is None:
        return [None] * n_panels
    if single_test(value):
        return [value] * n_panels
    if isinstance(value, (tuple, list)) and len(value) == n_panels:
        return list(value)
    raise ValueError(
        f"'{name}' must be a single value or a list of {n_panels} values."
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


def _validate_figsize(figsize):
    """Validate a ``(width, height)`` figure-size tuple."""
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise TypeError("'figsize' must be a tuple/list of length 2.")


def _validate_ylim(ylim):
    """Validate an optional ``(ymin, ymax)`` y-limits tuple."""
    if ylim is not None and (not isinstance(ylim, (tuple, list)) or len(ylim) != 2):
        raise TypeError("'ylim' must be None or a tuple/list of length 2.")


def _validate_y_step(y_step):
    """Validate an optional positive y-axis step."""
    if y_step is not None and (not _is_number(y_step) or y_step <= 0):
        raise ValueError("'y_step' must be None or a positive number.")


def _validate_x_step(x_step):
    """Validate an optional positive x-axis step."""
    if x_step is not None and (not _is_number(x_step) or x_step <= 0):
        raise ValueError("'x_step' must be None or a positive number.")


def _validate_y_format(y_format):
    """Validate an optional ``printf``-style y-axis format string."""
    if y_format is not None and not isinstance(y_format, str):
        raise TypeError("'y_format' must be None or a string.")


def _validate_year_locator_interval(interval):
    """Validate a positive integer used as a year-tick interval."""
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("'year_locator_interval' must be a positive integer.")


def _resolve_p_adjust_method(method):
    """Resolve an alias and validate a p-adjustment method name."""
    key = method.lower()
    if key not in _VALID_P_ADJUST_METHODS:
        raise ValueError(
            f"Unsupported p-adjustment method: '{method}'. "
            f"Valid methods: {sorted(_VALID_P_ADJUST_METHODS)}."
        )
    return key


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


def _hide_yaxis(ax):
    """Hide the y-axis ticks and labels of ``ax``."""
    ax.tick_params(
        axis="y", which="both",
        left=False, right=False,
        labelleft=False, labelright=False,
    )


def _apply_yaxis_settings(ax, ylim=None, y_step=None, y_format=None, yticks=None):
    """Apply optional y-axis limits, locator, formatter and ticks to ``ax``."""
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_step is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))
    if y_format is not None:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(y_format))
    if yticks is not None:
        ax.set_yticks(yticks)

def _table_style_rules(left_align_positions=(1,)):
    """Return the shared HTML style rules used by every styled table.

    Args:
        left_align_positions (tuple of int): 1-based positions of the
            columns whose cells should be left-aligned. All other columns
            are centered. Defaults to ``(1,)``.

    Returns:
        list: A list of style rule dictionaries compatible with
        ``pandas.Styler.set_table_styles``.
    """
    rules = [
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