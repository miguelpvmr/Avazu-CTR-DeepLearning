"""Build derived features for the Avazu CTR dataset without inducing leakage.

This script performs all preprocessing steps that do not depend on the
target variable and that can be computed without introducing distribution
or target leakage. The output consists of two Parquet files, one for the
training subset and one for the test subset.

Steps performed:
    1. Hash-based split (70 / 30) using ``create_split_column`` from the
       EDA toolkit, with the same deterministic and stratified-by-hash
       properties documented in the exploratory analysis.
    2. Device frequency aggregates computed once on the training subset
       and materialized as small tables.
    3. One export query per subset that joins the raw split view with the
       materialized frequency tables, applies the temporal decomposition,
       applies the log transform and writes the result to Parquet.

The categorical variables are exported in their raw form. Their encoding
is deferred to the modeling stage.

Usage:
    $ cd Avazu-CTR-DeepLearning
    $ python src/data/build_features.py

The script is idempotent: it overwrites the output files on each run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from utils.duckdb_eda_toolkit import create_split_column  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE_ID_PLACEHOLDER = "a99f214a"
TRAIN_RATIO = 0.70
TRAIN_LABEL = "Entrenamiento"
HOLDOUT_LABEL = "Prueba"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def register_raw(con: duckdb.DuckDBPyConnection, data_path: Path) -> None:
    """Register the raw Parquet file as a view named ``raw``.

    Args:
        con: Active DuckDB connection.
        data_path: Path to the raw Parquet file.

    Raises:
        FileNotFoundError: If ``data_path`` does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Source not found: {data_path}")

    con.execute(
        f"CREATE OR REPLACE VIEW raw AS "
        f"SELECT * FROM read_parquet('{data_path.as_posix()}')"
    )
    logger.info("Registered raw view from %s", data_path.name)


def build_split(con: duckdb.DuckDBPyConnection) -> str:
    """Create the hash-based train/holdout split view.

    Delegates to ``create_split_column`` from the EDA toolkit, which
    provides reproducibility, shuffling and practical stratification by
    construction.

    Args:
        con: Active DuckDB connection with a view named ``raw`` already
            registered.

    Returns:
        str: Name of the registered split view.
    """
    view_name = create_split_column(
        con,
        "raw",
        id_column="id",
        train_ratio=TRAIN_RATIO,
        train_label=TRAIN_LABEL,
        holdout_label=HOLDOUT_LABEL,
    )
    logger.info("Registered split view: %s", view_name)
    return view_name


def materialize_frequencies(
    con: duckdb.DuckDBPyConnection,
    split_view: str,
) -> None:
    """Materialize device frequency aggregates as small tables.

    Both aggregates are computed once on the training subset and stored
    as tables in memory. This avoids recomputing the GROUP BY every time
    the frequency tables are joined during the export step. The two
    tables together occupy roughly 200 MB.

    Args:
        con: Active DuckDB connection with the split view registered.
        split_view: Name of the view that contains the ``split`` column.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE freq_id AS
        SELECT device_id, COUNT(*) AS n
        FROM {split_view}
        WHERE split = '{TRAIN_LABEL}'
          AND device_id != '{DEVICE_ID_PLACEHOLDER}'
        GROUP BY device_id
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE freq_ip AS
        SELECT device_ip, COUNT(*) AS n
        FROM {split_view}
        WHERE split = '{TRAIN_LABEL}'
        GROUP BY device_ip
    """)

    n_id = con.sql("SELECT COUNT(*) AS n FROM freq_id").df()["n"].iloc[0]
    n_ip = con.sql("SELECT COUNT(*) AS n FROM freq_ip").df()["n"].iloc[0]
    logger.info(
        "Materialized frequencies | device_id: %s | device_ip: %s",
        f"{int(n_id):,}",
        f"{int(n_ip):,}",
    )


def export_subset(
    con: duckdb.DuckDBPyConnection,
    split_view: str,
    split_label: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Export one subset to Parquet with all derived features applied.

    Executes a single SELECT that joins the split view with the two
    materialized frequency tables, applies the temporal decomposition and
    the log transform, and writes the result to Parquet. All
    transformations are either deterministic per row or use only the
    training-side frequency tables, so no leakage is introduced.

    Args:
        con: Active DuckDB connection with all views and tables ready.
        split_view: Name of the view that contains the ``split`` column.
        split_label: Value of the split column that identifies the subset.
        filename: Name of the output Parquet file.
        output_dir: Directory where the output file will be written.
    """
    target_path = output_dir / filename

    con.execute(f"""
        COPY (
            SELECT
                s.click,
                CAST(s.hour / 100 % 100 AS INTEGER) AS day,
                SIN(2 * PI() * CAST(s.hour % 100 AS INTEGER) / 24)
                    AS hour_sin,
                COS(2 * PI() * CAST(s.hour % 100 AS INTEGER) / 24)
                    AS hour_cos,
                CASE
                    WHEN CAST(s.hour % 100 AS INTEGER) BETWEEN 0 AND 5
                        THEN 'Madrugada'
                    WHEN CAST(s.hour % 100 AS INTEGER) BETWEEN 6 AND 11
                        THEN 'Mañana'
                    WHEN CAST(s.hour % 100 AS INTEGER) BETWEEN 12 AND 17
                        THEN 'Tarde'
                    ELSE 'Noche'
                END AS time_band,
                LN(
                    CASE
                        WHEN s.device_id = '{DEVICE_ID_PLACEHOLDER}' THEN 1
                        ELSE COALESCE(fid.n, 0) + 1
                    END
                ) AS log_freq_device_id,
                LN(COALESCE(fip.n, 0) + 1) AS log_freq_device_ip,
                s.site_id,
                s.site_domain,
                s.site_category,
                s.app_id,
                s.app_domain,
                s.app_category,
                s.device_model,
                s.device_type,
                s.device_conn_type,
                s.banner_pos,
                s."C1"  AS C1,
                s."C14" AS C14,
                s."C15" AS C15,
                s."C16" AS C16,
                s."C17" AS C17,
                s."C18" AS C18,
                s."C19" AS C19,
                s."C20" AS C20,
                s."C21" AS C21
            FROM {split_view} AS s
            LEFT JOIN freq_id AS fid ON s.device_id = fid.device_id
            LEFT JOIN freq_ip AS fip ON s.device_ip = fip.device_ip
            WHERE s.split = '{split_label}'
        )
        TO '{(target_path).as_posix()}'
        (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    size_mb = target_path.stat().st_size / 1_000_000
    logger.info("%s written | size: %.1f MB", filename, size_mb)


def log_summary(con: duckdb.DuckDBPyConnection, split_view: str) -> None:
    """Log row counts and click rate per subset as a sanity check.

    Args:
        con: Active DuckDB connection with the split view registered.
        split_view: Name of the view that contains the ``split`` column.
    """
    df = con.sql(f"""
        SELECT
            split,
            COUNT(*) AS n,
            ROUND(AVG(click) * 100, 4) AS ctr_pct
        FROM {split_view}
        GROUP BY split
        ORDER BY split
    """).df()

    for _, row in df.iterrows():
        logger.info(
            "%s | rows: %s | click rate: %.4f%%",
            row["split"], f"{int(row['n']):,}", row["ctr_pct"],
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full preprocessing pipeline."""
    con = duckdb.connect(database=":memory:")

    register_raw(con, DATA_PATH)
    split_view = build_split(con)

    logger.info("Materializing frequency aggregates")
    materialize_frequencies(con, split_view)

    log_summary(con, split_view)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting training subset")
    export_subset(
        con, split_view,
        TRAIN_LABEL, "train.parquet",
        PROCESSED_DIR,
    )

    logger.info("Exporting test subset")
    export_subset(
        con, split_view,
        HOLDOUT_LABEL, "test.parquet",
        PROCESSED_DIR,
    )

    logger.info("Preprocessing completed")


if __name__ == "__main__":
    main()