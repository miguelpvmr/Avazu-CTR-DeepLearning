"""Convert the Avazu CTR training file from gzip-compressed CSV to Parquet.

This module reads the raw ``train.gz`` file provided by the Avazu
Click-Through Rate Prediction dataset, parses it in chunks to bound memory
usage, and writes the result as a Snappy-compressed Parquet file. The
output is placed alongside the source file within ``data/raw`` so that
downstream code can consume it directly.

The script is intended to be executed from the project root, that is, the
directory that contains the ``data`` folder:

    $ cd Avazu-CTR-DeepLearning
    $ python src/utils/convert_to_parquet.py

The script is idempotent: if ``train.parquet`` already exists, the
conversion is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# The script assumes it is executed from the project root, which is the
# directory that contains the ``data`` folder.
PROJECT_ROOT: Path = Path.cwd()
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"

SOURCE_PATH: Path = RAW_DIR / "train.gz"
TARGET_PATH: Path = RAW_DIR / "train.parquet"

# ---------------------------------------------------------------------------
# Column data types
# ---------------------------------------------------------------------------
# Explicit dtypes prevent pandas from inferring wide integer or object
# columns, which reduces peak memory usage during conversion and yields a
# smaller Parquet output. High-cardinality identifiers are stored as
# ``category`` so that Parquet encodes them as dictionaries.
#
# The value type is annotated as ``Any`` because pandas' ``read_csv``
# accepts a broad union of dtype specifications (str, np.dtype,
# ExtensionDtype, type, ...). Using ``Any`` keeps the type checker happy
# without relying on pandas' private typing module.
DTYPES: dict[str, Any] = {
    "id": "int32",
    "click": "int8",
    "hour": "int32",
    "C1": "int16",
    "banner_pos": "int8",
    "site_id": "category",
    "site_domain": "category",
    "site_category": "category",
    "app_id": "category",
    "app_domain": "category",
    "app_category": "category",
    "device_id": "category",
    "device_ip": "category",
    "device_model": "category",
    "device_type": "int8",
    "device_conn_type": "int8",
    "C14": "int32",
    "C15": "int16",
    "C16": "int16",
    "C17": "int32",
    "C18": "int32",
    "C19": "int32",
    "C20": "int32",
    "C21": "int32",
}

# Default number of rows read per chunk. A value of 1,000,000 rows requires
# roughly 500 MB of free RAM during conversion. Lower it to 500,000 if the
# host machine has 8 GB of RAM or less.
DEFAULT_CHUNK_SIZE: int = 1_000_000


def gzip_csv_to_snappy_parquet(
    input_path: Path,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Convert a gzip-compressed CSV file into a Snappy-compressed Parquet file.

    The conversion is performed in chunks to bound peak memory usage, which
    makes the function suitable for datasets that do not fit in RAM. Each
    chunk is written as an independent row group inside the resulting
    Parquet file.

    Because pandas ``category`` columns are translated to PyArrow dictionary
    types, the dictionary index width (int8, int16, int32) can vary between
    chunks depending on how many unique values each chunk happens to
    contain. To keep the Parquet schema stable, every chunk after the first
    one is explicitly cast to the schema established by the first chunk.

    Args:
        input_path: Path to the source ``.gz`` file containing CSV data.
        output_path: Path where the resulting ``.parquet`` file will be
            written. Any existing file at this location must be removed
            beforehand, as this function does not overwrite by default.
        chunk_size: Number of rows to read per chunk. Defaults to
            ``DEFAULT_CHUNK_SIZE``.

    Returns:
        None. The function writes the Parquet file to ``output_path``.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If ``chunk_size`` is not a positive integer.
        OSError: If the output file cannot be written to disk.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Source file not found: {input_path}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    logger.info("Processing: %s", input_path.name)
    logger.info("Output:     %s", output_path.name)

    reader = pd.read_csv(
        input_path,
        compression="gzip",
        chunksize=chunk_size,
        dtype=DTYPES,
    )  # type: ignore[call-overload]

    writer: pq.ParquetWriter | None = None
    total_rows: int = 0

    try:
        for chunk_index, chunk in enumerate(reader, start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="snappy",
                )
            else:
                table = table.cast(writer.schema)

            writer.write_table(table)
            total_rows += len(chunk)
            logger.info(
                "Chunk %d written | cumulative rows: %s",
                chunk_index,
                f"{total_rows:,}",
            )
    finally:
        if writer is not None:
            writer.close()

    size_mb = output_path.stat().st_size / 1_000_000
    logger.info(
        "%s completed | rows: %s | size: %.1f MB",
        output_path.name,
        f"{total_rows:,}",
        size_mb,
    )


def main() -> None:
    """Run the conversion for the training file.

    Skips the conversion if the target Parquet file already exists.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DIR}. "
            "Make sure the script is executed from the project root."
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_PATH}")

    if TARGET_PATH.exists():
        logger.info("Target already exists, skipping: %s", TARGET_PATH.name)
        return

    gzip_csv_to_snappy_parquet(SOURCE_PATH, TARGET_PATH)
    logger.info("Conversion process finished.")


if __name__ == "__main__":
    main()