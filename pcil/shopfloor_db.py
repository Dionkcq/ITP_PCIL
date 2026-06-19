"""PostgreSQL shop-floor data source for the trigger / _pull_slice path.

Optional: only used when a recipe sets ``trigger.source_type: postgres``. The
connection comes from ``DATABASE_URL`` (the dockerized postgres service the
compose file provisions); the recipe names the ``table``. CSV stays the default
source in ``orchestrator._pull_slice`` so existing recipes and tests are
unaffected.

Two responsibilities:

  seed_table_from_csv()  one-off load of a CSV into a table (idempotent), so a
                         dev/demo deployment can populate the dockerized DB from
                         the same mock_shop_floor.csv the CSV path uses.
  query_slice()          turn a recipe trigger (all / time_range / last_n) into
                         a single SQL query and return a pandas DataFrame, so
                         the rest of the pipeline (preprocess -> adapter ->
                         context model) is identical whether the slice came from
                         a CSV or the DB.

psycopg is imported lazily so the CSV path and the test suite run without a
PostgreSQL driver installed.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Postgres identifiers we interpolate into SQL (table + timestamp column) are
# validated against this instead of parameterised, because identifiers cannot
# be bound as query parameters. Values (time bounds, limits) ARE parameterised.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_url() -> str | None:
    return os.environ.get("DATABASE_URL")


def _connect(*, autocommit: bool = False):
    url = database_url()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set; a postgres shop-floor source needs it "
            "(the docker-compose file sets it for the bundled postgres service)."
        )
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - exercised only in postgres mode
        raise RuntimeError(
            "psycopg is not installed; install psycopg[binary] to use a "
            "postgres shop-floor source."
        ) from exc
    return psycopg.connect(url, autocommit=autocommit, row_factory=dict_row)


def _safe_ident(name: str, *, kind: str) -> str:
    if not name or not _IDENT_RE.match(name):
        raise ValueError(
            f"unsafe {kind} {name!r}: only letters, digits and underscore are "
            "allowed and it must not start with a digit"
        )
    return name


def _column_type(series: pd.Series, *, is_timestamp: bool) -> str:
    if is_timestamp:
        return "TIMESTAMPTZ"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_numeric_dtype(series):
        return "DOUBLE PRECISION"
    return "TEXT"


def _to_py(value: Any) -> Any:
    """Coerce a pandas/numpy cell to a plain Python value psycopg can COPY."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar -> python scalar
        return value.item()
    return value


def table_exists_with_rows(table: str) -> bool:
    """True when the table exists and holds at least one row. Never raises."""
    try:
        table = _safe_ident(table, kind="table name")
        with _connect() as conn:
            row = conn.execute(
                f'SELECT count(*) AS n FROM "{table}"'
            ).fetchone()
        return bool(row and row["n"])
    except Exception:  # noqa: BLE001 - table missing / DB down -> "no rows"
        return False


def seed_table_from_csv(
    csv_path: str | Path,
    table: str,
    timestamp_column: str,
    *,
    if_empty: bool = True,
) -> dict[str, Any]:
    """Create ``table`` (schema inferred from the CSV) and load the CSV rows.

    The table mirrors the CSV 1:1: the timestamp column becomes TIMESTAMPTZ
    (with an index, since every slice filters/orders by time), and the other
    columns map to BIGINT / DOUBLE PRECISION / BOOLEAN / TEXT by dtype.

    Idempotent: with ``if_empty`` (the default) the load is skipped when the
    table already has rows, so container restarts do not duplicate data. Pass
    ``if_empty=False`` to force a TRUNCATE + reload.
    """
    table = _safe_ident(table, kind="table name")
    ts = _safe_ident(timestamp_column, kind="timestamp column")

    df = pd.read_csv(csv_path)
    if ts in df.columns:
        df[ts] = pd.to_datetime(df[ts])

    columns = list(df.columns)
    coldefs = ", ".join(
        f'"{c}" {_column_type(df[c], is_timestamp=(c == ts))}' for c in columns
    )

    with _connect(autocommit=True) as conn:
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({coldefs})')
        if ts in columns:
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{table}_ts" '
                f'ON "{table}" ("{ts}")'
            )
        existing = conn.execute(
            f'SELECT count(*) AS n FROM "{table}"'
        ).fetchone()["n"]
        if existing and if_empty:
            return {"status": "skipped", "table": table, "rows": int(existing)}
        if existing and not if_empty:
            conn.execute(f'TRUNCATE "{table}"')

        # NaN -> NULL before loading; COPY is the fast bulk path.
        clean = df.where(pd.notna(df), None)
        collist = ", ".join(f'"{c}"' for c in columns)
        with conn.cursor() as cur:
            with cur.copy(f'COPY "{table}" ({collist}) FROM STDIN') as copy:
                for row in clean.itertuples(index=False, name=None):
                    copy.write_row(tuple(_to_py(v) for v in row))

    return {"status": "loaded", "table": table, "rows": int(len(df))}


def query_slice(
    table: str,
    timestamp_column: str,
    *,
    mode: str = "all",
    start_time: str | None = None,
    end_time: str | None = None,
    last_n: int | None = None,
) -> pd.DataFrame:
    """Pull a slice from ``table`` as a DataFrame, mapping the trigger to SQL.

    all        -> SELECT * ... ORDER BY <ts>
    time_range -> ... WHERE <ts> BETWEEN %s AND %s ORDER BY <ts>
    last_n     -> ... ORDER BY <ts> DESC LIMIT %s, then re-sorted ascending
                  so downstream code sees chronological rows (matches the CSV
                  slice_last_n_rows contract).
    """
    table = _safe_ident(table, kind="table name")
    ts = _safe_ident(timestamp_column, kind="timestamp column")
    mode = (mode or "all").lower()

    with _connect() as conn:
        if mode == "all":
            rows = conn.execute(
                f'SELECT * FROM "{table}" ORDER BY "{ts}"'
            ).fetchall()
        elif mode == "time_range":
            if not start_time or not end_time:
                raise ValueError(
                    "trigger.mode=time_range requires start_time and end_time"
                )
            rows = conn.execute(
                f'SELECT * FROM "{table}" WHERE "{ts}" BETWEEN %s AND %s '
                f'ORDER BY "{ts}"',
                (start_time, end_time),
            ).fetchall()
        elif mode == "last_n":
            if last_n is None:
                raise ValueError("trigger.mode=last_n requires last_n")
            rows = conn.execute(
                f'SELECT * FROM "{table}" ORDER BY "{ts}" DESC LIMIT %s',
                (int(last_n),),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            raise ValueError(f"unknown trigger.mode: {mode}")

    return pd.DataFrame(rows)
