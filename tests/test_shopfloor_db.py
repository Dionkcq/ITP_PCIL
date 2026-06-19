"""Tests for the postgres shop-floor data source (pcil.shopfloor_db).

A live PostgreSQL is not required: a fake connection records the SQL issued and
returns canned rows, so the slice->SQL mapping, the CSV->table seed (including
NaN->NULL handling), and the identifier-safety guard are all exercised offline.
The orchestrator's _pull_slice postgres branch is covered by mocking query_slice.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pcil.shopfloor_db as sdb


# ── Fake psycopg connection ────────────────────────────────────────

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeCopyCtx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write_row(self, row):
        self.conn.copied_rows.append(row)


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def copy(self, sql):
        self.conn.executed.append((sql, None))
        return _FakeCopyCtx(self.conn)


class FakeConn:
    def __init__(self, *, select_rows=None, count=0):
        self.select_rows = select_rows or []
        self.count = count
        self.executed: list[tuple] = []
        self.copied_rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        low = sql.lower()
        if "count(" in low:
            return _Result([{"n": self.count}])
        if low.lstrip().startswith("select"):
            return _Result(self.select_rows)
        return _Result([])

    def cursor(self):
        return _FakeCursor(self)


# ── Identifier safety ──────────────────────────────────────────────

def test_safe_ident_accepts_plain_names():
    assert sdb._safe_ident("shop_floor", kind="table name") == "shop_floor"


def test_safe_ident_rejects_injection():
    with pytest.raises(ValueError):
        sdb._safe_ident("shop_floor; DROP TABLE x", kind="table name")
    with pytest.raises(ValueError):
        sdb._safe_ident("1bad", kind="table name")


def test_column_type_inference():
    assert sdb._column_type(pd.Series(["x"]), is_timestamp=True) == "TIMESTAMPTZ"
    assert sdb._column_type(pd.Series([1, 2, 3]), is_timestamp=False) == "BIGINT"
    assert sdb._column_type(pd.Series([1.0, 2.5]), is_timestamp=False) == "DOUBLE PRECISION"


# ── query_slice ────────────────────────────────────────────────────

def test_query_slice_all_orders_by_timestamp(monkeypatch):
    rows = [{"timestamp": "2026-01-01T00:00:00+00:00", "vibration": 0.1}]
    conn = FakeConn(select_rows=rows)
    monkeypatch.setattr(sdb, "_connect", lambda *a, **k: conn)

    df = sdb.query_slice("shop_floor", "timestamp", mode="all")

    assert list(df.columns) == ["timestamp", "vibration"]
    assert len(df) == 1
    sql = conn.executed[-1][0]
    assert 'FROM "shop_floor"' in sql
    assert 'ORDER BY "timestamp"' in sql


def test_query_slice_last_n_reverses_to_ascending(monkeypatch):
    # The DB returns DESC; query_slice must re-sort to chronological order.
    rows = [{"timestamp": 3}, {"timestamp": 2}, {"timestamp": 1}]
    conn = FakeConn(select_rows=rows)
    monkeypatch.setattr(sdb, "_connect", lambda *a, **k: conn)

    df = sdb.query_slice("shop_floor", "timestamp", mode="last_n", last_n=3)

    assert list(df["timestamp"]) == [1, 2, 3]
    assert "LIMIT" in conn.executed[-1][0].upper()


def test_query_slice_time_range_requires_bounds(monkeypatch):
    conn = FakeConn(select_rows=[])
    monkeypatch.setattr(sdb, "_connect", lambda *a, **k: conn)
    with pytest.raises(ValueError):
        sdb.query_slice("shop_floor", "timestamp", mode="time_range")


# ── seed_table_from_csv ────────────────────────────────────────────

def test_seed_loads_rows_and_maps_nan_to_null(monkeypatch, tmp_path):
    csv = tmp_path / "mock.csv"
    csv.write_text(
        "timestamp,vibration,flag\n"
        "2026-01-01T00:00:00+00:00,0.5,1\n"
        "2026-01-01T00:00:01+00:00,,0\n",
        encoding="utf-8",
    )
    conn = FakeConn(count=0)
    monkeypatch.setattr(sdb, "_connect", lambda *a, **k: conn)

    result = sdb.seed_table_from_csv(csv, "shop_floor", "timestamp")

    assert result["status"] == "loaded"
    assert result["rows"] == 2
    # The missing vibration value (row 2, col index 1) became None, not NaN.
    assert conn.copied_rows[1][1] is None
    # The timestamp column was declared TIMESTAMPTZ.
    create_sql = next(s for s, _ in conn.executed if "create table" in s.lower())
    assert "TIMESTAMPTZ" in create_sql


def test_seed_skips_when_table_already_has_rows(monkeypatch, tmp_path):
    csv = tmp_path / "mock.csv"
    csv.write_text(
        "timestamp,vibration\n2026-01-01T00:00:00+00:00,0.5\n", encoding="utf-8"
    )
    conn = FakeConn(count=5)
    monkeypatch.setattr(sdb, "_connect", lambda *a, **k: conn)

    result = sdb.seed_table_from_csv(csv, "shop_floor", "timestamp", if_empty=True)

    assert result["status"] == "skipped"
    assert result["rows"] == 5
    assert conn.copied_rows == []  # nothing was loaded


# ── orchestrator _pull_slice postgres branch ───────────────────────

def test_pull_slice_uses_postgres_when_recipe_says_so(
    client, shop_floor_tiny_path, monkeypatch,
):
    """A recipe with source_type=postgres routes _pull_slice through
    query_slice; the rest of the pipeline is unchanged."""
    df = pd.read_csv(shop_floor_tiny_path)
    monkeypatch.setattr(sdb, "query_slice", lambda *a, **k: df)

    r = client.post(
        "/pipeline/run",
        json={
            "config_path": "systems/inkjet_printer/config_postgres.yaml",
            "persist": False,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_rows"] == len(df)
    assert body["impacts"]["system"] == "inkjet_printer"
