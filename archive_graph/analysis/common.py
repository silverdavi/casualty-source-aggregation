"""Shared loading / output helpers for the archive_graph analyses.

All published artefacts are counts-only aggregates. No personal names, no
captions, no clip paths may appear in anything written under ``aggregates/``
or ``figures/``. Every JSON artefact passes through
``filters.assert_no_killed_min_sum`` before it is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # archive_graph/
RAW = ROOT / "raw"
AGGREGATES = ROOT / "aggregates"
FIGURES = ROOT / "figures"

sys.path.insert(0, str(ROOT / "ingest"))
import filters  # noqa: E402

BUILD_ID = filters.BUILD_ID


def load(table: str, **kwargs) -> pd.DataFrame:
    path = RAW / f"{table}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run archive_graph/ingest/download.sh first"
        )
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)


def load_events(**kwargs) -> pd.DataFrame:
    events = load("events", **kwargs)
    missing = filters.missing_columns(events, filters.EVENT_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"events table missing columns: {missing}")
    return events


_NAME_KEYS = ("name", "names", "entity_name", "title", "caption", "victim_name")


def _assert_counts_only(payload: Any, path: str = "$") -> None:
    """Reject dict keys that look like they carry personal names or captions."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in _NAME_KEYS:
                raise ValueError(
                    f"potential name/caption field {path}.{key} in published aggregate"
                )
            _assert_counts_only(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            _assert_counts_only(item, f"{path}[{i}]")


def write_aggregate(name: str, payload: dict[str, Any]) -> Path:
    payload = {"build_id": BUILD_ID, **payload}
    filters.assert_no_killed_min_sum(payload)
    _assert_counts_only(payload)
    AGGREGATES.mkdir(parents=True, exist_ok=True)
    out = AGGREGATES / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {out}")
    return out


def savefig(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return out


def month_of(date: pd.Series) -> pd.Series:
    return pd.to_datetime(date, errors="coerce").dt.strftime("%Y-%m")
