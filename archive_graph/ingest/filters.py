"""Analysis-ready filters and anti-misuse guards for the event graph.

An event is a footage cluster, not a verified incident. Casualty columns are
caption-stated figures for that cluster. Never publish a sum of ``killed_min``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

import pandas as pd

BUILD_ID = "f76bd2fb"

EVENT_REQUIRED_COLUMNS = (
    "id",
    "title",
    "date",
    "date_source",
    "place_id",
    "place",
    "lat",
    "lon",
    "facility",
    "category",
    "category2",
    "evidence_type",
    "killed_min",
    "killed_max",
    "injured_min",
    "injured_max",
    "cumulative_min",
    "cumulative_max",
    "figures_disputed",
    "methods",
    "perpetrators",
    "victim_groups",
    "verification_tier",
    "verification_anchor",
    "report_day",
    "dating",
    "n_clips",
    "n_distinct_footage",
    "n_archives",
    "n_posts",
    "mapped",
)

DATE_SOURCE_STATED = "stated"
DATE_SOURCE_POST_PROXY = "post_proxy"

# Keys that would smuggle a killed_min total into a published artefact.
# Do not match flags such as never_sum_killed_min.
_FORBIDDEN_KEY_RE = re.compile(
    r"(^|[_-])(sum[_-]?of[_-]?killed[_-]?min|killed[_-]?min[_-]?sum|"
    r"total[_-]?killed[_-]?min|killed[_-]?min[_-]?total|sum_killed_min)($|[_-])",
    re.IGNORECASE,
)


class CasualtyMisuseError(ValueError):
    """Raised when a caller would publish a prohibited casualty aggregate."""


def missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    have = set(df.columns)
    return [c for c in required if c not in have]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_blank_or_zero(series: pd.Series) -> pd.Series:
    """True when a cumulative/optional numeric field is empty or zero."""
    text = series.astype("string").str.strip()
    blank = series.isna() | text.isin(("", "nan", "None", "NaN", "<NA>"))
    num = _numeric(series)
    return blank | num.eq(0)


def mask_stated(events: pd.DataFrame) -> pd.Series:
    src = events["date_source"].astype("string").str.strip()
    return src.eq(DATE_SOURCE_STATED)


def mask_mapped(events: pd.DataFrame) -> pd.Series:
    mapped = _numeric(events["mapped"])
    return mapped.eq(1)


def mask_has_killed_min(events: pd.DataFrame) -> pd.Series:
    return _numeric(events["killed_min"]).notna()


def mask_cumulative_empty(events: pd.DataFrame) -> pd.Series:
    return is_blank_or_zero(events["cumulative_min"]) & is_blank_or_zero(
        events["cumulative_max"]
    )


def mask_non_cumulative_killed(events: pd.DataFrame) -> pd.Series:
    """Incident-level caption figures: killed_min present, cumulative columns empty/zero."""
    return mask_has_killed_min(events) & mask_cumulative_empty(events)


def analysis_ready_mask(
    events: pd.DataFrame,
    *,
    stated: bool = True,
    mapped: bool = True,
    non_cumulative_killed: bool = False,
) -> pd.Series:
    """Boolean mask for the default analysis-ready event subset.

    Defaults keep stated dates and mapped place anchors. Set
    ``non_cumulative_killed=True`` when the unit of analysis is caption-stated
    cluster size (never a death toll).
    """
    missing = missing_columns(events, EVENT_REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"events table missing columns: {missing}")
    mask = pd.Series(True, index=events.index)
    if stated:
        mask &= mask_stated(events)
    if mapped:
        mask &= mask_mapped(events)
    if non_cumulative_killed:
        mask &= mask_non_cumulative_killed(events)
    return mask


def analysis_ready_events(events: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return events.loc[analysis_ready_mask(events, **kwargs)].copy()


def killed_min_distribution(killed_min: pd.Series) -> dict[str, Any]:
    """Publishable cluster-size stats. Deliberately omits any sum."""
    values = _numeric(killed_min).dropna()
    n = int(values.size)
    stats: dict[str, Any] = {
        "n_with_killed_min": n,
        "unit": "caption-stated cluster size, not a death toll",
        "not_a_death_toll": True,
    }
    if n == 0:
        stats.update(
            {
                "min": None,
                "p50": None,
                "p90": None,
                "p99": None,
                "max": None,
                "share_ge_10": None,
            }
        )
        return stats
    stats.update(
        {
            "min": float(values.min()),
            "p50": float(values.quantile(0.5)),
            "p90": float(values.quantile(0.9)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
            "share_ge_10": float((values >= 10).mean()),
        }
    )
    for key in ("min", "p50", "p90", "p99", "max", "share_ge_10"):
        stats[key] = round(stats[key], 6)
    if "sum" in stats:
        raise CasualtyMisuseError("killed_min_distribution must not include a sum")
    return stats


def assert_no_killed_min_sum(payload: Any, *, path: str = "$") -> None:
    """Walk a published JSON-like object and reject killed_min totals."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_s = str(key)
            here = f"{path}.{key_s}"
            if _FORBIDDEN_KEY_RE.search(key_s):
                raise CasualtyMisuseError(
                    f"forbidden casualty total at {here}: never publish a sum of killed_min"
                )
            if key_s == "sum" and "killed" in path.lower():
                raise CasualtyMisuseError(
                    f"forbidden casualty total at {here}: never publish a sum of killed_min"
                )
            assert_no_killed_min_sum(value, path=here)
        return
    if isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            assert_no_killed_min_sum(item, path=f"{path}[{i}]")


def load_json_artifact(path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def refuse_sum_killed_min(series: pd.Series) -> None:
    """Explicit API: summing killed_min is not a supported operation."""
    raise CasualtyMisuseError(
        "never sum killed_min; duplicates and over-split footage make the "
        "total a false death toll. Use killed_min_distribution() instead."
    )
