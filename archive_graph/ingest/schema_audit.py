#!/usr/bin/env python3
"""Schema audit for the pinned Archive Genocide event-graph build.

Writes count-only JSON (no names, no captions, no sum of killed_min).
Requires a verified raw download in archive_graph/raw/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from filters import (
    BUILD_ID,
    EVENT_REQUIRED_COLUMNS,
    analysis_ready_mask,
    assert_no_killed_min_sum,
    killed_min_distribution,
    mask_cumulative_empty,
    mask_has_killed_min,
    mask_mapped,
    mask_non_cumulative_killed,
    mask_stated,
    missing_columns,
)

MODULE = HERE.parent
DEFAULT_RAW = MODULE / "raw"
DEFAULT_OUT = MODULE / "aggregates" / "schema_audit.json"

TABLES = {
    "events": "events.csv.gz",
    "event_clips": "event_clips.csv.gz",
    "people": "people.csv.gz",
    "actors": "actors.csv.gz",
    "facilities": "facilities.csv.gz",
    "event_relations": "event_relations.csv.gz",
    "event_entities": "event_entities.csv.gz",
    "places": "places.csv.gz",
}

PEOPLE_REQUIRED = ("key", "in_victims_list", "victim_ambiguous")
ACTORS_REQUIRED = ("key",)
FACILITIES_REQUIRED = ("key",)
EVENT_ENTITIES_REQUIRED = ("relation", "entity_key")
PLACES_REQUIRED = ("id",)


def _read_csv(path: Path, *, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False, usecols=usecols)


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.astype("string").fillna("<NA>")
    return dict(Counter(values.tolist()))


def _int_flag_counts(series: pd.Series) -> dict[str, int]:
    num = pd.to_numeric(series, errors="coerce")
    return {
        "n_1": int((num == 1).sum()),
        "n_0": int((num == 0).sum()),
        "n_null": int(num.isna().sum()),
    }


def audit(raw_dir: Path) -> dict:
    pinned = (MODULE / "BUILD_ID").read_text(encoding="utf-8").strip()
    if pinned != BUILD_ID:
        raise SystemExit(f"BUILD_ID file {pinned!r} != code pin {BUILD_ID!r}")

    readme = raw_dir / "README.txt"
    if not readme.is_file():
        raise SystemExit(
            f"missing {readme}; run archive_graph/ingest/download.sh first"
        )
    readme_text = readme.read_text(encoding="utf-8", errors="replace")
    if f"Build {BUILD_ID}" not in readme_text:
        raise SystemExit(f"raw README.txt is not build {BUILD_ID}")

    missing_files = [
        name for name in TABLES.values() if not (raw_dir / name).is_file()
    ]
    if missing_files:
        raise SystemExit(f"missing raw files: {missing_files}")

    events = _read_csv(raw_dir / TABLES["events"])
    clips = _read_csv(
        raw_dir / TABLES["event_clips"],
        usecols=lambda c: c in {"event_id", "entry_id", "archive", "kind"},
    )
    people = _read_csv(
        raw_dir / TABLES["people"],
        usecols=lambda c: c in {"key", "in_victims_list", "victim_ambiguous"},
    )
    actors = _read_csv(raw_dir / TABLES["actors"], usecols=lambda c: c == "key")
    facilities = _read_csv(
        raw_dir / TABLES["facilities"], usecols=lambda c: c in {"key", "resolved"}
    )
    relations = _read_csv(
        raw_dir / TABLES["event_relations"],
        usecols=lambda c: c in {"a", "b", "kind"},
    )
    entities = _read_csv(
        raw_dir / TABLES["event_entities"],
        usecols=lambda c: c in {"relation", "entity_key"},
    )
    places = _read_csv(raw_dir / TABLES["places"], usecols=lambda c: c == "id")

    problems: list[str] = []
    for label, df, required in (
        ("events", events, EVENT_REQUIRED_COLUMNS),
        ("event_clips", clips, ("event_id", "entry_id")),
        ("people", people, PEOPLE_REQUIRED),
        ("actors", actors, ACTORS_REQUIRED),
        ("facilities", facilities, FACILITIES_REQUIRED),
        ("event_relations", relations, ("a", "b", "kind")),
        ("event_entities", entities, EVENT_ENTITIES_REQUIRED),
        ("places", places, PLACES_REQUIRED),
    ):
        miss = missing_columns(df, required)
        if miss:
            problems.append(f"{label} missing columns: {miss}")

    stated = mask_stated(events)
    mapped = mask_mapped(events)
    non_cum_killed = mask_non_cumulative_killed(events)
    ready = analysis_ready_mask(events, stated=True, mapped=True)
    ready_killed = analysis_ready_mask(
        events, stated=True, mapped=True, non_cumulative_killed=True
    )

    killed_all = killed_min_distribution(events.loc[mask_has_killed_min(events), "killed_min"])
    killed_non_cum = killed_min_distribution(
        events.loc[non_cum_killed, "killed_min"]
    )

    notes = [
        "event_relations.csv has kind/why, not a grade column; "
        "kind=possible_same_event is the documented possible-link analogue",
    ]
    if (killed_non_cum.get("max") or 0) >= 10_000:
        notes.append(
            "non_cumulative_killed still includes extreme killed_min outliers; "
            "caption figures can be mis-filed war totals. Do not sum them."
        )

    report = {
        "build_id": BUILD_ID,
        "source": "https://archivegenocide.com/lab/data/",
        "credit": "Archive Genocide",
        "licence": "CC BY 4.0",
        "anti_misuse": {
            "event_is_footage_cluster": True,
            "killed_min_is_not_a_toll": True,
            "coordinates_are_place_anchors": True,
            "verification_tier_is_archive_corroboration_not_truth": True,
        },
        "n_rows": {
            "events": int(len(events)),
            "event_clips": int(len(clips)),
            "people": int(len(people)),
            "actors": int(len(actors)),
            "facilities": int(len(facilities)),
            "event_relations": int(len(relations)),
            "event_entities": int(len(entities)),
            "places": int(len(places)),
        },
        "schema": {
            "events_columns": list(events.columns),
            "problems": problems,
            "notes": notes,
        },
        "date_source": _counts(events["date_source"]),
        "mapped": _int_flag_counts(events["mapped"]),
        "verification_tier": _counts(events["verification_tier"]),
        "filters": {
            "stated": int(stated.sum()),
            "mapped": int(mapped.sum()),
            "stated_and_mapped": int(ready.sum()),
            "has_killed_min": int(mask_has_killed_min(events).sum()),
            "cumulative_empty": int(mask_cumulative_empty(events).sum()),
            "non_cumulative_killed": int(non_cum_killed.sum()),
            "stated_mapped_non_cumulative_killed": int(ready_killed.sum()),
        },
        "killed_min_distribution_all_non_null": killed_all,
        "killed_min_distribution_non_cumulative": killed_non_cum,
        "people_flags": {
            "in_victims_list": _int_flag_counts(people["in_victims_list"]),
            "victim_ambiguous": _int_flag_counts(people["victim_ambiguous"]),
        },
        "event_relations_kind": _counts(relations["kind"])
        if "kind" in relations.columns
        else {},
        "event_relations_grade": _counts(relations["grade"])
        if "grade" in relations.columns
        else {"absent_from_csv": True},
        "event_entities_relation": _counts(entities["relation"])
        if "relation" in entities.columns
        else {},
    }
    if report["n_rows"]["events"] != 92588:
        problems.append(
            f"events row count {report['n_rows']['events']} != 92588 advertised for this build"
        )
    if report["n_rows"]["event_clips"] != 131387:
        problems.append(
            f"event_clips row count {report['n_rows']['event_clips']} != 131387 advertised for this build"
        )
    report["schema"]["problems"] = problems
    assert_no_killed_min_sum(report)
    report["ok"] = not problems
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    report = audit(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  events={report['n_rows']['events']} clips={report['n_rows']['event_clips']}")
    print(f"  stated_and_mapped={report['filters']['stated_and_mapped']}")
    print(f"  non_cumulative_killed={report['filters']['non_cumulative_killed']}")
    print("  killed_min_is_not_a_toll=true")
    if not report["ok"]:
        print("schema problems:", report["schema"]["problems"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
