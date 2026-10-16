#!/usr/bin/env python3
"""Focused checks for schema filters and the killed_min anti-misuse rule.

Run:
    python3 archive_graph/ingest/test_schema_filters.py
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from filters import (  # noqa: E402
    BUILD_ID,
    CasualtyMisuseError,
    EVENT_REQUIRED_COLUMNS,
    analysis_ready_events,
    analysis_ready_mask,
    assert_no_killed_min_sum,
    killed_min_distribution,
    mask_non_cumulative_killed,
    refuse_sum_killed_min,
)

FAILURES: list[str] = []


def check_true(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    suffix = f": {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not cond:
        FAILURES.append(name)


def sample_events() -> pd.DataFrame:
    rows = [
        {
            "id": "e1",
            "title": "stated mapped incident",
            "date": "2024-01-02",
            "date_source": "stated",
            "place_id": "p1",
            "place": "Gaza City",
            "lat": 31.5,
            "lon": 34.5,
            "facility": "",
            "category": "airstrike",
            "category2": "",
            "evidence_type": "attack",
            "killed_min": 4,
            "killed_max": 4,
            "injured_min": 1,
            "injured_max": 1,
            "cumulative_min": "",
            "cumulative_max": "",
            "figures_disputed": 0,
            "methods": "airstrike",
            "perpetrators": "",
            "victim_groups": "children",
            "verification_tier": "A",
            "verification_anchor": "independent footage",
            "report_day": "",
            "dating": "stated",
            "n_clips": 2,
            "n_distinct_footage": 2,
            "n_archives": 2,
            "n_posts": 2,
            "mapped": 1,
        },
        {
            "id": "e2",
            "title": "post_proxy mapped",
            "date": "2024-01-03",
            "date_source": "post_proxy",
            "place_id": "p1",
            "place": "Gaza City",
            "lat": 31.5,
            "lon": 34.5,
            "facility": "",
            "category": "airstrike",
            "category2": "",
            "evidence_type": "attack",
            "killed_min": 3,
            "killed_max": 3,
            "injured_min": "",
            "injured_max": "",
            "cumulative_min": "",
            "cumulative_max": "",
            "figures_disputed": 0,
            "methods": "airstrike",
            "perpetrators": "",
            "victim_groups": "",
            "verification_tier": "D",
            "verification_anchor": "single clip",
            "report_day": "",
            "dating": "post_proxy",
            "n_clips": 1,
            "n_distinct_footage": 1,
            "n_archives": 1,
            "n_posts": 1,
            "mapped": 1,
        },
        {
            "id": "e3",
            "title": "stated unmapped",
            "date": "2024-01-04",
            "date_source": "stated",
            "place_id": "",
            "place": "",
            "lat": "",
            "lon": "",
            "facility": "",
            "category": "airstrike",
            "category2": "",
            "evidence_type": "attack",
            "killed_min": 2,
            "killed_max": 2,
            "injured_min": "",
            "injured_max": "",
            "cumulative_min": 0,
            "cumulative_max": 0,
            "figures_disputed": 0,
            "methods": "",
            "perpetrators": "",
            "victim_groups": "",
            "verification_tier": "C",
            "verification_anchor": "link",
            "report_day": "",
            "dating": "stated",
            "n_clips": 1,
            "n_distinct_footage": 1,
            "n_archives": 1,
            "n_posts": 1,
            "mapped": 0,
        },
        {
            "id": "e4",
            "title": "stated mapped cumulative total",
            "date": "2024-01-05",
            "date_source": "stated",
            "place_id": "p2",
            "place": "Khan Younis",
            "lat": 31.3,
            "lon": 34.3,
            "facility": "",
            "category": "statement",
            "category2": "",
            "evidence_type": "statement",
            "killed_min": 20000,
            "killed_max": 20000,
            "injured_min": "",
            "injured_max": "",
            "cumulative_min": 20000,
            "cumulative_max": 20000,
            "figures_disputed": 0,
            "methods": "",
            "perpetrators": "",
            "victim_groups": "",
            "verification_tier": "B",
            "verification_anchor": "report",
            "report_day": 90,
            "dating": "stated",
            "n_clips": 1,
            "n_distinct_footage": 1,
            "n_archives": 1,
            "n_posts": 1,
            "mapped": 1,
        },
        {
            "id": "e5",
            "title": "stated mapped high-kill cluster",
            "date": "2024-01-06",
            "date_source": "stated",
            "place_id": "p2",
            "place": "Khan Younis",
            "lat": 31.3,
            "lon": 34.3,
            "facility": "",
            "category": "massacre",
            "category2": "",
            "evidence_type": "attack",
            "killed_min": 12,
            "killed_max": 15,
            "injured_min": "",
            "injured_max": "",
            "cumulative_min": "",
            "cumulative_max": "",
            "figures_disputed": 0,
            "methods": "airstrike",
            "perpetrators": "",
            "victim_groups": "children",
            "verification_tier": "A",
            "verification_anchor": "independent footage",
            "report_day": "",
            "dating": "stated",
            "n_clips": 3,
            "n_distinct_footage": 3,
            "n_archives": 2,
            "n_posts": 3,
            "mapped": 1,
        },
    ]
    df = pd.DataFrame(rows)
    extra = [c for c in EVENT_REQUIRED_COLUMNS if c not in df.columns]
    check_true("sample covers required event columns", extra == [], str(extra))
    return df


def test_filters() -> None:
    events = sample_events()
    ready = analysis_ready_events(events, stated=True, mapped=True)
    check_true(
        "stated+mapped keeps e1,e4,e5",
        set(ready["id"]) == {"e1", "e4", "e5"},
        str(list(ready["id"])),
    )
    killed = analysis_ready_events(
        events, stated=True, mapped=True, non_cumulative_killed=True
    )
    check_true(
        "non-cumulative killed drops war-to-date row e4",
        set(killed["id"]) == {"e1", "e5"},
        str(list(killed["id"])),
    )
    check_true(
        "mask_non_cumulative_killed includes unmapped e3 and excludes e4",
        set(events.loc[mask_non_cumulative_killed(events), "id"]) == {"e1", "e2", "e3", "e5"},
    )
    mixed = analysis_ready_mask(events, stated=True, mapped=True)
    check_true("post_proxy excluded from default mask", not bool(mixed.iloc[1]))


def test_killed_min_never_summed() -> None:
    events = sample_events()
    stats = killed_min_distribution(events.loc[mask_non_cumulative_killed(events), "killed_min"])
    check_true("distribution has no sum key", "sum" not in stats, str(stats.keys()))
    check_true("n_with_killed_min is 4", stats["n_with_killed_min"] == 4)
    check_true("share_ge_10 is 0.25", abs(stats["share_ge_10"] - 0.25) < 1e-12)
    check_true("not_a_death_toll flag set", stats["not_a_death_toll"] is True)

    raised = False
    try:
        refuse_sum_killed_min(events["killed_min"])
    except CasualtyMisuseError:
        raised = True
    check_true("refuse_sum_killed_min raises", raised)

    raised = False
    try:
        assert_no_killed_min_sum({"killed_min_sum": 99})
    except CasualtyMisuseError:
        raised = True
    check_true("assert_no_killed_min_sum rejects killed_min_sum", raised)

    raised = False
    try:
        assert_no_killed_min_sum({"toll": {"sum_killed_min": 1}})
    except CasualtyMisuseError:
        raised = True
    check_true("assert_no_killed_min_sum rejects nested sum_killed_min", raised)

    ok = {"killed_min_distribution_non_cumulative": stats, "n_events": 5}
    try:
        assert_no_killed_min_sum(ok)
        nested_ok = True
    except CasualtyMisuseError:
        nested_ok = False
    check_true("safe artefact is accepted", nested_ok)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text(json.dumps({"total_killed_min": 123}), encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
        raised = False
        try:
            assert_no_killed_min_sum(payload)
        except CasualtyMisuseError:
            raised = True
        check_true("JSON artefact with total_killed_min is rejected", raised)


def test_ingest_sources_do_not_sum_killed_min() -> None:
    for path in HERE.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "sum":
                continue
            target = node.func.value
            if isinstance(target, ast.Attribute) and target.attr == "killed_min":
                hits.append(f"{path.name}:{node.lineno}")
            elif isinstance(target, ast.Subscript):
                sl = target.slice
                if isinstance(sl, ast.Constant) and sl.value == "killed_min":
                    hits.append(f"{path.name}:{node.lineno}")
        check_true(f"{path.name} does not call .sum on killed_min", hits == [], str(hits))
        check_true(
            f"{path.name} does not assign killed_min_sum",
            not any(
                isinstance(n, ast.Name) and n.id == "killed_min_sum"
                for n in ast.walk(tree)
            ),
        )


def test_build_pin() -> None:
    pinned = (HERE.parent / "BUILD_ID").read_text(encoding="utf-8").strip()
    check_true("BUILD_ID file matches code pin", pinned == BUILD_ID, pinned)
    check_true("pinned build is f76bd2fb", BUILD_ID == "f76bd2fb")
    source = json.loads((HERE / "PINNED_SOURCE.json").read_text(encoding="utf-8"))
    check_true("PINNED_SOURCE build_id", source["build_id"] == BUILD_ID)
    check_true(
        "PINNED_SOURCE lists event graph files only",
        all("torrent" not in f and "media" not in f for f in source["files"]),
    )
    check_true(
        "PINNED_SOURCE pins SHA256SUMS.txt hash",
        len(source.get("sha256_sha256sums", "")) == 64,
    )


def main() -> int:
    print("archive_graph ingest/schema-filter checks")
    test_build_pin()
    test_filters()
    test_killed_min_never_summed()
    test_ingest_sources_do_not_sum_killed_min()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} checks: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
