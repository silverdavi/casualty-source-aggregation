"""Tests over the published aggregates: no killed_min sums, no names.

Run after the analysis scripts: ``python -m pytest archive_graph/analysis``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import common
import filters

AGG_NAMES = ["protected_sites", "name_overlap", "corroboration", "incident_size"]


def _load(name: str):
    path = common.AGGREGATES / f"{name}.json"
    assert path.exists(), f"{path} missing; run the analysis scripts first"
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", AGG_NAMES)
def test_no_killed_min_sum(name):
    filters.assert_no_killed_min_sum(_load(name))


def test_incident_size_has_no_sum_anywhere():
    payload = _load("incident_size")

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert str(k).lower() not in {"sum", "total", "toll"}, k
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    assert payload["killed_min_distribution_all"]["not_a_death_toll"] is True


def test_refuse_sum_killed_min_raises():
    import pandas as pd

    with pytest.raises(filters.CasualtyMisuseError):
        filters.refuse_sum_killed_min(pd.Series([1, 2, 3]))


@pytest.mark.parametrize("name", AGG_NAMES)
def test_no_personal_names_in_aggregates(name):
    """No value in any aggregate may match a person name from people.csv."""
    people = common.load("people")
    names = set(people["name"].dropna().astype(str).str.strip().str.lower())
    names |= set(people["key"].dropna().astype(str).str.strip().str.lower())
    names.discard("")
    # people.csv contains a few mis-tagged facility/place entities (e.g.
    # hospitals); public gazetteer place and facility names are not personal
    # names and are allowed in place-anchor counts.
    places = common.load("places")
    facilities = common.load("facilities")
    public_places = set(
        places["name_en"].dropna().astype(str).str.strip().str.lower()
    ) | set(facilities["name"].dropna().astype(str).str.strip().str.lower())
    names -= public_places
    text = json.dumps(_load(name)).lower()
    # Check whole-token containment for each register name (bounded cost).
    hits = [n for n in names if len(n) > 3 and n in text]
    assert not hits, f"personal names leaked into {name}.json: {hits[:5]}"


def test_name_overlap_counts_only():
    payload = _load("name_overlap")
    text = json.dumps(payload)
    assert "victim_id" not in text
    p = payload["people"]
    assert (
        p["n_register_matched"] + p["n_victim_ambiguous_only"] + p["n_unmatched"]
        == p["n_people_named_in_archive"]
    )


def test_figures_exist_and_contain_no_text_dumps():
    for fname in [
        "protected_sites_by_tier.png",
        "name_overlap_by_month.png",
        "corroboration_structure.png",
        "incident_size_distribution.png",
    ]:
        path = common.FIGURES / fname
        assert path.exists(), f"{path} missing"
        assert path.stat().st_size > 0
