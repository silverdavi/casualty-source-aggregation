"""Protected-site and victim-group documentation intensity.

Joins facilities + event_entities (relation ``struck``) + event category /
victim_groups. Outputs counts of hospital / school / shelter / mosque
facility-strike events and journalist / children-mentioned events, stratified
by verification_tier and n_distinct_footage.

This measures documentation intensity inside one archive. It is NOT a legal
finding, NOT a verified strike list, and event counts are upper bounds on
distinct incidents (footage clusters).
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common

# Keyword classes over facility names (English + common Arabic terms).
FACILITY_CLASSES = {
    "hospital_medical": r"hospital|medical|clinic|infirmary|مستشفى|مستوصف|عيادة",
    "school_university": r"school|university|college|kindergarten|مدرسة|جامعة|روضة",
    "shelter_camp": r"shelter|refugee camp|\bcamp\b|\brc\b|ملجأ|مخيم",
    "mosque_church": r"mosque|church|مسجد|كنيسة",
}

TIER_ORDER = ["A", "B", "C", "D", "E"]
FOOTAGE_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10**9, "5+")]


def footage_bin(n: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce")
    out = pd.Series("unknown", index=n.index)
    for lo, hi, label in FOOTAGE_BINS:
        out[(n >= lo) & (n <= hi)] = label
    return out


def strat_counts(events: pd.DataFrame, mask: pd.Series) -> dict:
    sub = events.loc[mask]
    by_tier = sub["verification_tier"].value_counts().to_dict()
    by_footage = footage_bin(sub["n_distinct_footage"]).value_counts().to_dict()
    return {
        "n_events": int(mask.sum()),
        "by_verification_tier": {t: int(by_tier.get(t, 0)) for t in TIER_ORDER},
        "by_n_distinct_footage": {
            label: int(by_footage.get(label, 0)) for _, _, label in FOOTAGE_BINS
        },
    }


def main() -> None:
    events = common.load_events().set_index("id", drop=False)
    entities = common.load("event_entities")
    struck = entities[entities["relation"] == "struck"]

    # Facility-strike events by keyword class of the struck facility name.
    # The name is used only for classification; only counts are published.
    facility_masks: dict[str, pd.Series] = {}
    for cls, pattern in FACILITY_CLASSES.items():
        rx = re.compile(pattern, re.IGNORECASE)
        hit_events = set(
            struck.loc[
                struck["entity_name"].astype(str).str.contains(rx, na=False),
                "event_id",
            ]
        )
        facility_masks[cls] = events["id"].isin(hit_events)

    vg = events["victim_groups"].astype(str).str.lower()
    cat = events["category"].astype(str).str.lower()
    cat2 = events["category2"].astype(str).str.lower()
    group_masks = {
        "journalists_mentioned": vg.str.contains("journalist")
        | cat.eq("press")
        | cat2.eq("press"),
        "children_mentioned": vg.str.contains("child"),
        "medics_mentioned": vg.str.contains("medic")
        | vg.str.contains("civil_defence"),
    }

    any_struck = events["id"].isin(set(struck["event_id"]))
    payload = {
        "note": (
            "Documentation intensity within one archive; events are footage "
            "clusters and counts are upper bounds on distinct incidents. Not "
            "a legal finding."
        ),
        "n_events_total": int(len(events)),
        "n_events_with_struck_facility": int(any_struck.sum()),
        "facility_strikes": {
            cls: strat_counts(events, m) for cls, m in facility_masks.items()
        },
        "victim_groups": {
            grp: strat_counts(events, m) for grp, m in group_masks.items()
        },
    }
    common.write_aggregate("protected_sites", payload)

    # Figure: counts by class, stacked by tier.
    labels = list(facility_masks) + list(group_masks)
    tiers_present = [
        t
        for t in TIER_ORDER
        if any(
            payload[sec][k]["by_verification_tier"][t]
            for sec, keys in (
                ("facility_strikes", facility_masks),
                ("victim_groups", group_masks),
            )
            for k in keys
        )
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0] * len(labels)
    for tier in tiers_present:
        vals = []
        for lab in labels:
            sec = "facility_strikes" if lab in facility_masks else "victim_groups"
            vals.append(payload[sec][lab]["by_verification_tier"][tier])
        ax.bar(labels, vals, bottom=bottom, label=f"tier {tier}")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("documented events (footage clusters)")
    ax.set_title(
        f"Protected-site / victim-group documentation by verification tier "
        f"(build {common.BUILD_ID})"
    )
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    common.savefig(fig, "protected_sites_by_tier.png")


if __name__ == "__main__":
    main()
