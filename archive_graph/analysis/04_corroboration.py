"""Corroboration structure inside the archive.

Distribution of verification tiers A-E; n_archives x n_distinct_footage
cross-tab; figures_disputed rate; event-relation kinds (established structural
signals such as shared_entity / same_day / repeat_target / shared_person vs
kind=possible_same_event links).

verification_tier is corroboration inside this archive, not truth; tier D is
"single source in this archive", not "false".
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

TIER_ORDER = ["A", "B", "C", "D", "E"]
BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10**9, "5+")]

POSSIBLE_KINDS = {"possible_same_event"}


def binned(n: pd.Series) -> pd.Series:
    n = pd.to_numeric(n, errors="coerce")
    out = pd.Series("unknown", index=n.index)
    for lo, hi, label in BINS:
        out[(n >= lo) & (n <= hi)] = label
    return out


def main() -> None:
    events = common.load_events()
    relations = common.load("event_relations")

    tiers = events["verification_tier"].value_counts().to_dict()
    tier_counts = {t: int(tiers.get(t, 0)) for t in TIER_ORDER}

    arch = binned(events["n_archives"])
    foot = binned(events["n_distinct_footage"])
    labels = [label for _, _, label in BINS]
    xtab = pd.crosstab(arch, foot).reindex(index=labels, columns=labels).fillna(0)
    cross = {
        f"n_archives={a}": {
            f"n_distinct_footage={f}": int(xtab.loc[a, f]) for f in labels
        }
        for a in labels
    }

    disputed = pd.to_numeric(events["figures_disputed"], errors="coerce").eq(1)
    multi_footage = pd.to_numeric(events["n_distinct_footage"], errors="coerce") >= 2
    multi_archive = pd.to_numeric(events["n_archives"], errors="coerce") >= 2

    rel_kinds = relations["kind"].value_counts().astype(int).to_dict()
    n_possible = int(
        relations["kind"].isin(POSSIBLE_KINDS).sum()
    )

    payload = {
        "note": (
            "verification_tier is corroboration inside this archive, not "
            "truth; tier D means single-source here, not false. Event counts "
            "are footage clusters (upper bounds on incidents)."
        ),
        "n_events_total": int(len(events)),
        "verification_tier_counts": tier_counts,
        "verification_tier_shares": {
            t: round(c / len(events), 6) for t, c in tier_counts.items()
        },
        "n_archives_x_n_distinct_footage": cross,
        "share_with_2plus_distinct_footage": round(float(multi_footage.mean()), 6),
        "share_with_2plus_archives": round(float(multi_archive.mean()), 6),
        "figures_disputed": {
            "n_disputed": int(disputed.sum()),
            "rate": round(float(disputed.mean()), 6),
        },
        "event_relation_kinds": {
            "counts": rel_kinds,
            "n_established_structural_links": int(len(relations) - n_possible),
            "n_possible_same_event_links": n_possible,
            "note": (
                "kind=possible_same_event flags candidate duplicate incidents; "
                "other kinds are established structural signals (shared "
                "entity/person, same day, repeat target)."
            ),
        },
    }
    common.write_aggregate("corroboration", payload)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(TIER_ORDER, [tier_counts[t] for t in TIER_ORDER])
    axes[0].set_yscale("log")
    axes[0].set_title("Verification tier (within-archive corroboration)")
    axes[0].set_ylabel("events (log scale)")

    mat = xtab.to_numpy(dtype=float)
    im = axes[1].imshow(np.log10(mat + 1), cmap="viridis")
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_yticks(range(len(labels)), labels)
    axes[1].set_xlabel("n_distinct_footage")
    axes[1].set_ylabel("n_archives")
    axes[1].set_title("Archives x distinct footage (log10 count)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, int(mat[i, j]), ha="center", va="center",
                         color="white", fontsize=7)
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.suptitle(f"Corroboration structure (build {common.BUILD_ID})")
    common.savefig(fig, "corroboration_structure.png")


if __name__ == "__main__":
    main()
