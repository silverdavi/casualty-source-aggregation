"""Spatial structure of documented events vs a population-proportional null.

Governorate x month event densities for the analysis-ready subset
(``date_source == stated`` and ``mapped == 1``), compared against a null in
which documented events are distributed across the five Gaza governorates in
proportion to their 2023 PCBS populations (read from ``gaza_sim/facts.json``,
which this script never modifies).

Interpretation constraints (from the archive's own README):
- An event is a cluster of footage, not a verified incident.
- Coordinates are gazetteer place anchors (town/site centroids), NOT surveyed
  strike locations. Results are named-place documentation intensity, not
  crater maps.
- A separate sensitivity run uses ``date_source == post_proxy`` dates and is
  reported under its own key, never mixed into the primary numbers.

Governorate assignment method
-----------------------------
Events carry a gazetteer place anchor (lat/lon of a town, camp, neighbourhood
or facility). The places table has no governorate column, so we assign each
event to a governorate by nearest curated anchor: a hand-built list of well
known Gaza localities with unambiguous governorate membership (cities, camps,
neighbourhoods, border towns). Events are first restricted to a Gaza Strip
bounding box (West Bank / Israel / region-level anchors are excluded and
counted in the coverage report). Region-level anchors ("Gaza Strip") cannot
be assigned and are reported as unassigned. Assignment coverage is written to
the aggregates file.

Outputs (counts only; no personal names, no captions):
- archive_graph/aggregates/spatial.json
- archive_graph/figures/spatial_gov_month_heatmap.png
- archive_graph/figures/spatial_gov_share_vs_population_null.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive_graph" / "ingest"))

import filters  # noqa: E402

RAW = ROOT / "archive_graph" / "raw"
AGG = ROOT / "archive_graph" / "aggregates"
FIG = ROOT / "archive_graph" / "figures"

# Gaza Strip bounding box (generous; excludes West Bank and Israel anchors).
GAZA_LAT = (31.20, 31.62)
GAZA_LON = (34.15, 34.60)

# Curated locality anchors with unambiguous governorate membership.
# (name, lat, lon, governorate) -- coordinates are approximate centroids.
GOV_ANCHORS = [
    # North Gaza
    ("Jabalia", 31.5282, 34.4832, "North Gaza"),
    ("Beit Lahia", 31.5494, 34.5021, "North Gaza"),
    ("Beit Hanoun", 31.5386, 34.5361, "North Gaza"),
    ("Jabalia Camp", 31.5350, 34.4970, "North Gaza"),
    ("Atatra", 31.5560, 34.4830, "North Gaza"),
    # Gaza
    ("Gaza City", 31.5129, 34.4581, "Gaza"),
    ("Shujaiyya", 31.5049, 34.4795, "Gaza"),
    ("Zeitoun", 31.4926, 34.4550, "Gaza"),
    ("Tel al-Hawa", 31.4919, 34.4306, "Gaza"),
    ("Sheikh Radwan", 31.5320, 34.4550, "Gaza"),
    ("Rimal", 31.5180, 34.4440, "Gaza"),
    ("Zahra", 31.4700, 34.4090, "Gaza"),
    # Deir al-Balah
    ("Deir al-Balah", 31.4180, 34.3510, "Deir al-Balah"),
    ("Nuseirat", 31.4480, 34.3900, "Deir al-Balah"),
    ("Bureij", 31.4400, 34.3900, "Deir al-Balah"),
    ("Maghazi", 31.4210, 34.3860, "Deir al-Balah"),
    ("Zawayda", 31.4270, 34.3650, "Deir al-Balah"),
    # Khan Younis
    ("Khan Younis", 31.3400, 34.3060, "Khan Younis"),
    ("Al-Mawasi", 31.3400, 34.2600, "Khan Younis"),
    ("Bani Suheila", 31.3420, 34.3300, "Khan Younis"),
    ("Abasan", 31.3230, 34.3420, "Khan Younis"),
    ("Qarara", 31.3720, 34.3300, "Khan Younis"),
    # Rafah
    ("Rafah", 31.2870, 34.2544, "Rafah"),
    ("Tal as-Sultan", 31.2970, 34.2330, "Rafah"),
    ("Shaboura", 31.2900, 34.2500, "Rafah"),
]

GOV_ORDER = ["North Gaza", "Gaza", "Deir al-Balah", "Khan Younis", "Rafah"]


def load_population_shares() -> dict[str, float]:
    facts = json.loads((ROOT / "gaza_sim" / "facts.json").read_text())
    govs = facts["geography"]["governorates"]
    pops = {g["name"]: float(g["population_2023"]) for g in govs}
    total = sum(pops.values())
    return {k: v / total for k, v in pops.items()}, pops, total


def assign_governorates(events: pd.DataFrame) -> pd.DataFrame:
    """Nearest curated anchor within the Gaza bounding box.

    Returns events with an added ``governorate`` column (NaN for region-level
    anchors) and an added ``in_gaza_bbox`` boolean column.
    """
    ev = events.copy()
    lat = pd.to_numeric(ev["lat"], errors="coerce")
    lon = pd.to_numeric(ev["lon"], errors="coerce")
    ev["in_gaza_bbox"] = (
        lat.between(*GAZA_LAT) & lon.between(*GAZA_LON)
    )
    # Region-level anchors cannot be assigned to a governorate.
    region_anchor = ev["place_id"].astype("string").str.startswith("REGION-")

    anchors = np.array([[a[1], a[2]] for a in GOV_ANCHORS])
    anchor_gov = np.array([a[3] for a in GOV_ANCHORS])
    # Equirectangular distance (fine at this scale); lon scaled by cos(lat).
    coslat = math.cos(math.radians(31.4))
    pts = np.column_stack([lat.to_numpy(), lon.to_numpy() * coslat])
    anc = np.column_stack([anchors[:, 0], anchors[:, 1] * coslat])
    d2 = ((pts[:, None, :] - anc[None, :, :]) ** 2).sum(axis=2)
    nearest = anchor_gov[np.nanargmin(d2, axis=1)]

    gov = pd.Series(nearest, index=ev.index, dtype="object")
    gov[~ev["in_gaza_bbox"] | region_anchor] = np.nan
    ev["governorate"] = gov
    ev["region_anchor"] = region_anchor
    return ev


def chi2_sf_even_df(x: float, df: int) -> float:
    """Survival function of chi-square with even df (closed form)."""
    assert df % 2 == 0
    k = df // 2
    term = 1.0
    total = 1.0
    for i in range(1, k):
        term *= (x / 2) / i
        total += term
    return float(math.exp(-x / 2) * total)


def gov_month_analysis(ev: pd.DataFrame, pop_shares: dict[str, float]) -> dict:
    assigned = ev.dropna(subset=["governorate"]).copy()
    assigned["month"] = assigned["date"].astype(str).str.slice(0, 7)

    gov_counts = (
        assigned["governorate"].value_counts().reindex(GOV_ORDER).fillna(0).astype(int)
    )
    n = int(gov_counts.sum())
    expected = np.array([pop_shares[g] * n for g in GOV_ORDER])
    observed = gov_counts.to_numpy(dtype=float)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    df = len(GOV_ORDER) - 1
    p = chi2_sf_even_df(chi2, df)

    gov_month = (
        assigned.groupby(["governorate", "month"]).size().unstack(fill_value=0)
    ).reindex(GOV_ORDER)

    # Concentration among place anchors (named-place intensity, not craters).
    place_counts = assigned["place_id"].value_counts()
    k10 = max(1, int(math.ceil(0.10 * len(place_counts))))
    top_decile_share = float(place_counts.iloc[:k10].sum() / place_counts.sum())
    # Same measure among facility-anchored events only.
    fac = assigned[assigned["facility"].astype("string").str.strip().ne("")]
    fac = fac.dropna(subset=["facility"])
    fac_counts = fac["place_id"].value_counts()
    fac_top_share = None
    if len(fac_counts) >= 10:
        kf = max(1, int(math.ceil(0.10 * len(fac_counts))))
        fac_top_share = float(fac_counts.iloc[:kf].sum() / fac_counts.sum())

    # Gini coefficient over place-anchor counts.
    v = np.sort(place_counts.to_numpy(dtype=float))
    cum = np.cumsum(v)
    gini = float(1 - 2 * (cum / cum[-1]).mean() + 1 / len(v)) if len(v) else None

    return {
        "n_events_assigned": n,
        "governorate_counts": {g: int(gov_counts[g]) for g in GOV_ORDER},
        "governorate_shares_observed": {
            g: round(float(gov_counts[g]) / n, 6) for g in GOV_ORDER
        },
        "governorate_shares_population_null": {
            g: round(pop_shares[g], 6) for g in GOV_ORDER
        },
        "chi2_vs_population_null": {
            "statistic": round(chi2, 3),
            "df": df,
            "p_value": p if p > 1e-300 else 0.0,
            "note": "goodness-of-fit of governorate event counts against "
            "2023 PCBS population shares from gaza_sim/facts.json",
        },
        "governorate_month_counts": {
            g: {m: int(c) for m, c in row.items() if c > 0}
            for g, row in gov_month.iterrows()
        },
        "place_anchor_concentration": {
            "n_distinct_place_anchors": int(len(place_counts)),
            "top_decile_place_share": round(top_decile_share, 6),
            "top_decile_place_share_facility_anchored_events": (
                round(fac_top_share, 6) if fac_top_share is not None else None
            ),
            "n_facility_anchored_events": int(len(fac)),
            "gini_over_place_anchor_counts": round(gini, 6),
            "note": "coordinates are gazetteer place anchors, not strike "
            "locations; this is named-place documentation intensity",
        },
    }


def main() -> None:
    events = pd.read_csv(RAW / "events.csv.gz", low_memory=False)
    pop_shares, pops, pop_total = load_population_shares()

    def run(date_source_stated: bool) -> tuple[dict, pd.DataFrame]:
        if date_source_stated:
            mask = filters.analysis_ready_mask(events)  # stated + mapped
        else:
            mask = filters.mask_mapped(events) & events["date_source"].astype(
                "string"
            ).str.strip().eq(filters.DATE_SOURCE_POST_PROXY)
        sub = events.loc[mask]
        ev = assign_governorates(sub)
        n_total = int(len(ev))
        n_outside = int((~ev["in_gaza_bbox"]).sum())
        n_region = int((ev["in_gaza_bbox"] & ev["region_anchor"]).sum())
        n_assigned = int(ev["governorate"].notna().sum())
        coverage = {
            "n_events_in_subset": n_total,
            "n_outside_gaza_bbox_excluded": n_outside,
            "n_region_level_anchor_unassigned": n_region,
            "n_assigned_to_governorate": n_assigned,
            "assignment_coverage_of_gaza_events": round(
                n_assigned / max(1, n_total - n_outside), 6
            ),
            "method": "nearest curated locality anchor (equirectangular "
            "distance) within a Gaza Strip bounding box; region-level "
            "gazetteer anchors (e.g. 'Gaza Strip') left unassigned; "
            "anchors listed in GOV_ANCHORS in 01_spatial.py",
        }
        result = {"coverage": coverage}
        result.update(gov_month_analysis(ev, pop_shares))
        return result, ev

    primary, ev_primary = run(date_source_stated=True)
    sensitivity, _ = run(date_source_stated=False)

    payload = {
        "build_id": filters.BUILD_ID,
        "unit": "documented footage-cluster events (upper bound on distinct "
        "incidents), not verified incidents and not a death toll",
        "coordinate_caveat": "coordinates are gazetteer place anchors "
        "(town/site centroids), not surveyed strike locations",
        "population_null": {
            "source": "gaza_sim/facts.json geography.governorates "
            "population_2023 (PCBS)",
            "populations_2023": {k: int(v) for k, v in pops.items()},
            "total": int(pop_total),
        },
        "primary_stated_dates_mapped": primary,
        "sensitivity_post_proxy_dates_mapped": {
            "note": "post_proxy publication-date proxy; reported separately, "
            "never mixed into the primary stated-date numbers",
            **sensitivity,
        },
    }
    filters.assert_no_killed_min_sum(payload)
    AGG.mkdir(parents=True, exist_ok=True)
    out = AGG / "spatial.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")

    make_figures(ev_primary, primary, pop_shares)


def make_figures(ev: pd.DataFrame, primary: dict, pop_shares: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)

    assigned = ev.dropna(subset=["governorate"]).copy()
    assigned["month"] = assigned["date"].astype(str).str.slice(0, 7)
    gov_month = (
        assigned.groupby(["governorate", "month"]).size().unstack(fill_value=0)
    ).reindex(GOV_ORDER)
    months = sorted(gov_month.columns)
    gov_month = gov_month[months]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(months)), 3.2))
    im = ax.imshow(gov_month.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(GOV_ORDER)), GOV_ORDER)
    ax.set_xticks(range(len(months)), months, rotation=90, fontsize=7)
    ax.set_title(
        "Documented events (stated dates, mapped) by governorate x month\n"
        "footage clusters at gazetteer place anchors — not strike locations"
    )
    fig.colorbar(im, ax=ax, label="events")
    fig.tight_layout()
    p1 = FIG / "spatial_gov_month_heatmap.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"wrote {p1}")

    obs = primary["governorate_shares_observed"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(GOV_ORDER))
    ax.bar(x - 0.2, [obs[g] for g in GOV_ORDER], width=0.4, label="observed share")
    ax.bar(
        x + 0.2,
        [pop_shares[g] for g in GOV_ORDER],
        width=0.4,
        label="population-proportional null (PCBS 2023)",
    )
    ax.set_xticks(x, GOV_ORDER, rotation=20)
    ax.set_ylabel("share of documented events")
    ax.set_title(
        "Governorate share of documented events vs population null\n"
        "(place-anchor intensity; events are footage clusters)"
    )
    ax.legend()
    fig.tight_layout()
    p2 = FIG / "spatial_gov_share_vs_population_null.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
