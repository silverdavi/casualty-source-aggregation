"""Name-level overlap between archive-named people and the MoH identified-dead
register.

From people.csv: shares of in_victims_list==1 (unambiguous register match) vs
victim_ambiguous==1 (a lead, not a confirmation) vs unmatched. Then counts of
events that name a register-matched person, by month (stated dates only) and
by governorate / top place anchor.

COUNTS ONLY: no personal name, key, or victim_id is ever written to the
published aggregate or figures. This shows the MoH identified list is
populated by identities that also appear in primary captions; it does NOT
estimate combatant share q and does not replace a household survey.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import common
import filters

# Gaza Strip governorate anchors (from places.csv gov-* rows, hard-coded to
# avoid depending on gazetteer row order). Used only for coarse counting.
GAZA_BBOX = (31.15, 31.65, 34.15, 34.65)  # lat_min, lat_max, lon_min, lon_max


def gaza_governorates(places: pd.DataFrame) -> pd.DataFrame:
    gov = places[places["tier"] == "governorate"].copy()
    lat_lo, lat_hi, lon_lo, lon_hi = GAZA_BBOX
    return gov[
        gov["lat"].between(lat_lo, lat_hi) & gov["lon"].between(lon_lo, lon_hi)
    ]


def assign_governorate(events: pd.DataFrame, govs: pd.DataFrame) -> pd.Series:
    """Nearest Gaza-governorate centroid for mapped events inside the Gaza
    bbox; everything else is 'outside_gaza_or_unresolved'. Coarse, for counts
    only — place anchors are gazetteer centroids, not strike points."""
    lat = pd.to_numeric(events["lat"], errors="coerce")
    lon = pd.to_numeric(events["lon"], errors="coerce")
    lat_lo, lat_hi, lon_lo, lon_hi = GAZA_BBOX
    in_gaza = lat.between(lat_lo, lat_hi) & lon.between(lon_lo, lon_hi)
    out = pd.Series("outside_gaza_or_unresolved", index=events.index)
    if govs.empty:
        return out
    coslat = math.cos(math.radians(31.4))
    best = None
    best_d = None
    for _, g in govs.iterrows():
        d = ((lat - g["lat"]) ** 2 + ((lon - g["lon"]) * coslat) ** 2)
        if best is None:
            best, best_d = pd.Series(g["name_en"], index=events.index), d
        else:
            closer = d < best_d
            best = best.where(~closer, g["name_en"])
            best_d = best_d.where(~closer, d)
    return out.where(~in_gaza, best)


def main() -> None:
    people = common.load("people")
    n_people = len(people)
    matched = pd.to_numeric(people["in_victims_list"], errors="coerce").eq(1)
    ambiguous = (
        pd.to_numeric(people["victim_ambiguous"], errors="coerce").eq(1) & ~matched
    )
    unmatched = ~matched & ~ambiguous

    events = common.load_events().set_index("id", drop=False)
    entities = common.load("event_entities")
    named_in = entities[entities["relation"] == "named_in"]
    matched_keys = set(people.loc[matched, "key"])
    events_naming_matched = set(
        named_in.loc[named_in["entity_key"].isin(matched_keys), "event_id"]
    )
    ev_mask = events["id"].isin(events_naming_matched)
    ev = events.loc[ev_mask].copy()

    stated = filters.mask_stated(ev)
    ev_stated = ev.loc[stated].copy()
    ev_stated["month"] = common.month_of(ev_stated["date"])
    by_month = (
        ev_stated["month"].value_counts().sort_index().astype(int).to_dict()
    )

    places = common.load("places")
    govs = gaza_governorates(places)
    ev["governorate"] = assign_governorate(ev, govs)
    by_gov = ev["governorate"].value_counts().astype(int).to_dict()
    # Top place anchors are public gazetteer place names, not personal names.
    by_place = ev["place"].value_counts().head(20).astype(int).to_dict()

    payload = {
        "note": (
            "Counts only; no personal names. in_victims_list is an unambiguous "
            "match to the MoH identified-dead register; victim_ambiguous is a "
            "lead, not a confirmation. This does not estimate combatant share."
        ),
        "people": {
            "n_people_named_in_archive": int(n_people),
            "n_register_matched": int(matched.sum()),
            "n_victim_ambiguous_only": int(ambiguous.sum()),
            "n_unmatched": int(unmatched.sum()),
            "share_register_matched": round(float(matched.mean()), 6),
            "share_victim_ambiguous_only": round(float(ambiguous.mean()), 6),
            "share_unmatched": round(float(unmatched.mean()), 6),
        },
        "events_naming_register_matched_person": {
            "n_events": int(ev_mask.sum()),
            "n_events_stated_date": int(stated.sum()),
            "by_month_stated_dates": by_month,
            "by_governorate_nearest_anchor": by_gov,
            "by_place_anchor_top20": by_place,
        },
    }
    common.write_aggregate("name_overlap", payload)

    fig, ax = plt.subplots(figsize=(10, 4))
    months = list(by_month)
    ax.bar(months, [by_month[m] for m in months])
    ax.set_ylabel("events naming a register-matched person")
    ax.set_title(
        f"Events naming an MoH-register-matched person, stated dates "
        f"(build {common.BUILD_ID})"
    )
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    common.savefig(fig, "name_overlap_by_month.png")


if __name__ == "__main__":
    main()
