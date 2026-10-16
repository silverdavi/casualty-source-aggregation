"""Incident-size distribution — NEVER a death toll.

Distribution of caption-stated killed_min among non-cumulative rows
(quantiles, share with killed_min>=10, outlier flagging), cross-tabbed with
report_day presence (a daily MoH report exists for the date; not event-level
confirmation).

Known data issue: a handful of rows carry mis-filed war-to-date totals in
killed_min (max observed 514,000). These are flagged as outliers
(killed_min > OUTLIER_THRESHOLD) and quantiles are reported both with and
without them. Summing killed_min is forbidden and tested.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import filters

# Above this, a caption figure for a single footage cluster is treated as a
# mis-filed cumulative/war total, not an incident size. The deadliest single
# documented strikes are in the hundreds.
OUTLIER_THRESHOLD = 1000


def main() -> None:
    events = common.load_events()
    mask = filters.analysis_ready_mask(
        events, stated=True, mapped=True, non_cumulative_killed=True
    )
    sub = events.loc[mask].copy()
    killed = pd.to_numeric(sub["killed_min"], errors="coerce")

    outlier = killed > OUTLIER_THRESHOLD
    clean = killed[~outlier]

    # Broader sanity slice: all non-cumulative rows regardless of date source
    # or mapping, where the extreme mis-filed totals (max 514,000) live.
    nc_mask = filters.mask_non_cumulative_killed(events)
    nc_killed = pd.to_numeric(events.loc[nc_mask, "killed_min"], errors="coerce")
    nc_outlier = nc_killed > OUTLIER_THRESHOLD

    # Explicit guard: summing killed_min raises.
    try:
        filters.refuse_sum_killed_min(killed)
        raise AssertionError("refuse_sum_killed_min failed to raise")
    except filters.CasualtyMisuseError:
        pass

    has_report_day = sub["report_day"].notna() & (
        sub["report_day"].astype(str).str.strip() != ""
    )

    def dist(values: pd.Series) -> dict:
        return filters.killed_min_distribution(values)

    payload = {
        "note": (
            "Caption-stated cluster sizes among stated-date, mapped, "
            "non-cumulative rows. NEVER a death toll: duplicates and "
            "over-split footage make any sum meaningless; a sum-guard test "
            "is enforced on this file."
        ),
        "filters": {
            "stated_dates": True,
            "mapped": True,
            "non_cumulative_killed": True,
            "n_rows": int(mask.sum()),
        },
        "outliers": {
            "threshold_killed_min_gt": OUTLIER_THRESHOLD,
            "n_flagged_as_misfiled_totals": int(outlier.sum()),
            "max_flagged_value": float(killed[outlier].max())
            if outlier.any()
            else None,
            "note": (
                "Values above threshold are mis-filed war-to-date totals, not "
                "single-incident figures; excluded from the cleaned "
                "distribution below."
            ),
        },
        "all_non_cumulative_rows_any_date_source": {
            "n_rows": int(nc_mask.sum()),
            "n_flagged_as_misfiled_totals": int(nc_outlier.sum()),
            "max_flagged_value": float(nc_killed[nc_outlier].max())
            if nc_outlier.any()
            else None,
            **dist(nc_killed[~nc_outlier]),
        },
        "killed_min_distribution_all": dist(killed),
        "killed_min_distribution_excl_outliers": dist(clean),
        "by_report_day_presence": {
            "with_report_day": {
                "n_rows": int(has_report_day.sum()),
                **dist(killed[has_report_day & ~outlier]),
            },
            "without_report_day": {
                "n_rows": int((~has_report_day).sum()),
                **dist(killed[~has_report_day & ~outlier]),
            },
            "note": (
                "report_day means a daily MoH report exists for the date, not "
                "event-level confirmation."
            ),
        },
    }
    common.write_aggregate("incident_size", payload)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(0, np.log10(max(clean.max(), 10)), 40)
    ax.hist(
        [clean[has_report_day[clean.index]], clean[~has_report_day[clean.index]]],
        bins=bins,
        stacked=True,
        label=["daily report exists for date", "no daily report for date"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("caption-stated killed_min per footage cluster (not a toll)")
    ax.set_ylabel("events")
    ax.set_title(
        f"Incident-size distribution, non-cumulative rows, outliers "
        f">{OUTLIER_THRESHOLD} excluded (build {common.BUILD_ID})"
    )
    ax.legend()
    common.savefig(fig, "incident_size_distribution.png")


if __name__ == "__main__":
    main()
