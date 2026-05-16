"""Master multi-panel figure for the wars-1900-2026 dataset.

5 rows x 3 columns:
  Col 1 — overall scale / timeline / top conflicts
  Col 2 — composition of deaths and indirect-killing focus
  Col 3 — MILITARY vs CIVILIAN ratios, differences, asymmetry

Marker SIZE convention: when a marker's size encodes a death count, the
*area* of the marker is proportional to the count (matplotlib's scatter
`s` parameter is already area in points^2, so we just scale linearly --
no sqrt'ing of the value first, which would have made *radius*
proportional to the count instead).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import colormaps

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "per_war"


# ---------------------------------------------------------------------------
# Load + aggregate per-war
# ---------------------------------------------------------------------------

def _mid(lo, hi):
    if lo is None and hi is None: return None
    if lo is None: return hi
    if hi is None: return lo
    return (lo + hi) / 2


def load_wars():
    out = []
    for d in sorted(DATA.glob("*.json")):
        # Skip backup files (e.g. `*.v1_backup.json`).
        if d.stem.endswith("_backup") or d.stem.startswith("_"):
            continue
        j = json.loads(d.read_text())
        sides = j.get("sides") or []
        mil_lo = mil_hi = civ_dir_lo = civ_dir_hi = ind_lo = ind_hi = 0.0
        ind_by_type: dict[str, float] = {}
        for s in sides:
            mk = s.get("military_killed") or {}
            cd = s.get("civilians_killed_directly") or {}
            mil_lo += mk.get("low") or 0
            mil_hi += mk.get("high") or mk.get("low") or 0
            civ_dir_lo += cd.get("low") or 0
            civ_dir_hi += cd.get("high") or cd.get("low") or 0
            for ind in s.get("deaths_from_actions") or []:
                lo = ind.get("low") or 0
                hi = ind.get("high") or ind.get("low") or 0
                ind_lo += lo
                ind_hi += hi
                t = (ind.get("type") or "other").lower()
                ind_by_type[t] = ind_by_type.get(t, 0) + (lo + hi) / 2

        totals = j.get("totals") or {}
        g_lo = totals.get("grand_low")  or (mil_lo + civ_dir_lo + ind_lo)
        g_hi = totals.get("grand_high") or (mil_hi + civ_dir_hi + ind_hi)

        out.append({
            "id":        j.get("war_id", d.stem),
            "name":      j.get("name", d.stem),
            "start":     j.get("start_year"),
            "end":       j.get("end_year") or 2026,
            "ongoing":   bool(j.get("ongoing")),
            "regions":   j.get("regions") or [],
            "mil_lo":    mil_lo,    "mil_hi":    mil_hi,
            "civ_dir_lo": civ_dir_lo, "civ_dir_hi": civ_dir_hi,
            "ind_lo":    ind_lo,    "ind_hi":    ind_hi,
            "tot_lo":    g_lo,      "tot_hi":    g_hi,
            "mil_mid":   (mil_lo + mil_hi) / 2,
            "civ_dir_mid": (civ_dir_lo + civ_dir_hi) / 2,
            "ind_mid":   (ind_lo + ind_hi) / 2,
            "tot_mid":   _mid(g_lo, g_hi) or 0,
            "ind_by_type": ind_by_type,
        })
    return out


_ALL_WARS = load_wars()


# ---------------------------------------------------------------------------
# CLI: filter window + min death threshold
# ---------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--since", type=int, default=1900,
                help="Drop conflicts that ended before this year")
ap.add_argument("--until", type=int, default=2026,
                help="Right edge of the time-range plots")
ap.add_argument("--min-deaths", type=int, default=0,
                help="Drop conflicts whose mid-estimate total is below this")
ap.add_argument("--out", default="figure",
                help="Output basename (writes <out>.png and <out>.pdf)")
ap.add_argument("--title-suffix", default="",
                help="Extra text appended to the suptitle")
args = ap.parse_args()

WARS = [w for w in _ALL_WARS
        if (w["end"] or 0) >= args.since
        and (w["tot_mid"] or 0) >= args.min_deaths]
SINCE = args.since
UNTIL = args.until
print(f"Plotting {len(WARS)}/{len(_ALL_WARS)} conflicts "
      f"(since={SINCE}, min_deaths={args.min_deaths})")


# ---------------------------------------------------------------------------
# Region grouping
# ---------------------------------------------------------------------------

REGION_MAP = {
    "Europe":      ["europ", "spain", "irish", "balkan", "yugoslav", "kosovo",
                    "chechn", "bosnia", "ireland", "uk", "german", "ukrain",
                    "russia", "ussr", "soviet", "atlantic", "mediterranean",
                    "donbas"],
    "MENA":        ["israel", "palestin", "gaza", "syria", "iraq", "iran",
                    "yemen", "lebanon", "libya", "algeri", "morocco", "rif",
                    "egypt", "kuwait", "ottoman", "anatolia", "mideast",
                    "middle east", "sinai", "golan", "armenia", "turk"],
    "Africa":      ["congo", "rwanda", "sudan", "darfur", "biafra", "nigeria",
                    "angola", "mozambi", "ethiopi", "tigray", "somal",
                    "kenya", "namibia", "herero", "south africa", "guin",
                    "uganda", "boko", "lake chad", "mali", "burkina"],
    "Asia":        ["china", "japan", "korea", "vietnam", "cambodia", "lao",
                    "indochina", "myanmar", "burma", "rohingya", "philippin",
                    "indonesia", "timor", "afghan", "kashmir", "india",
                    "pakistan", "bangladesh", "sri lanka", "malay",
                    "manchuria", "asia-pac", "asia"],
    "Americas":    ["mexic", "guatemala", "salvador", "nicaragu", "colomb",
                    "chaco", "paraguay", "bolivia", "philippines", "us",
                    "americ", "cuba", "haiti"],
    "Caucasus":    ["nagorno", "karabakh", "azerbai", "armenia", "georgia",
                    "chechn"],
}


def region_of(w) -> str:
    blob = " ".join(w["regions"]).lower() + " " + w["name"].lower()
    if "world war" in blob or w["id"].startswith("wwi") or "global" in blob:
        return "Global (World Wars)"
    for region, keys in REGION_MAP.items():
        for k in keys:
            if k in blob:
                return region
    return "Other"


for w in WARS:
    w["region"] = region_of(w)
    civ_total = w["civ_dir_mid"] + w["ind_mid"]
    w["civ_total_mid"] = civ_total
    denom = max(civ_total + w["mil_mid"], 1)
    w["civ_share"] = civ_total / denom
    w["civ_minus_mil"] = civ_total - w["mil_mid"]


REGION_COLORS = {
    "Global (World Wars)": "#222222",
    "Europe":              "#1f77b4",
    "MENA":                "#d62728",
    "Africa":              "#2ca02c",
    "Asia":                "#ff7f0e",
    "Americas":            "#9467bd",
    "Caucasus":            "#8c564b",
    "Other":               "#888888",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_log_humans(x, _):
    if x <= 0: return ""
    if x >= 1e9: return f"{x/1e9:g}B"
    if x >= 1e6: return f"{x/1e6:g}M"
    if x >= 1e3: return f"{x/1e3:g}k"
    return f"{int(x)}"


# Marker AREA proportional to deaths (matplotlib `s` is points^2 = area).
def area_size(deaths, ref=1e7, ref_area=1500.0, floor=10.0):
    return np.maximum(floor, np.asarray(deaths, dtype=float) * (ref_area / ref))


def short(name, n=32):
    s = name.split("(")[0].strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def deconflict(ax, texts, **kw):
    """Wrap adjustText if available, otherwise no-op."""
    if HAS_ADJUST and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.4, alpha=0.5),
            expand=(1.2, 1.4),
            force_static=0.4, force_text=0.6, force_explode=0.4,
            min_arrow_len=4, **kw,
        )


# ---------------------------------------------------------------------------
# Figure setup — 5 rows x 3 cols, larger size + tighter layout control
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.titlepad": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.framealpha": 0.92,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.4,
})

fig = plt.figure(figsize=(28, 34))
gs = fig.add_gridspec(
    5, 3,
    hspace=0.70,                   # extra vertical room for titles + tick labels
    wspace=0.40,                   # extra horizontal room for long y-tick labels
    left=0.055, right=0.985,
    top=0.955,  bottom=0.025,
)

# ============================ COL 1: scale / timeline =====================

# ------- (1,1) Timeline ---------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for w in WARS:
    y = max(w["tot_mid"], 1)
    ax.hlines(y, w["start"], w["end"],
              colors=REGION_COLORS[w["region"]], alpha=0.35, lw=1.2)
ax.scatter(
    [(w["start"] + w["end"]) / 2 for w in WARS],
    [max(w["tot_mid"], 1) for w in WARS],
    s=area_size([w["tot_mid"] for w in WARS], ref=1e7, ref_area=1300.0),
    c=[REGION_COLORS[w["region"]] for w in WARS],
    alpha=0.7, edgecolors="black", linewidths=0.4, zorder=3,
)
texts = []
for w in sorted(WARS, key=lambda w: -w["tot_mid"])[:7]:
    texts.append(ax.text((w["start"] + w["end"]) / 2,
                         max(w["tot_mid"], 1),
                         short(w["name"], 26), fontsize=7,
                         ha="center", va="bottom"))
deconflict(ax, texts)
ax.set_yscale("log")
ax.set_xlim(SINCE - 2, UNTIL + 2)
ax.set_ylim(1e2, 1e9)
ax.set_xlabel("Year"); ax.set_ylabel("Total deaths (mid, log)")
ax.set_title("(1) Timeline — line=duration, area∝deaths, color=region")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.grid(True, which="both", alpha=0.25)
# Region legend BELOW the plot, not floating in the data
handles = [Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=c, markeredgecolor="black",
                  markersize=6, label=r)
           for r, c in REGION_COLORS.items()]
ax.legend(handles=handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=7,
          frameon=True, borderaxespad=0)

# ------- (2,1) Top-20 range bars ------------------------------------------
ax = fig.add_subplot(gs[1, 0])
top20 = sorted(WARS, key=lambda w: -w["tot_hi"])[:20]
y = np.arange(len(top20))
los = np.array([max(w["tot_lo"], 1) for w in top20])
his = np.array([max(w["tot_hi"], 1) for w in top20])
colors = [REGION_COLORS[w["region"]] for w in top20]
ax.barh(y, his - los, left=los, color=colors, alpha=0.45,
        edgecolor="black", linewidth=0.4)
ax.scatter((los + his) / 2, y, color=colors, s=35,
           edgecolors="black", linewidths=0.5, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([short(w["name"], 36) for w in top20], fontsize=7)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(left=max(1e3, los.min() * 0.7))
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlabel("Deaths low–high (log)")
ax.set_title("(2) Top-20 conflicts by upper bound")
ax.grid(True, axis="x", alpha=0.25, which="both")

# ------- (3,1) Cumulative deaths timeline ----------------------------------
ax = fig.add_subplot(gs[2, 0])
years = np.arange(SINCE - 1, UNTIL + 1)
per_year = np.zeros_like(years, dtype=float)
for w in WARS:
    span = max(w["end"] - w["start"], 1)
    rate = w["tot_mid"] / span
    for yr in range(w["start"], w["end"] + 1):
        idx = yr - years[0]
        if 0 <= idx < len(years):
            per_year[idx] += rate
cum = np.cumsum(per_year)
ax.fill_between(years, cum, alpha=0.3, color="#b8313a")
ax.plot(years, cum, color="#b8313a", lw=1.6)
ALL_MARKERS = [("WWI", 1918), ("WWII end", 1945),
               ("Mao famine", 1962), ("Cold-war proxies", 1980),
               ("post-2010 wars", 2020), ("Russia–Ukraine", 2022),
               ("Gaza war", 2023)]
texts = []
for label, yr in ALL_MARKERS:
    if not (SINCE <= yr <= UNTIL):
        continue
    idx = yr - years[0]
    texts.append(ax.text(yr, cum[idx], label, fontsize=8,
                         ha="center", va="bottom"))
deconflict(ax, texts)
ax.set_xlabel("Year"); ax.set_ylabel("Cumulative deaths (mid)")
ax.set_xlim(SINCE - 2, UNTIL + 2)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_title(f"(3) Cumulative war/atrocity deaths, {SINCE}→{UNTIL}")
ax.grid(True, alpha=0.25)

# ------- (4,1) Deaths per decade, stacked by region -----------------------
ax = fig.add_subplot(gs[3, 0])
decades = list(range((SINCE // 10) * 10, ((UNTIL // 10) + 1) * 10 + 1, 10))
matrix = {r: np.zeros(len(decades)) for r in REGION_COLORS}
for w in WARS:
    span = max(w["end"] - w["start"], 1)
    rate = w["tot_mid"] / span
    for d_idx, d in enumerate(decades):
        overlap = max(0, min(w["end"], d + 10) - max(w["start"], d))
        if overlap > 0:
            matrix[w["region"]][d_idx] += rate * overlap
bottom = np.zeros(len(decades))
for r in REGION_COLORS:
    ax.bar(decades, matrix[r], width=8, bottom=bottom,
           color=REGION_COLORS[r], label=r, edgecolor="black", linewidth=0.3)
    bottom += matrix[r]
ax.set_xlabel("Decade"); ax.set_ylabel("Deaths attributed to that decade")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_title("(4) Deaths per decade, stacked by region")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=7, frameon=True, borderaxespad=0)
ax.grid(True, axis="y", alpha=0.25)

# ------- (5,1) Recent / ongoing wars -------------------------------------
ax = fig.add_subplot(gs[4, 0])
_recent_cut = max(SINCE, UNTIL - 16)  # ~last 16 years of the window
recent = sorted([w for w in WARS
                 if w["end"] >= _recent_cut and w["start"] >= _recent_cut - 4],
                key=lambda w: -w["tot_hi"])[:30]
y = np.arange(len(recent))
ax.barh(y, [max(w["tot_hi"], 1) - max(w["tot_lo"], 1) for w in recent],
        left=[max(w["tot_lo"], 1) for w in recent],
        color=[REGION_COLORS[w["region"]] for w in recent],
        alpha=0.5, edgecolor="black", linewidth=0.4)
ax.scatter([(max(w["tot_lo"], 1) + max(w["tot_hi"], 1)) / 2 for w in recent],
           y, color=[REGION_COLORS[w["region"]] for w in recent],
           s=70, edgecolors="black", linewidths=0.5, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{short(w['name'], 30)}  ({w['start']}–"
                    f"{'now' if w['ongoing'] else w['end']})"
                    for w in recent], fontsize=7)
ax.invert_yaxis()
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlabel("Deaths low–high (log)")
ax.set_title("(5) Recent / ongoing conflicts (2010s–2020s)")
ax.grid(True, axis="x", alpha=0.25, which="both")


# ============================ COL 2: composition / indirect ===============

# ------- (1,2) Stacked top-25 mil/civ-direct/indirect ---------------------
ax = fig.add_subplot(gs[0, 1])
top25 = sorted(WARS, key=lambda w: -w["tot_mid"])[:25]
y = np.arange(len(top25))
mil = np.array([w["mil_mid"]     for w in top25])
civ = np.array([w["civ_dir_mid"] for w in top25])
ind = np.array([w["ind_mid"]     for w in top25])
ax.barh(y, mil, color="#3b6fb6", label="Military killed")
ax.barh(y, civ, left=mil, color="#d4a017", label="Civilians killed directly")
ax.barh(y, ind, left=mil + civ, color="#b8313a",
        label="Indirect (camps/famine/blockade)")
ax.set_yticks(y)
ax.set_yticklabels([short(w["name"], 36) for w in top25], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Deaths (mid-estimate)")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_title("(6) Cause-of-death stack, top-25 by total")
ax.legend(loc="lower right", framealpha=0.95, fontsize=7)
ax.grid(True, axis="x", alpha=0.25)

# ------- (2,2) Direct civilian vs indirect civilian (log-log) -------------
ax = fig.add_subplot(gs[1, 1])
xs = np.array([max(w["civ_dir_mid"], 1) for w in WARS])
ys = np.array([max(w["ind_mid"],     1) for w in WARS])
sz = area_size([w["tot_mid"] for w in WARS], ref=1e7, ref_area=1500.0)
cl = [REGION_COLORS[w["region"]] for w in WARS]
ax.scatter(xs, ys, s=sz, c=cl, alpha=0.65,
           edgecolors="black", linewidths=0.4)
lim = [1, 1e8]
ax.plot(lim, lim, "--", color="grey", lw=0.8, label="indirect = direct")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlabel("Civilians killed DIRECTLY")
ax.set_ylabel("Indirect civilian deaths")
ax.set_title("(7) Direct vs indirect civilian — area∝total")
texts = []
for w in sorted(WARS, key=lambda w: -w["ind_mid"])[:8]:
    texts.append(ax.text(max(w["civ_dir_mid"], 1),
                         max(w["ind_mid"], 1),
                         short(w["name"], 24), fontsize=7,
                         ha="center", va="bottom"))
deconflict(ax, texts)
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, which="both", alpha=0.25)

# ------- (3,2) Civilian share vs scale ------------------------------------
ax = fig.add_subplot(gs[2, 1])
xs, ys, sz, cl, names = [], [], [], [], []
for w in WARS:
    tot = w["civ_total_mid"] + w["mil_mid"]
    if tot < 1000: continue
    xs.append(tot); ys.append(100 * w["civ_share"])
    sz.append(tot); cl.append(REGION_COLORS[w["region"]]); names.append(w["name"])
ax.scatter(xs, ys, s=area_size(sz, ref=1e7, ref_area=1300.0),
           c=cl, alpha=0.65, edgecolors="black", linewidths=0.4)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlabel("Total deaths (log)"); ax.set_ylabel("Civilian share (%)")
ax.set_ylim(-8, 110)
ax.axhline(50, color="grey", ls="--", lw=0.7)
ax.set_title("(8) Civilian share vs scale — area∝total")
# Pick a *small* set of distinguished labels and let adjustText space them
labelable = (sorted(zip(xs, ys, names), key=lambda t: -t[0])[:4]
             + sorted(zip(xs, ys, names), key=lambda t:  t[1])[:3])
seen = set()
texts = []
for x, yv, n in labelable:
    if n in seen: continue
    seen.add(n)
    texts.append(ax.text(x, yv, short(n, 26), fontsize=7,
                         ha="center", va="bottom"))
deconflict(ax, texts)
ax.grid(True, which="both", alpha=0.25)

# ------- (4,2) Indirect deaths by cause ------------------------------------
ax = fig.add_subplot(gs[3, 1])
totals_by_cause: dict[str, float] = {}
for w in WARS:
    for t, v in w["ind_by_type"].items():
        key = t.replace("_deaths", "").replace("-", "_")
        if   "famine" in key or "starv" in key or "blockade" in key:
            key = "famine / starvation / blockade"
        elif "camp" in key or "exterm" in key:
            key = "camps"
        elif "disease" in key or "epidem" in key:
            key = "disease / epidemics"
        elif "displac" in key or "deport" in key or "ethnic_cleans" in key:
            key = "displacement / ethnic cleansing"
        elif "forced_labor" in key or "slave" in key:
            key = "forced labor"
        elif "scorched" in key or "destruction" in key:
            key = "scorched-earth / infrastructure"
        else:
            key = "other"
        totals_by_cause[key] = totals_by_cause.get(key, 0) + v
items = sorted(totals_by_cause.items(), key=lambda kv: -kv[1])
labels = [k for k, _ in items]; vals = [v for _, v in items]
ax.barh(range(len(labels)), vals, color="#b8313a", alpha=0.85,
        edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis(); ax.set_xscale("log")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
# Pad right for value labels
ax.set_xlim(right=max(vals) * 3)
ax.set_xlabel("Indirect deaths summed across all conflicts (log)")
ax.set_title("(9) Indirect deaths by cause")
for i, v in enumerate(vals):
    ax.text(v, i, f"  {fmt_log_humans(v, None)}",
            va="center", fontsize=8)
ax.grid(True, axis="x", alpha=0.25, which="both")

# ------- (5,2) Indirect-deaths-dominated atrocities through time ---------
ax = fig.add_subplot(gs[4, 1])
ind_heavy = [w for w in WARS
             if w["ind_mid"] > w["civ_dir_mid"] + 1e3 and w["ind_mid"] > 5e4]
ind_heavy.sort(key=lambda w: -w["ind_mid"])
xs = [(w["start"] + w["end"]) / 2 for w in ind_heavy]
ys = [max(w["ind_mid"], 1) for w in ind_heavy]
sz = area_size(ys, ref=1e7, ref_area=2000.0)
cl = [REGION_COLORS[w["region"]] for w in ind_heavy]
ax.scatter(xs, ys, s=sz, c=cl, alpha=0.65,
           edgecolors="black", linewidths=0.4)
texts = []
for w, x, yv in zip(ind_heavy[:18], xs[:18], ys[:18]):
    texts.append(ax.text(x, yv, short(w["name"], 24), fontsize=7,
                         ha="center", va="bottom"))
deconflict(ax, texts)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlim(SINCE - 2, UNTIL + 2)
ax.set_ylim(2e3, 1e9)
ax.set_xlabel("Year"); ax.set_ylabel("Indirect deaths (log)")
ax.set_title("(10) Atrocities where indirect > direct civilian killings")
ax.grid(True, which="both", alpha=0.25)


# ============================ COL 3: MIL vs CIV ratios / diffs ============

# ------- (1,3) Military vs Civilian scatter (log-log) ---------------------
ax = fig.add_subplot(gs[0, 2])
xs = np.array([max(w["mil_mid"],       1) for w in WARS])
ys = np.array([max(w["civ_total_mid"], 1) for w in WARS])
sz = area_size([w["tot_mid"] for w in WARS], ref=1e7, ref_area=1500.0)
cl = [REGION_COLORS[w["region"]] for w in WARS]
ax.scatter(xs, ys, s=sz, c=cl, alpha=0.65,
           edgecolors="black", linewidths=0.4)
lim = [1, 1e8]
ax.plot(lim, lim, "--", color="grey", lw=0.9, label="civ = mil")
ax.plot(lim, [v * 10 for v in lim], ":", color="grey", lw=0.7,
        label="civ = 10× mil")
ax.plot(lim, [v / 10 for v in lim], ":", color="grey", lw=0.7,
        label="civ = mil/10")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_xlabel("Military killed (mid)")
ax.set_ylabel("Civilian deaths total (direct + indirect, mid)")
ax.set_title("(11) Military vs civilian — area∝total")
texts = []
for w in sorted(WARS, key=lambda w: -w["civ_total_mid"])[:5]:
    texts.append(ax.text(max(w["mil_mid"], 1), max(w["civ_total_mid"], 1),
                         short(w["name"], 24), fontsize=7,
                         ha="center", va="bottom"))
for w in sorted([w for w in WARS if w["mil_mid"] > 1e4],
                key=lambda w: w["civ_total_mid"] / max(w["mil_mid"], 1))[:3]:
    texts.append(ax.text(max(w["mil_mid"], 1), max(w["civ_total_mid"], 1),
                         short(w["name"], 24), fontsize=7,
                         ha="center", va="bottom", color="#1f4f8a"))
deconflict(ax, texts)
ax.legend(loc="upper left", fontsize=7)
ax.grid(True, which="both", alpha=0.25)

# ------- (2,3) Civilian:Military ratio, top wars --------------------------
ax = fig.add_subplot(gs[1, 2])
ratio_wars = [w for w in WARS if w["mil_mid"] > 0 and w["civ_total_mid"] > 0]
ratio_wars.sort(key=lambda w: -(w["civ_total_mid"] / w["mil_mid"]))
top_ratio = ratio_wars[:25]
y = np.arange(len(top_ratio))
ratios = np.array([w["civ_total_mid"] / w["mil_mid"] for w in top_ratio])
colors = [REGION_COLORS[w["region"]] for w in top_ratio]
ax.barh(y, ratios, color=colors, alpha=0.7,
        edgecolor="black", linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels([short(w["name"], 36) for w in top_ratio], fontsize=7)
ax.invert_yaxis()
ax.set_xscale("log")
# Pad right edge so the "× 12345" labels fit
ax.set_xlim(right=ratios.max() * 4)
ax.axvline(1, color="grey", ls="--", lw=0.8)
ax.set_xlabel("Civilian deaths / military deaths (log)")
ax.set_title("(12) Civilian : military death ratio — top 25")
for i, r in enumerate(ratios):
    if   r >= 10:   txt = f"  {r:.0f}×"
    elif r >= 1:    txt = f"  {r:.1f}×"
    else:           txt = f"  {r:.2f}×"
    ax.text(r, i, txt, va="center", fontsize=7)
ax.grid(True, axis="x", alpha=0.25, which="both")

# ------- (3,3) Histogram of civilian share --------------------------------
ax = fig.add_subplot(gs[2, 2])
shares = np.array([100 * w["civ_share"] for w in WARS
                   if w["civ_total_mid"] + w["mil_mid"] > 1e3])
bins = np.arange(0, 105, 5)
n, _, patches = ax.hist(shares, bins=bins, edgecolor="black", linewidth=0.4)
cmap = colormaps["RdYlBu_r"]
for p, b in zip(patches, bins[:-1]):
    p.set_facecolor(cmap((b + 2.5) / 100))
ax.axvline(50, color="grey", ls="--", lw=0.7)
ax.set_xlabel("Civilian share of deaths (%)")
ax.set_ylabel("Number of conflicts")
ax.set_title("(13) Distribution of civilian share across conflicts")
ax.grid(True, axis="y", alpha=0.25)
median = float(np.median(shares))
ax.axvline(median, color="black", lw=1.4)
# Annotate median + n cleanly in upper-left, no overlap with bars
ax.text(0.02, 0.98,
        f"n = {len(shares)} conflicts (≥1k deaths)\n"
        f"median civilian share = {median:.0f}%",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.9))

# ------- (4,3) Mil vs civ deaths per decade -------------------------------
ax = fig.add_subplot(gs[3, 2])
mil_by_d = np.zeros(len(decades))
civ_by_d = np.zeros(len(decades))
for w in WARS:
    span = max(w["end"] - w["start"], 1)
    for d_idx, d in enumerate(decades):
        ov = max(0, min(w["end"], d + 10) - max(w["start"], d))
        if ov > 0:
            mil_by_d[d_idx] += w["mil_mid"]       * ov / span
            civ_by_d[d_idx] += w["civ_total_mid"] * ov / span
total_d = mil_by_d + civ_by_d
share_d = np.where(total_d > 0, 100 * civ_by_d / np.maximum(total_d, 1), 0)
bar_w = 8
ax.bar(decades, mil_by_d, width=bar_w, color="#3b6fb6",
       edgecolor="black", linewidth=0.3, label="Military")
ax.bar(decades, civ_by_d, width=bar_w, bottom=mil_by_d, color="#b8313a",
       edgecolor="black", linewidth=0.3, label="Civilian (direct + indirect)")
ax.set_xlabel("Decade"); ax.set_ylabel("Deaths attributed to decade")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_title("(14) Military vs civilian deaths per decade")
# Pad top so the % twin-axis line + labels don't collide with bars
ax.set_ylim(top=total_d.max() * 1.20)
ax.legend(loc="upper left", fontsize=7)
ax.grid(True, axis="y", alpha=0.25)
ax2 = ax.twinx()
ax2.plot(decades, share_d, color="black", marker="o", lw=1.6, zorder=5)
ax2.set_ylabel("Civilian share (%)")
ax2.set_ylim(0, 110)
ax2.axhline(50, color="grey", ls="--", lw=0.6)
# Stagger the % labels above the line; if total deaths in that decade are
# tiny, drop the label to avoid clutter.
for x, s, t in zip(decades, share_d, total_d):
    if t < total_d.max() * 0.005:
        continue
    ax2.annotate(f"{s:.0f}%", (x, s), fontsize=7, ha="center", va="bottom",
                 xytext=(0, 6), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white",
                           ec="none", alpha=0.85))
ax2.legend([Line2D([0], [0], color="black", marker="o", lw=1.4)],
           ["civilian share"], loc="upper right", fontsize=7)

# ------- (5,3) Civilian-excess (civ - mil), top wars ---------------------
ax = fig.add_subplot(gs[4, 2])
diff_wars = sorted(WARS, key=lambda w: -w["civ_minus_mil"])[:15] + \
            sorted(WARS, key=lambda w:  w["civ_minus_mil"])[:5][::-1]
y = np.arange(len(diff_wars))
diffs = np.array([w["civ_minus_mil"] for w in diff_wars])
colors = ["#b8313a" if d >= 0 else "#3b6fb6" for d in diffs]
ax.barh(y, diffs, color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.4)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([short(w["name"], 36) for w in diff_wars], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Civilian deaths − Military deaths (mid)")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_log_humans))
ax.set_title("(15) Civilian-excess: top civ-heavy + top mil-heavy")
# Pad both edges so value text fits inside the axes
pos_max = float(diffs[diffs > 0].max()) if (diffs > 0).any() else 1
neg_min = float(diffs[diffs < 0].min()) if (diffs < 0).any() else -1
ax.set_xlim(neg_min * 1.45, pos_max * 1.20)
for i, d in enumerate(diffs):
    txt = f"  {fmt_log_humans(abs(d), None)}"
    ax.text(d, i, txt, va="center", fontsize=7,
            ha="left" if d >= 0 else "right")
ax.grid(True, axis="x", alpha=0.25)
# Legend explaining sign
ax.legend(handles=[
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#b8313a",
           markersize=10, label="civilian-heavy (civ > mil)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#3b6fb6",
           markersize=10, label="military-heavy (mil > civ)"),
], loc="lower right", fontsize=7)


# ---------------------------------------------------------------- title
_filter_bits = []
if SINCE > 1900: _filter_bits.append(f"since {SINCE}")
if args.min_deaths > 0:
    _filter_bits.append(f"≥{fmt_log_humans(args.min_deaths, None)} deaths")
_filter_str = (" — " + ", ".join(_filter_bits)) if _filter_bits else ""

fig.suptitle(
    f"Wars and mass atrocities, {SINCE}–{UNTIL}{_filter_str}"
    f"{(' — ' + args.title_suffix) if args.title_suffix else ''}\n"
    f"{len(WARS)} conflicts.   "
    "Col 1: scale & timeline   |   "
    "Col 2: composition & indirect killing   |   "
    "Col 3: military vs civilian.   "
    "Marker AREA ∝ death count throughout.",
    fontsize=14, y=0.985,
)

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
out_png = FIGURES / f"{args.out}.png"
out_pdf = FIGURES / f"{args.out}.pdf"
fig.savefig(out_png, dpi=110)
fig.savefig(out_pdf)
print(f"Wrote {out_png}\nWrote {out_pdf}")
