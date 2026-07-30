"""Build publication figures for the paper.

The figures are intentionally restrained: vector PDF output, embedded labels,
little colour, fixed label placement, and no hard-coded "illustrative" claims
except for the Gaza IDF claim band that is stored in the underlying data file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
WARS_DIR = ROOT.parent / "data" / "per_war"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

sns.set_theme(
    context="paper",
    style="ticks",
    font="DejaVu Serif",
    rc={
        "font.size": 9.3,
        "axes.titlesize": 10,
        "axes.labelsize": 9.2,
        "axes.linewidth": 0.55,
        "axes.edgecolor": "#444444",
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
    },
)
PALETTE = sns.color_palette("colorblind", 8)
INK = "#2f2f2f"
FAINT = "#c7c7c7"
BLUE = PALETTE[0]
RUST = PALETTE[3]
SAND = PALETTE[6]


@dataclass
class War:
    ident: str
    name: str
    start: int | None
    end: int
    ongoing: bool
    regions: str
    mil_lo: float
    mil_hi: float
    civ_lo: float
    civ_hi: float
    ind_lo: float
    ind_hi: float
    tot_lo: float
    tot_hi: float
    # True when some side leaves the class unquantified and no curated total
    # covers the gap; the class sum is then only a lower bound.
    mil_incomplete: bool = False
    civ_incomplete: bool = False

    @property
    def total_mid(self) -> float:
        return (self.tot_lo + self.tot_hi) / 2

    @property
    def mil_mid(self) -> float:
        return (self.mil_lo + self.mil_hi) / 2

    @property
    def civ_mid(self) -> float:
        return (self.civ_lo + self.civ_hi) / 2

    @property
    def civ_share_mid(self) -> float:
        return self.civ_mid / max(self.civ_mid + self.mil_mid, 1)

    @property
    def civ_share_lo(self) -> float:
        return self.civ_lo / max(self.civ_lo + self.mil_hi, 1)

    @property
    def civ_share_hi(self) -> float:
        return self.civ_hi / max(self.civ_hi + self.mil_lo, 1)

    @property
    def id_share_lo(self) -> float:
        """Lower end of the identified civilian-share interval.

        An incompletely attributed military class means the true share could
        be anywhere down to 0: the recorded interval is only an upper bound.
        """
        return 0.0 if self.mil_incomplete else self.civ_share_lo

    @property
    def id_share_hi(self) -> float:
        """Upper end of the identified civilian-share interval (see id_share_lo)."""
        return 1.0 if self.civ_incomplete else self.civ_share_hi

    @property
    def indirect_share(self) -> float:
        # Indirect deaths are a subset of civilian deaths (civ = direct + indirect),
        # so this share cannot exceed 1. When a curated civilian total in the
        # totals block undercounts the summed indirect component, the raw ratio
        # can spuriously exceed 100%; clamp it to keep the displayed share sane.
        ratio = (self.ind_lo + self.ind_hi) / max(self.civ_lo + self.civ_hi, 1)
        return min(1.0, ratio)

    @property
    def uncertainty_width(self) -> float:
        return self.civ_share_hi - self.civ_share_lo


def _number_range(d: dict | None) -> tuple[float, float]:
    """Return a usable low/high pair from the heterogeneous source JSON."""
    if not d:
        return 0.0, 0.0
    low = d.get("low")
    high = d.get("high")
    point = d.get("point")
    if low is None:
        low = point if point is not None else high
    if high is None:
        high = point if point is not None else low
    if low is None and high is None:
        return 0.0, 0.0
    low = float(low or 0)
    high = float(high or low)
    if high < low:
        high = low
    return low, high


def _sum_ranges(items: list[dict]) -> tuple[float, float]:
    lows, highs = zip(*(_number_range(x) for x in items), strict=False) if items else ((), ())
    return float(sum(lows)), float(sum(highs))


def load_war(path: Path) -> War | None:
    raw = json.loads(path.read_text())
    if raw.get("_parse_error"):
        return None
    sides = raw.get("sides") or []
    mil_lo, mil_hi = _sum_ranges([s.get("military_killed") or {} for s in sides])
    civ_dir_lo, civ_dir_hi = _sum_ranges([s.get("civilians_killed_directly") or {} for s in sides])
    indirect = [item for s in sides for item in (s.get("deaths_from_actions") or [])]
    ind_lo, ind_hi = _sum_ranges(indirect)
    civ_lo, civ_hi = civ_dir_lo + ind_lo, civ_dir_hi + ind_hi

    def _has_num(d: dict | None) -> bool:
        return bool(d) and any(d.get(k) is not None for k in ("low", "high", "point"))

    def _open_above(d: dict | None) -> bool:
        """Entry gives no usable ceiling: fully null, or a floor with no
        high/point (sources state a minimum and leave the rest unquantified)."""
        if not _has_num(d):
            return True
        return d.get("high") is None and d.get("point") is None

    # A class is incompletely attributed when its ceiling is unstated for some
    # side -- e.g. the Six-Day War records Israeli civilian deaths but leaves
    # Arab civilian deaths null, and North Yemen gives only a royalist
    # military floor.  A curated *_high in the totals block cures this;
    # a floor-only totals entry does not.
    mil_open = any(_open_above(s.get("military_killed")) for s in sides)
    civ_open = any(
        _open_above(s.get("civilians_killed_directly"))
        and not any(_has_num(a) and not _open_above(a) for a in (s.get("deaths_from_actions") or []))
        for s in sides
    )

    totals = raw.get("totals") or {}
    # Prefer the curated military/civilian split in totals when present:
    # side-level fields are often null for wars where sources do not
    # attribute deaths per side, which would otherwise zero out one class.
    t_ml, t_mh = totals.get("military_low"), totals.get("military_high")
    if t_ml is not None or t_mh is not None:
        mil_lo = float(t_ml if t_ml is not None else t_mh)
        # A floor-only totals entry keeps the side-derived ceiling if larger.
        mil_hi = float(t_mh) if t_mh is not None else max(mil_hi, mil_lo)
    t_cl, t_ch = totals.get("civilian_low"), totals.get("civilian_high")
    if t_cl is not None or t_ch is not None:
        civ_lo = float(t_cl if t_cl is not None else t_ch)
        civ_hi = float(t_ch) if t_ch is not None else max(civ_hi, civ_lo)
    mil_no_ceiling = t_mh is None and (mil_open or t_ml is not None)
    civ_no_ceiling = t_ch is None and (civ_open or t_cl is not None)
    tot_lo = totals.get("grand_low")
    tot_hi = totals.get("grand_high")

    # Residual closure: a missing class ceiling is closed mechanically by the
    # grand-total residual (grand high minus the other class's floor).  This
    # keeps the interval two-sided without inventing figures.  It requires a
    # grand ceiling and a real ceiling on the other class; otherwise the
    # closure would be vacuous and the war stays flagged incomplete.
    grand_hi = float(tot_hi) if tot_hi is not None else None
    mil_incomplete = civ_incomplete = False
    if mil_no_ceiling:
        if grand_hi is not None and not civ_no_ceiling:
            mil_hi = max(mil_lo, grand_hi - civ_lo)
        else:
            mil_incomplete = True
    if civ_no_ceiling:
        if grand_hi is not None and not mil_no_ceiling:
            civ_hi = max(civ_lo, grand_hi - mil_lo)
        else:
            civ_incomplete = True
    if tot_lo is None:
        tot_lo = mil_lo + civ_lo
    if tot_hi is None:
        tot_hi = mil_hi + civ_hi
    if tot_hi < tot_lo:
        tot_hi = tot_lo

    regions = raw.get("regions") or raw.get("region") or ""
    if isinstance(regions, list):
        regions = ", ".join(regions[:2])

    return War(
        ident=raw.get("war_id", path.stem),
        name=raw.get("name", path.stem),
        start=raw.get("start_year"),
        end=raw.get("end_year") or 2026,
        ongoing=bool(raw.get("ongoing")),
        regions=regions,
        mil_lo=mil_lo,
        mil_hi=mil_hi,
        civ_lo=civ_lo,
        civ_hi=civ_hi,
        ind_lo=ind_lo,
        ind_hi=ind_hi,
        tot_lo=float(tot_lo or 0),
        tot_hi=float(tot_hi or 0),
        mil_incomplete=mil_incomplete,
        civ_incomplete=civ_incomplete,
    )


def load_wars() -> list[War]:
    wars: list[War] = []
    for path in sorted(WARS_DIR.glob("*.json")):
        if path.stem.endswith("_backup") or path.stem.startswith("_"):
            continue
        war = load_war(path)
        if war and war.total_mid > 0:
            wars.append(war)
    return wars


WARS = load_wars()
print(f"loaded {len(WARS)} wars")


def wars_frame(wars: list[War]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [w.ident for w in wars],
            "name": [w.name for w in wars],
            "short": [short_name(w.name, 36) for w in wars],
            "start": [w.start for w in wars],
            "end": [w.end for w in wars],
            "total_mid": [w.total_mid for w in wars],
            "mil_mid": [w.mil_mid for w in wars],
            "civ_mid": [w.civ_mid for w in wars],
            "civ_share_mid": [100 * w.civ_share_mid for w in wars],
            "civ_share_lo": [100 * w.civ_share_lo for w in wars],
            "civ_share_hi": [100 * w.civ_share_hi for w in wars],
            "uncertainty_width": [100 * w.uncertainty_width for w in wars],
            "indirect_share": [100 * w.indirect_share for w in wars],
            "mil_incomplete": [w.mil_incomplete for w in wars],
            "civ_incomplete": [w.civ_incomplete for w in wars],
            "id_share_lo": [100 * w.id_share_lo for w in wars],
            "id_share_hi": [100 * w.id_share_hi for w in wars],
        }
    )


def fmt_humans(x: float, _pos=None) -> str:
    """Human-readable counts capped at 3 significant figures (93.3k, 14.1M)."""
    if not np.isfinite(x) or x <= 0:
        return ""
    if x >= 1e9:
        return f"{x/1e9:.3g}B"
    if x >= 1e6:
        return f"{x/1e6:.3g}M"
    if x >= 1e3:
        return f"{x/1e3:.3g}k"
    return f"{int(x)}"


def short_name(name: str, max_chars: int = 38) -> str:
    name = name.split("(")[0].replace(" / ", "/").strip()
    return name if len(name) <= max_chars else name[: max_chars - 1] + "…"


def area_size(values: np.ndarray, floor: float = 10, ceil: float = 720) -> np.ndarray:
    """Marker area increasing in deaths: min-max affine scaling with a floor."""
    values = np.maximum(values, 1)
    scaled = (values - values.min()) / max(values.max() - values.min(), 1)
    return floor + scaled * (ceil - floor)


def draw_clean_axis(ax, x_bounds: tuple[float, float] | None = None, y_bounds: tuple[float, float] | None = None):
    sns.despine(ax=ax, trim=True, offset=4)
    if x_bounds:
        ax.spines["bottom"].set_bounds(*x_bounds)
    if y_bounds:
        ax.spines["left"].set_bounds(*y_bounds)
    ax.tick_params(axis="both", length=2.4, pad=2)


# ---------------------------------------------------------------------------
# 1. Data-backed uncertainty ladder: replaces unsupported claim chart.
# ---------------------------------------------------------------------------


def fig_uncertainty_ladder():
    interesting_ids = {
        "israel_gaza_war_2023",
        "russia_ukraine_war_2022",
        "syrian_civil_war",
        "yemeni_civil_war",
        "tigray_war",
        "iraq_war_2003",
        "war_in_afghanistan_2001",
        "sudan_war_2023",
        "second_congo_war",
        "vietnam_war",
        "rwandan_genocide",
        "the_holocaust",
        "partition_of_india",
        "korean_war",
    }
    # Wars where neither casualty class is attributable identify nothing on
    # this axis and are excluded; one-sided wars show the widened interval.
    selected = [
        w for w in WARS
        if w.ident in interesting_ids and not (w.mil_incomplete and w.civ_incomplete)
    ]
    selected.sort(key=lambda w: ((w.id_share_lo + w.id_share_hi) / 2, w.total_mid))
    df = wars_frame(selected)
    df["y"] = np.arange(len(df))
    df["onesided"] = df["mil_incomplete"] | df["civ_incomplete"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hlines(df["y"], df["id_share_lo"], df["id_share_hi"], color=FAINT, lw=4, zorder=2)
    ax.hlines(df["y"], df["id_share_lo"], df["id_share_hi"], color=INK, lw=0.6, zorder=3)
    df_pt = df[~df["onesided"]]
    ax.scatter(df_pt["civ_share_mid"], df_pt["y"], s=32, color=BLUE,
               edgecolor="white", linewidth=0.4, zorder=4)
    for row in df[df["onesided"]].itertuples():
        # Open triangle at the identified bound, pointing into the interval.
        if row.mil_incomplete:
            x, marker = row.civ_share_hi, "<"
        else:
            x, marker = row.civ_share_lo, ">"
        ax.scatter([x], [row.y], s=36, marker=marker, facecolors="none",
                   edgecolors=BLUE, linewidths=0.9, zorder=4)
    for row in df.itertuples():
        ax.text(-2.0, row.y, row.short, ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(103, row.y, fmt_humans(row.total_mid), ha="left", va="center",
                fontsize=8.0, color=INK)

    ax.axvline(50, color=FAINT, lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-1, 115)
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    ax.set_xlabel("civilian share of deaths (identified interval and midpoint)")
    ax.set_title("Civilian-share uncertainty ladder", loc="left", fontsize=10.5, pad=6)
    ax.text(103, len(selected) - 0.35, "total deaths", ha="left", va="bottom",
            fontsize=8.0, color=INK, fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="x", length=2.4, pad=2)
    ax.tick_params(axis="y", left=False)
    out = FIGS / "fig_uncertainty_ladder.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_uncertainty_ladder()


# ---------------------------------------------------------------------------
# 2. All-war civilian share vs scale with deterministic label placement.
# ---------------------------------------------------------------------------


def fig_range_frame_civshare():
    wars = [w for w in WARS if w.total_mid >= 1_000]
    df = wars_frame(wars)
    df["deaths_for_plot"] = df["total_mid"]
    df["marker_area"] = area_size(df["total_mid"].to_numpy(), floor=8, ceil=650)
    df["start_year"] = df["start"].fillna(1900).astype(int)

    # Wars with an incompletely attributed class only identify a one-sided
    # bound on the civilian share: plot them at that bound with an open
    # triangle pointing into the feasible region. Wars where neither class
    # is attributable are dropped from this share-based view.
    df = df[~(df["mil_incomplete"] & df["civ_incomplete"])].copy()
    df["plot_y"] = np.where(
        df["mil_incomplete"], df["civ_share_hi"],
        np.where(df["civ_incomplete"], df["civ_share_lo"], df["civ_share_mid"]),
    )

    # Diverging colour scale over onset year: blue = early 20th century,
    # neutral = 1990 (end of the Cold War), red = recent conflicts.
    year_norm = mcolors.TwoSlopeNorm(vmin=1900, vcenter=1990, vmax=2026)
    year_cmap = plt.get_cmap("coolwarm")
    point_colors = year_cmap(year_norm(df["start_year"].to_numpy()))

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    
    # Plot all except Gaza as points; one-sided wars get open triangles.
    df_no_gaza = df[df["id"] != "israel_gaza_war_2023"]
    complete = df_no_gaza[~(df_no_gaza["mil_incomplete"] | df_no_gaza["civ_incomplete"])]
    onesided = df_no_gaza[df_no_gaza["mil_incomplete"] | df_no_gaza["civ_incomplete"]]
    colors_complete = year_cmap(year_norm(complete["start_year"].to_numpy()))

    ax.scatter(
        complete["deaths_for_plot"], complete["plot_y"],
        s=complete["marker_area"], c=colors_complete, alpha=0.45,
        edgecolor="white", linewidth=0.45, zorder=2,
    )
    ax.scatter(
        complete["deaths_for_plot"], complete["plot_y"],
        s=5, c=colors_complete, alpha=0.95, zorder=3,
    )
    for row in onesided.itertuples():
        marker = "v" if row.mil_incomplete else "^"
        color = year_cmap(year_norm(row.start_year))
        ax.scatter(
            [row.deaths_for_plot], [row.plot_y],
            s=max(row.marker_area * 0.55, 16), marker=marker,
            facecolors="none", edgecolors=[color], linewidths=0.9,
            alpha=0.85, zorder=3,
        )
    sm = plt.cm.ScalarMappable(norm=year_norm, cmap=year_cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.035, pad=0.015, ticks=[1900, 1945, 1990, 2026])
    cbar.set_label("conflict onset year", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.0, length=2.0)
    cbar.outline.set_visible(False)
    ax.axhline(50, color=FAINT, lw=0.6, ls=":", zorder=1)
    ax.text(1.05e3, 51.5, "50% civilian share", fontsize=7.2, color=FAINT, va="bottom")

    gaza_row = df[df["id"] == "israel_gaza_war_2023"]
    if not gaza_row.empty:
        x_gaza = gaza_row["deaths_for_plot"].iloc[0]
        c_gaza = year_cmap(year_norm(gaza_row["start_year"].iloc[0]))
        # Gaza civilian-share bounds derived from the validation report
        # (civ share = 100*(1 - q) at the identified-set upper endpoints).
        headline = json.loads(
            (ROOT.parent / "analysis" / "validation_report.json").read_text())["headline"]
        civ_lo_agnostic = 100 * (1 - headline["q1_moh"])
        civ_lo_calibrated = 100 * (1 - headline["q2_moh"])
        ax.plot([x_gaza, x_gaza], [civ_lo_agnostic, 100], color=c_gaza, lw=0.5, alpha=0.5, zorder=2)
        ax.plot([x_gaza, x_gaza], [civ_lo_calibrated, 100], color=c_gaza, lw=3.5, alpha=0.9, zorder=3)

    # (xmult, yoff, ha) tuned so labels don't collide and each leader line
    # unambiguously reaches its marker; label format: (text, xmult, yoff, ha)
    label_specs = {
        "wwii_european_theater":     ("WWII (Europe/Africa)", 0.13, -14, "center"),
        "wwii_pacific_theater":      ("WWII (Pacific)",       0.40,  -2, "center"),
        "great_leap_forward_famine": ("Great Leap Forward",   0.60,   4.5, "center"),
        "the_holocaust":             ("Holocaust",            1.30,  -3, "left"),
        "second_congo_war":          ("Second Congo",         1.00,  -9, "center"),
        "rwandan_genocide":          ("Rwandan genocide",     0.85,   5.5, "right"),
        "vietnam_war":               ("Vietnam",              0.40,   6, "right"),
        "korean_war":                ("Korean",               1.35,  -7, "left"),
        "syrian_civil_war":          ("Syrian civil war",     0.45,  10, "center"),
        "russia_ukraine_war_2022":   ("Russia\u2013Ukraine",      0.45,   9, "center"),
        "israel_gaza_war_2023":      ("Israel\u2013Gaza 2023",    0.30,   5.5, "right"),
        "wwi":                       ("WWI",                  1.30,  -8, "left"),
        "iraq_war_2003":             ("Iraq 2003",            0.22,  -8, "right"),
        "war_in_afghanistan_2001":   ("Afghanistan 2001",     0.40, -14, "center"),
    }
    by_id = {row.id: row for row in df.itertuples()}
    for ident, (text, xmult, yoff, ha) in label_specs.items():
        if ident not in by_id:
            continue
        w = by_id[ident]
        x, y = w.deaths_for_plot, w.plot_y
        txt = ax.annotate(
            text, (x, y),
            xytext=(x * xmult, y + yoff),
            textcoords="data", ha=ha,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#8a8a8a",
                        "shrinkA": 1.5, "shrinkB": 1.0},
            fontsize=7.6, color=INK, zorder=5,
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_humans))
    ax.set_xlim(1e3, df["deaths_for_plot"].max() * 2.0)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("total deaths (log scale)")
    ax.set_ylabel("civilian share of deaths")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    ax.set_title("Civilian share vs. total deaths, 81 conflicts (1899–2026)",
                 loc="left", fontsize=10.5, pad=6)
    draw_clean_axis(ax, (1e3, df["deaths_for_plot"].max() * 2.0), (0, 100))
    out = FIGS / "fig_range_frame_civshare.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_range_frame_civshare()


# ---------------------------------------------------------------------------
# 3. Top-conflict timeline: same scale, clear labels, duration visible.
# ---------------------------------------------------------------------------


def fig_sparkline_timeline():
    big = sorted(WARS, key=lambda w: -w.total_mid)[:12]
    fig, axes = plt.subplots(len(big), 1, figsize=(7.2, 0.50 * len(big)), sharex=True)
    for ax, w in zip(axes, big, strict=True):
        start = max(w.start or 1900, 1900)
        end = min(w.end or 2026, 2026)
        total = w.total_mid
        timeline = pd.DataFrame({"year": [start, end], "cumulative_deaths": [0, total]})
        sns.lineplot(
            data=timeline,
            x="year",
            y="cumulative_deaths",
            ax=ax,
            color=RUST,
            lw=0.9,
            legend=False,
        )
        ax.fill_between([start, end], [0, total], color=RUST, alpha=0.13, lw=0)
        sns.scatterplot(x=[end], y=[total], ax=ax, s=12, color=RUST, legend=False, zorder=3)
        ax.text(1898, total * 0.5, short_name(w.name, 37), ha="right", va="center", fontsize=8.2)
        ax.text(end + 1.0, total, fmt_humans(total), ha="left", va="center", fontsize=8.0, color=INK)
        ax.set_xlim(1899, 2031)
        ax.set_ylim(0, max(total * 1.06, 1))
        ax.set_axis_off()

    axes[-1].set_axis_on()
    for spine in ("top", "right", "left"):
        axes[-1].spines[spine].set_visible(False)
    axes[-1].set_yticks([])
    axes[-1].set_xticks([1900, 1925, 1950, 1975, 2000, 2025])
    axes[-1].tick_params(axis="x", length=3, labelsize=8.0)
    fig.suptitle("Cumulative deaths over duration (top-12 conflicts by midpoint total)", fontsize=10, y=1.01)
    out = FIGS / "fig_sparkline_timeline.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_sparkline_timeline()


# ---------------------------------------------------------------------------
# 4. Sensitivity curves using real Gaza demographic anchors, not hypotheticals.
# ---------------------------------------------------------------------------


def q_bound(mu: np.ndarray | float, omega: float, w: float = 0.733, p_am: float = 0.267, f: float = 0.02):
    return 1.0 - omega * (w + mu * (p_am - f)) / w


def omega_required(q: np.ndarray | float, mu: float = 2.5, w: float = 0.733, p_am: float = 0.267, f: float = 0.02):
    return (1.0 - q) * w / (w + mu * (p_am - f))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probs=(0.025, 0.5, 0.975)) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order] / weights.sum()
    cdf = np.cumsum(weights)
    return np.interp(probs, cdf, values)


def fig_gaza_diagnostic():
    posterior = ROOT.parent / "gaza_sim" / "posterior.npz"
    if not posterior.exists():
        print("  (skipping fig_gaza_diagnostic: posterior.npz not found)")
        return
    p = np.load(posterior)
    weights = p["weights"] / p["weights"].sum()

    q_post = p["q"] * 100
    q_lo, q_med, q_hi = weighted_quantile(q_post, weights)

    D_obs = p["D_obs"]
    D_milt = p["D_milt"]
    civ_milt = p["civ_milt_ratio"]
    cm_lo, cm_med, cm_hi = weighted_quantile(civ_milt, weights)
    dmilt_lo, dmilt_med, dmilt_hi = weighted_quantile(D_milt, weights)

    # IDF claim converted to q over the D scenarios used in the paper:
    # 70k (MoH confirmed) .. 119.2k (GMS-corrected + missing).  The
    # posterior q is a share of TRUE direct deaths, so the claim band
    # spans the denominator scenarios: the envelope [17k/119.2k,
    # 25k/70k] gives the claim its most favourable conversion at the
    # low end.
    IDF_LO_K, IDF_HI_K = 17_000, 25_000
    D_LO, D_HI = 70_000, 119_200
    idf_q_lo = IDF_LO_K / D_HI * 100
    idf_q_hi = IDF_HI_K / D_LO * 100

    # Diagnostic constants (AM = males 18+, so a = 1 - w)
    # Adult-male share is 1 - w by the paper's class partition, so it
    # tracks w wherever w varies.
    F_SHARE = 0.020    # combatant pop share
    MU_BAR  = 2.5

    fig = plt.figure(figsize=(7.6, 6.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.78, 0.78],
                          hspace=0.70, wspace=0.30)

    # ---------------- Panel A: 2D HEATMAP (the killer panel) ----------------
    ax = fig.add_subplot(gs[0, :])
    w_grid = np.linspace(0.66, 0.80, 160)
    omega_grid = np.linspace(0.30, 0.78, 200)
    W, O = np.meshgrid(w_grid, omega_grid, indexing="ij")
    # a = 1 - w exactly (the paper's class partition, Table 1), so the
    # adult-male share varies with w across the grid.
    Q = 1.0 - O * (W + MU_BAR * ((1.0 - W) - F_SHARE)) / W
    Q = np.clip(Q, 0, 1) * 100

    im = ax.pcolormesh(omega_grid * 100, w_grid * 100, Q,
                       cmap="RdYlBu_r", vmin=0, vmax=40, shading="auto",
                       rasterized=True)
    levels = [5, 10, 15, 20, 25, 30, 35]
    cs = ax.contour(omega_grid * 100, w_grid * 100, Q,
                    levels=levels,
                    colors="black", linewidths=0.45, alpha=0.55)
    # Compute ω at top of axis (w = 78.5%) for each level so labels go on
    # the actual contour line, in clear sky.
    w_top = 0.785
    coef_top = (w_top + MU_BAR * ((1.0 - w_top) - F_SHARE)) / w_top
    label_pts = [(((1 - q/100) / coef_top) * 100, 78.3) for q in levels]
    ax.clabel(cs, inline=True, fontsize=6.3, fmt="%d%%",
              manual=label_pts, inline_spacing=1)

    # IDF-required omega band: at w=73.5%, the required omega for q in claim band.
    w_ref = 0.735
    coef = (w_ref + MU_BAR * ((1.0 - w_ref) - F_SHARE)) / w_ref
    omega_req_lo = (1 - idf_q_hi/100) / coef * 100
    omega_req_hi = (1 - idf_q_lo/100) / coef * 100
    ax.axvspan(omega_req_lo, omega_req_hi, color=RUST, alpha=0.20, zorder=2)
    ax.text((omega_req_lo + omega_req_hi)/2, 73,
            f"$\\omega$ required\nby IDF claim",
            ha="center", va="center", fontsize=7.6, color=RUST,
            fontweight="bold", rotation=0,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # Anchor markers (place along w=73.5% horizontal line)
    anchors = [
        ("MoH",   56.0, 73.5, "#5a5a5a", (-32, 16)),
        ("blend", 62.3, 73.5, BLUE,      (8, 16)),
        ("OHCHR", 69.3, 73.5, "#222222", (10, -22)),
    ]
    for label, omega_pct, w_pct, color, (dx, dy) in anchors:
        ax.scatter([omega_pct], [w_pct], s=80, color="white",
                   edgecolor=color, linewidth=1.6, zorder=6)
        ax.scatter([omega_pct], [w_pct], s=14, color=color, zorder=7)
        ax.annotate(f"{label}\n$\\omega$={omega_pct:.1f}%",
                    (omega_pct, w_pct),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.6, ha="left", color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=color, lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    cbar = fig.colorbar(im, ax=ax, pad=0.012, aspect=22, shrink=0.92)
    cbar.set_label("implied combatant share $q$ (%)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    ax.set_xlim(30, 78)
    ax.set_ylim(67, 79)
    ax.set_xlabel(r"observed women+children share among dead, $\omega$ (%)")
    ax.set_ylabel(r"population share $w$ (%)", labelpad=1)
    ax.set_title(
        r"A. Identified $q$ on the $(\omega,w)$ plane at $\bar\mu=2.5$",
        loc="left", fontsize=9.5, pad=4, fontweight="bold")
    ax.set_aspect("auto")

    # ---------------- Panel B: posterior on q histogram ----------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 45, 70)
    ax.hist(q_post, bins=bins, weights=weights, density=True,
            color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.25)
    ax.axvspan(idf_q_lo, idf_q_hi, color=RUST, alpha=0.20, zorder=1)
    for x, ls, lw in [(q_med, "-", 1.0), (q_lo, "--", 0.8), (q_hi, "--", 0.8)]:
        ax.axvline(x, color=INK, ls=ls, lw=lw, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"posterior:\n{q_med:.1f}% [{q_lo:.1f}, {q_hi:.1f}]",
                xy=(q_med, 0.55 * ymax),
                xytext=(5.5, 0.85 * ymax),
                fontsize=7.6, color=BLUE, fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    ax.text((idf_q_lo + idf_q_hi)/2, 0.55 * ymax,
            f"IDF claim\n{idf_q_lo:.0f}–{idf_q_hi:.0f}%\n"
            r"($\rho_\omega\approx$8–31$\,\sigma_\omega$)",
            ha="center", va="center", fontsize=7.6, color=RUST,
            fontweight="bold")
    ax.set_xlim(0, 45)
    ax.set_xlabel(r"combatant share $q$ (%)")
    ax.set_ylabel("posterior density")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    ax.set_title("B. Posterior on $q$ vs. IDF claim band",
                 loc="left", fontsize=9.5, pad=4, fontweight="bold")
    draw_clean_axis(ax, (0, 45))

    # ---------------- Panel C: implied combatants (symlog) ----------------
    ax = fig.add_subplot(gs[1, 1])
    bins_c = np.logspace(np.log10(300), np.log10(40_000), 70)
    ax.hist(D_milt, bins=bins_c, weights=weights, density=True,
            color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.25)
    ax.axvspan(IDF_LO_K, IDF_HI_K, color=RUST, alpha=0.20, zorder=1)
    for x, ls, lw in [(dmilt_med, "-", 1.0),
                      (dmilt_lo,  "--", 0.8),
                      (dmilt_hi,  "--", 0.8)]:
        ax.axvline(x, color=INK, ls=ls, lw=lw, zorder=4)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)
    ymax = ax.get_ylim()[1]
    ax.text(dmilt_med, 0.96 * ymax,
            f"posterior\n{int(dmilt_med):,}\n[{int(dmilt_lo):,}–{int(dmilt_hi):,}]",
            ha="center", va="top", fontsize=7.4, color=BLUE,
            fontweight="bold")
    ax.text(IDF_LO_K * 0.93, 0.55 * ymax,
            f"IDF claim\n{IDF_LO_K//1000}–{IDF_HI_K//1000}k",
            ha="right", va="center", fontsize=7.6, color=RUST,
            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(300, 40_000)
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    ax.set_xlabel("implied combatants killed")
    ax.set_ylabel("posterior density")
    ax.set_title("C. Implied combatant count (log scale)",
                 loc="left", fontsize=9.5, pad=4, fontweight="bold")
    draw_clean_axis(ax)

    # ---------------- Panel D: stacked decomposition of dead ----------------
    ax = fig.add_subplot(gs[2, 0])
    D_civAM = p["D_civAM"]
    D_WC = p["D_WC"]
    cats = ["combatants", "civ. adult\nmales", "women +\nchildren"]
    medians = []
    los = []
    his = []
    for arr in [D_milt, D_civAM, D_WC]:
        lo, med, hi = weighted_quantile(arr, weights)
        medians.append(med); los.append(lo); his.append(hi)
    medians = np.array(medians); los = np.array(los); his = np.array(his)
    colors = [RUST, "#d4a017", BLUE]
    xs = np.arange(len(cats))
    ax.bar(xs, medians, color=colors, alpha=0.82,
           edgecolor=INK, linewidth=0.5, width=0.62)
    ax.errorbar(xs, medians, yerr=[medians - los, his - medians],
                fmt="none", color=INK, lw=1.0, capsize=3, capthick=0.8)
    for x, m, hi in zip(xs, medians, his):
        # Labels go above the upper CI whisker, one decimal below 10k so
        # small medians (e.g. 1,775) are not truncated to "1k".
        lab = (f"{m/1000:.1f}k" if 1000 <= m < 10_000
               else f"{int(round(m/1000))}k" if m >= 1000 else f"{int(m)}")
        ax.text(x, hi + 0.03 * his.max(), lab,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=8.0)
    ax.set_ylabel("posterior median deaths")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    ax.set_ylim(0, his.max() * 1.20)
    ax.set_title("D. Decomposition of true direct deaths",
                 loc="left", fontsize=9.5, pad=4, fontweight="bold")
    sns.despine(ax=ax, trim=False, offset=3)
    ax.tick_params(axis="x", length=0)

    # ---------------- Panel E: civ:mil ratio histogram ----------------
    ax = fig.add_subplot(gs[2, 1])
    bins_r = np.linspace(0, 90, 60)
    ax.hist(civ_milt, bins=bins_r, weights=weights, density=True,
            color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.25)
    for x, ls, lw in [(cm_med, "-", 1.0),
                      (cm_lo,  "--", 0.8),
                      (cm_hi,  "--", 0.8)]:
        ax.axvline(x, color=INK, ls=ls, lw=lw, zorder=4)

    # IDF-implied ratio band
    idf_ratio_lo = (1 - idf_q_hi/100) / (idf_q_hi/100)
    idf_ratio_hi = (1 - idf_q_lo/100) / (idf_q_lo/100)
    ax.axvspan(idf_ratio_lo, idf_ratio_hi, color=RUST, alpha=0.20, zorder=1)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)
    ymax = ax.get_ylim()[1]
    ax.text(cm_med, 0.96 * ymax,
            f"posterior\n{cm_med:.0f}:1 [{cm_lo:.0f}–{cm_hi:.0f}]",
            ha="center", va="top", fontsize=7.6, color=BLUE,
            fontweight="bold")
    ax.text((idf_ratio_lo + idf_ratio_hi)/2 + 4, 0.55 * ymax,
            f"IDF-implied\n{idf_ratio_lo:.0f}\u2013{idf_ratio_hi:.0f}:1",
            ha="left", va="center", fontsize=7.6, color=RUST,
            fontweight="bold")
    ax.set_xlabel("civilian : combatant ratio")
    ax.set_ylabel("posterior density")
    ax.set_xlim(0, 90)
    ax.set_title("E. Civilian-to-combatant ratio",
                 loc="left", fontsize=9.5, pad=4, fontweight="bold")
    draw_clean_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = FIGS / "fig_gaza_diagnostic.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_gaza_diagnostic()


def fig_q_identified_set():
    anchors = [
        ("OHCHR identified sample", 0.693, "women+children = 69.3%"),
        ("Blended anchor", 0.623, "geometric consensus = 62.3%"),
        ("MoH full record", 0.560, "women+children = 56.0%"),
    ]
    mus = np.linspace(1.0, 3.0, 160)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    for ax, (title, omega, subtitle) in zip(axes, anchors, strict=True):
        q_vals = np.clip(q_bound(mus, omega), 0, 1) * 100
        curve = pd.DataFrame({"mu": mus, "q_upper": q_vals})
        ax.fill_between(mus, 0, q_vals, color=BLUE, alpha=0.10, lw=0)
        sns.lineplot(data=curve, x="mu", y="q_upper", ax=ax, color=BLUE, lw=0.9, legend=False)
        ax.axvline(2.5, color=FAINT, lw=0.65, ls=":")
        q1 = max(0, q_bound(1.0, omega) * 100)
        q25 = max(0, q_bound(2.5, omega) * 100)
        sns.scatterplot(x=[1.0, 2.5], y=[q1, q25], ax=ax, color=RUST, s=18, legend=False, zorder=4)
        ax.text(1.02, min(q1 + 2.2, 49), f"{q1:.1f}%", fontsize=7.0, color=INK)
        ax.text(2.52, min(q25 + 2.2, 49), f"{q25:.1f}%", fontsize=7.0, color=INK)
        ax.set_title(title, loc="left", fontsize=8.8)
        ax.text(0.02, 0.90, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=INK)
        ax.set_xlim(1.0, 3.0)
        ax.set_ylim(0, 50)
        ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel(r"civilian adult-male exposure cap $\bar\mu$")
        draw_clean_axis(ax, (1, 3), (0, 50))
    axes[0].set_ylabel(r"upper bound on combatant share $q$")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
    fig.suptitle(r"Gaza identified set: $q^{up}(\bar\mu)$ under real demographic anchors", fontsize=10, y=1.04)
    out = FIGS / "fig_q_id_set_smallmult.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_q_identified_set()


# ---------------------------------------------------------------------------
# 5. Gaza posterior strips with quantile labels and claim band.
# ---------------------------------------------------------------------------


def fig_gaza_sparklines():
    posterior = ROOT.parent / "gaza_sim" / "posterior.npz"
    if not posterior.exists():
        print("  (skipping fig_gaza_sparklines: posterior.npz not found)")
        return
    p = np.load(posterior)
    weights = p["weights"] / p["weights"].sum()

    panels = [
        ("Combatant share $q$", p["q"] * 100, 0, 12, RUST, "%"),
        ("Combatants killed", p["D_milt"] / 1000, 0, 27, BLUE, "k"),
        ("Civilian:militant ratio", p["civ_milt_ratio"], 0, 110, RUST, "x"),
        ("Adult-male share among dead: model fit", (p["D_civAM"] + p["D_milt"]) / np.maximum(p["D_obs"], 1) * 100, 18, 50, BLUE, "%"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(5.4, 4.1))
    for ax, (label, arr, x0, x1, colour, unit) in zip(axes, panels, strict=True):
        lo, med, hi = weighted_quantile(arr, weights)
        sns.kdeplot(
            x=arr,
            weights=weights,
            ax=ax,
            color=colour,
            fill=True,
            alpha=0.24,
            linewidth=0.85,
            bw_adjust=0.9,
            clip=(x0, x1),
        )
        ymax = max(ax.get_ylim()[1], 1e-9)
        ax.axvline(med, color=INK, lw=0.65)
        ax.plot([lo, hi], [0.86 * ymax, 0.86 * ymax], color=INK, lw=0.65)
        label_text = f"{med:.1f}{unit}  [{lo:.1f}, {hi:.1f}]"
        ax.text(med, 0.93 * ymax, label_text, ha="center", va="bottom", fontsize=7.0, color=INK)
        ax.text(0.0, 0.93, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=INK)
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, 1.05 * ymax)
        ax.set_axis_off()

    axes[1].axvspan(17, 25, color=RUST, alpha=0.14)
    axes[1].text(21, 0.72, "IDF public claim", transform=axes[1].get_xaxis_transform(),
                 color=RUST, fontsize=7.2, ha="center", va="bottom")
    axes[3].axvline(30.7, color=INK, lw=0.5)
    axes[3].text(30.7, 0.58, "OHCHR", transform=axes[3].get_xaxis_transform(),
                 color=INK, fontsize=6.9, ha="center")
    axes[3].axvline(44.0, color=INK, lw=0.5)
    axes[3].text(44.0, 0.58, "MoH", transform=axes[3].get_xaxis_transform(),
                 color=INK, fontsize=6.9, ha="center")

    out = FIGS / "fig_gaza_sparklines.pdf"
    fig.savefig(out)
    print(f"  wrote {out.name}")


fig_gaza_sparklines()


# ---------------------------------------------------------------------------
# 6. The adult-male death budget: three-component decomposition of the toll
#    under six hypotheses.  All numbers come from the validation report so
#    the figure and the prose cannot drift apart.
# ---------------------------------------------------------------------------


def fig_gaza_budget():
    report_path = ROOT.parent / "analysis" / "validation_report.json"
    if not report_path.exists():
        print("  (skipping fig_gaza_budget: validation_report.json not found;"
              " run analysis/validate_bounds.py first)")
        return
    b = json.loads(report_path.read_text())["budget"]
    rows = b["rows"]
    D = b["D"]
    D_MEN = b["D_men"]
    RATE_WC = b["rate_wc"]
    N_POP = b["N_pop"]
    pop_shares = [(b["M_stock"] / N_POP, "combatants"),
                  (b["N_amciv"] / N_POP, "civilian men"),
                  (b["N_wc"] / N_POP, "women + children")]

    GOLD = "#d4a017"          # civilian adult men (matches fig_gaza_diagnostic D)
    C_WC, C_AMCIV, C_COMB = BLUE, GOLD, RUST
    SEG_COLORS = [C_COMB, C_AMCIV, C_WC]

    display = [
        # (row index, label, short label, is_claim, is_reference)
        (0, "random violence, no targeting\n(rejected by the record, $z=-8.1$)",
         "random violence, no targeting", False, True),
        (1, "calibrated exposure, upper end ($\\mu=2$)",
         "calibrated exposure ($\\mu=2$)", False, False),
        (2, "equal exposure ($\\mu=1$),\nthe mathematical cap",
         "equal exposure ($\\mu=1$)", False, False),
        (3, "IDF claim, 17k", "IDF claim, 17k", True, False),
        (4, "IDF claim, 25k", "IDF claim, 25k", True, False),
        (5, "all adult men counted\nas combatants",
         "all adult men combatants", False, False),
    ]
    n = len(display)
    ys = np.arange(n)[::-1]   # first hypothesis row on top
    bar_h = 0.62
    GAP = 1.05                # gap between population row and hypotheses
    y_pop = ys[0] + GAP + 1.0

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.2, 6.6), sharey=False,
        gridspec_kw={"height_ratios": [1.22, 0.88], "hspace": 0.44})

    # ---------------- Panel A: allocation of the 70,000 dead ----------------
    # Reference row: population composition, drawn as shares of the bar width.
    left = 0.0
    for (share, _), color in zip(pop_shares, SEG_COLORS, strict=True):
        ax_a.barh(y_pop, share * D, left=left, height=bar_h, color=color,
                  alpha=0.55, edgecolor="white", linewidth=0.6, zorder=3)
        left += share * D
    ax_a.text(pop_shares[0][0] * D + 900, y_pop + bar_h / 2 + 0.08, "2.0%",
              ha="left", va="bottom", fontsize=7.0, color=C_COMB,
              fontweight="bold")
    ax_a.text((pop_shares[0][0] + pop_shares[1][0] / 2) * D, y_pop, "24.7%",
              ha="center", va="center", fontsize=7.4, color="white",
              fontweight="bold")
    ax_a.text((1 - pop_shares[2][0] / 2) * D, y_pop, "73.3%",
              ha="center", va="center", fontsize=7.4, color="white",
              fontweight="bold")
    ax_a.text(D + 1_500, y_pop, "population\nshares,\n$N=2.25$M", ha="left",
              va="center", fontsize=7.0, color="#6a6a6a")

    for (idx, label, _, is_claim, is_ref), y in zip(display, ys, strict=True):
        r = rows[idx]
        segs = [(r["d_combatant"], C_COMB), (r["d_amciv"], C_AMCIV),
                (r["d_wc"], C_WC)]
        left = 0.0
        alpha = 0.55 if is_ref else 0.88
        for val, color in segs:
            if val <= 0:
                continue
            ax_a.barh(y, val, left=left, height=bar_h, color=color,
                      alpha=alpha, edgecolor="white", linewidth=0.6, zorder=3)
            left += val
        # combatant count at the segment it measures
        dc = r["d_combatant"]
        txt = f"{round(dc/100)/10:g}k"
        if dc > 5_500:
            ax_a.text(dc / 2, y, txt, ha="center", va="center",
                      fontsize=7.4, color="white", fontweight="bold", zorder=4)
        else:
            ax_a.text(dc + 900, y + bar_h / 2 + 0.08, txt, ha="left",
                      va="bottom", fontsize=7.0, color=C_COMB,
                      fontweight="bold", zorder=4)
        # implied exposure ratio at the right margin
        mu = r["mu_implied"]
        note = f"$\\mu={mu:.2f}$" if np.isfinite(mu) else "no civilian\nmen left"
        ax_a.text(D + 1_500, y, note, ha="left", va="center", fontsize=7.4,
                  color=RUST if (is_claim or not np.isfinite(mu)) else INK)

    # the men's budget line: fixed by the demographic record for rows 2-6
    y_span_lo = ys[-1] - bar_h / 2 - 0.12
    y_span_hi = ys[1] + bar_h / 2 + 0.12
    ax_a.plot([D_MEN, D_MEN], [y_span_lo, y_span_hi],
              color=INK, lw=0.9, ls=(0, (4, 2)), zorder=5)
    ax_a.text(D_MEN, ys[0] + (GAP + 1.0) / 2,
              "adult-male dead: 30,800 (fixed by the demographic record)",
              ha="center", va="center", fontsize=7.4, color=INK)

    ax_a.set_yticks(np.concatenate([[y_pop], ys]))
    ax_a.set_yticklabels(["who lives in Gaza\n(population, for reference)"]
                         + [lbl for _, lbl, _, _, _ in display], fontsize=7.8)
    ax_a.set_xlim(0, D * 1.24)
    ax_a.set_ylim(ys[-1] - 0.75, y_pop + 0.85)
    ax_a.set_xticks(np.arange(0, 70_001, 10_000))
    ax_a.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax_a.set_xlabel("deaths (MoH-confirmed toll, $D=70{,}000$)")
    ax_a.set_title("A. Who the 70,000 dead were, under each hypothesis",
                   loc="left", fontsize=9.5, pad=4, fontweight="bold")
    sns.despine(ax=ax_a, trim=False, offset=2)
    ax_a.tick_params(axis="y", length=0)

    # ---------------- Panel B: share of each group killed --------------------
    # Calibrated exposure band for civilian men: mu in [2, 3.5] times the
    # women+children rate.
    cal_lo, cal_hi = 2.0 * RATE_WC * 100, 3.5 * RATE_WC * 100
    ax_b.axvspan(cal_lo, cal_hi, color=GOLD, alpha=0.14, zorder=1)
    ax_b.text(np.sqrt(cal_lo * cal_hi), ys[0] + 0.98,
              "civilian men in past\nGaza conflicts ($\\mu\\in[2,3.5]$)",
              ha="center", va="bottom", fontsize=7.0, color="#8a6a0a")
    ax_b.axvline(RATE_WC * 100, color=C_WC, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax_b.text(RATE_WC * 100 * 0.93, ys[0] + 0.98,
              "women + children:\n2.4% killed, fixed",
              ha="right", va="bottom", fontsize=7.0, color=C_WC)

    for (idx, _, _, is_claim, is_ref), y in zip(display, ys, strict=True):
        r = rows[idx]
        alpha = 0.60 if is_ref else 1.0
        pts = [(r["rate_wc"] * 100, C_WC),
               (r["rate_amciv"] * 100 if np.isfinite(r["rate_amciv"]) else None,
                C_AMCIV),
               (r["rate_combatant"] * 100, C_COMB)]
        xs = [p for p, _ in pts if p is not None and p > 0]
        ax_b.plot([min(xs), max(xs)], [y, y], color=FAINT, lw=0.7, zorder=2)
        for p, color in pts:
            if p is None or p <= 0:
                continue
            ax_b.scatter([p], [y], s=30, color=color, edgecolor="white",
                         linewidth=0.6, zorder=4, alpha=alpha)
        # print the combatant share killed above its dot
        rc = r["rate_combatant"] * 100
        rc_txt = f"{rc:.0f}%" if rc >= 10 else f"{rc:.1f}%"
        ax_b.text(rc, y + 0.30, rc_txt, ha="center", va="bottom",
                  fontsize=6.8, color=C_COMB, fontweight="bold")
        sel = r["selectivity"]
        if is_claim and np.isfinite(sel):
            ax_b.text(rc * 1.30, y, f"{sel:.0f}$\\times$ the civilian-men rate",
                      ha="left", va="center", fontsize=7.2, color=C_COMB)
        if idx == 0:
            ax_b.text(rc * 1.30, y, "every group dies at 3.1%",
                      ha="left", va="center", fontsize=7.2, color="#6a6a6a")

    ax_b.set_xscale("log")
    ax_b.set_xlim(0.7, 300)
    ax_b.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax_b.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f"{x:g}%"))
    ax_b.xaxis.set_minor_locator(mtick.NullLocator())
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([short for _, _, short, _, _ in display], fontsize=7.8)
    ax_b.set_ylim(ys[-1] - 0.75, ys[0] + 2.55)
    ax_b.set_xlabel("share of the group killed (log scale)")
    ax_b.set_title("B. What share of each group was killed, under the same"
                   " hypotheses", loc="left", fontsize=9.5, pad=4,
                   fontweight="bold")
    sns.despine(ax=ax_b, trim=False, offset=2)
    ax_b.tick_params(axis="y", length=0)

    # shared legend below the figure
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_COMB, alpha=0.88, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc=C_AMCIV, alpha=0.88, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc=C_WC, alpha=0.88, ec="none"),
    ]
    fig.legend(handles, ["combatants", "civilian adult men", "women + children"],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               frameon=False, fontsize=7.6, handlelength=1.2,
               handleheight=0.9, columnspacing=1.4)

    fig.tight_layout()
    out = FIGS / "fig_gaza_budget.pdf"
    fig.savefig(out)
    fig.savefig(FIGS / "fig_gaza_budget.png", dpi=300)
    print(f"  wrote {out.name}")


fig_gaza_budget()


# ---------------------------------------------------------------------------
# Appendix table: all conflicts behind Figures 3-4, as a longtable.
# ---------------------------------------------------------------------------


def latex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%")
             .replace("#", r"\#").replace("_", r"\_")
             .replace("\u2026", r"\dots{}"))


def fmt_count(x: float) -> str:
    if x >= 1e6:
        return f"{x/1e6:.3g}M"
    if x >= 1e3:
        return f"{x/1e3:.3g}k"
    return f"{int(round(x))}"


def write_all_conflicts_table():
    wars = sorted(
        [w for w in WARS if w.total_mid >= 1_000],
        key=lambda w: (w.start or 1900, w.name),
    )
    lines = [
        "% Auto-generated by paper/build_figures.py -- do not edit by hand.",
        r"\begin{footnotesize}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{longtable}{@{}lccrc@{}}",
        r"\caption{\textbf{The 81-conflict dataset behind Figures~\ref{fig:range-frame} "
        r"and~\ref{fig:uncertainty-ladder}} (conflicts with at least 1{,}000 attributed "
        r"deaths shown). Deaths are midpoints of the curated total-death ranges "
        r"(military $+$ civilian, including indirect where sources attribute it); the "
        r"civilian-share interval spans the most and least civilian-heavy readings of the "
        r"source ranges. Where sources state no ceiling for one casualty class, the ceiling "
        r"is closed mechanically by the grand-total residual (total high minus the other "
        r"class's floor) rather than treated as zero. Per-conflict sources are in the "
        r"supplement.}"
        r"\label{tab:all-conflicts}\\",
        r"\toprule",
        r"Conflict & Period & Deaths (mid) & Civilian share & lo--hi \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:all-conflicts} continued}}\\",
        r"\toprule",
        r"Conflict & Period & Deaths (mid) & Civilian share & lo--hi \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for w in wars:
        period = f"{w.start or '?'}--{'' if w.ongoing else w.end}"
        if w.mil_incomplete and w.civ_incomplete:
            share, rng = "--", "n.a."
        elif w.mil_incomplete:
            share = "--"
            rng = f"$\\le${100*w.civ_share_hi:.0f}\\%"
        elif w.civ_incomplete:
            share = "--"
            rng = f"$\\ge${100*w.civ_share_lo:.0f}\\%"
        else:
            share = f"{100*w.civ_share_mid:.0f}\\%"
            rng = f"{100*w.civ_share_lo:.0f}--{100*w.civ_share_hi:.0f}\\%"
        deaths = fmt_count(w.total_mid)
        lines.append(
            f"{latex_escape(short_name(w.name, 44))} & {period} & {deaths} & {share} & {rng} \\\\"
        )
    lines += [r"\end{longtable}", r"\end{footnotesize}", ""]
    out = ROOT / "content" / "tab_all_conflicts.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out.name} ({len(wars)} conflicts)")


write_all_conflicts_table()

print(f"\nAll figures in {FIGS}")
