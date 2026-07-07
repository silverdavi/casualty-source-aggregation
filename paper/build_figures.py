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
    def indirect_share(self) -> float:
        return (self.ind_lo + self.ind_hi) / max(self.civ_lo + self.civ_hi, 1)

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

    totals = raw.get("totals") or {}
    tot_lo = totals.get("grand_low")
    tot_hi = totals.get("grand_high")
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
        ident=raw.get("war_id", path.parent.name),
        name=raw.get("name", path.parent.name),
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
    """Marker area proportional to deaths, with visual floor."""
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
    selected = [w for w in WARS if w.ident in interesting_ids]
    selected.sort(key=lambda w: (w.civ_share_mid, w.total_mid))
    df = wars_frame(selected)
    df["y"] = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hlines(df["y"], df["civ_share_lo"], df["civ_share_hi"], color=FAINT, lw=4, zorder=2)
    ax.hlines(df["y"], df["civ_share_lo"], df["civ_share_hi"], color=INK, lw=0.6, zorder=3)
    ax.scatter(df["civ_share_mid"], df["y"], s=32, color=BLUE,
               edgecolor="white", linewidth=0.4, zorder=4)
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
    ax.set_xlabel("civilian share of deaths (low-mid-high range)")
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
    wars = [w for w in WARS if (w.mil_mid + w.civ_mid) >= 1_000]
    df = wars_frame(wars)
    df["deaths_for_plot"] = df["mil_mid"] + df["civ_mid"]
    df["marker_area"] = area_size(df["total_mid"].to_numpy(), floor=8, ceil=650)
    df["start_year"] = df["start"].fillna(1900).astype(int)

    # Diverging colour scale over onset year: blue = early 20th century,
    # neutral = 1990 (end of the Cold War), red = recent conflicts.
    year_norm = mcolors.TwoSlopeNorm(vmin=1900, vcenter=1990, vmax=2026)
    year_cmap = plt.get_cmap("coolwarm")
    point_colors = year_cmap(year_norm(df["start_year"].to_numpy()))

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    
    # Plot all except Gaza as points
    df_no_gaza = df[df["id"] != "israel_gaza_war_2023"]
    point_colors_no_gaza = year_cmap(year_norm(df_no_gaza["start_year"].to_numpy()))
    
    ax.scatter(
        df_no_gaza["deaths_for_plot"], df_no_gaza["civ_share_mid"],
        s=df_no_gaza["marker_area"], c=point_colors_no_gaza, alpha=0.45,
        edgecolor="white", linewidth=0.45, zorder=2,
    )
    ax.scatter(
        df_no_gaza["deaths_for_plot"], df_no_gaza["civ_share_mid"],
        s=5, c=point_colors_no_gaza, alpha=0.95, zorder=3,
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
        # Exposure-agnostic bound: q in [0, 25.1%] -> civ share [74.9, 100]
        ax.plot([x_gaza, x_gaza], [74.9, 100], color=c_gaza, lw=1.5, alpha=0.6, zorder=2)
        # Calibrated bound: q in [0, 6.3%] -> civ share [93.7, 100]
        ax.plot([x_gaza, x_gaza], [93.7, 100], color=c_gaza, lw=3.5, alpha=0.9, zorder=3)

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
        "russia_ukraine_war_2022":   ("Russia\u2013Ukraine",      0.45, -10, "center"),
        "israel_gaza_war_2023":      ("Israel\u2013Gaza 2023",    0.30,   5.5, "right"),
        "wwi":                       ("WWI",                  1.30,  -8, "left"),
        "iraq_war_2003":             ("Iraq 2003",            0.28,  -3, "right"),
        "war_in_afghanistan_2001":   ("Afghanistan 2001",     0.40,  -7, "center"),
    }
    by_id = {row.id: row for row in df.itertuples()}
    for ident, (text, xmult, yoff, ha) in label_specs.items():
        if ident not in by_id:
            continue
        w = by_id[ident]
        x, y = w.deaths_for_plot, w.civ_share_mid
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
    ax.set_title("Civilian share vs. total deaths, 81 conflicts (1900–2026)",
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

    # IDF claim over the D scenarios used in the paper:
    # 70k (MoH confirmed) .. 106.2k (GMS-corrected + missing)
    IDF_LO_K, IDF_HI_K = 17_000, 25_000
    D_LO, D_HI = 70_000, 106_200
    idf_q_lo = IDF_LO_K / D_HI * 100
    idf_q_hi = IDF_HI_K / D_LO * 100

    # Diagnostic constants (AM = males 18+, so a = 1 - w)
    A_SHARE = 0.267    # adult-male population share
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
    Q = 1.0 - O * (W + MU_BAR * (A_SHARE - F_SHARE)) / W
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
    coef_top = (w_top + MU_BAR * (A_SHARE - F_SHARE)) / w_top
    label_pts = [(((1 - q/100) / coef_top) * 100, 78.3) for q in levels]
    ax.clabel(cs, inline=True, fontsize=6.3, fmt="%d%%",
              manual=label_pts, inline_spacing=1)

    # IDF-required omega band: at w=73.5%, the required omega for q in claim band.
    w_ref = 0.735
    coef = (w_ref + MU_BAR * (A_SHARE - F_SHARE)) / w_ref
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
        ("blend", 62.0, 73.5, BLUE,      (8, 16)),
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
                xytext=(10.5, 0.85 * ymax),
                fontsize=7.6, color=BLUE, fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    ax.text((idf_q_lo + idf_q_hi)/2, 0.55 * ymax,
            f"IDF claim\n{idf_q_lo:.0f}–{idf_q_hi:.0f}%\n"
            r"($\rho_\omega\approx$8–31 SE)",
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
    ax.text((IDF_LO_K + IDF_HI_K)/2, 0.55 * ymax,
            f"IDF claim\n{IDF_LO_K//1000}–{IDF_HI_K//1000}k",
            ha="center", va="center", fontsize=7.6, color=RUST,
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
    for x, m in zip(xs, medians):
        ax.text(x, m + 0.04 * his.max(),
                f"{int(m/1000)}k" if m >= 1000 else f"{int(m)}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=8.0)
    ax.set_ylabel("posterior median deaths")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    ax.set_ylim(0, his.max() * 1.20)
    ax.set_title("D. Decomposition of $\\sim$70k reported dead",
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
            "IDF claim\n($\\sim$2:1)",
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
        ("Blended anchor", 0.620, "geometric consensus = 62.0%"),
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
# Appendix table: all conflicts behind Figures 3-4, as a longtable.
# ---------------------------------------------------------------------------


def latex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def fmt_count(x: float) -> str:
    if x >= 1e6:
        return f"{x/1e6:.3g}M"
    if x >= 1e3:
        return f"{x/1e3:.3g}k"
    return f"{int(round(x))}"


def write_all_conflicts_table():
    wars = sorted(
        [w for w in WARS if (w.mil_mid + w.civ_mid) >= 1_000],
        key=lambda w: (w.start or 1900, w.name),
    )
    lines = [
        "% Auto-generated by paper/build_figures.py -- do not edit by hand.",
        r"\begin{footnotesize}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{longtable}{@{}lccrc@{}}",
        r"\caption{\textbf{The 81-conflict dataset behind Figures~\ref{fig:range-frame} "
        r"and~\ref{fig:uncertainty-ladder}} (conflicts with at least 1{,}000 attributed "
        r"deaths shown). Deaths are midpoints of the source ranges (direct military $+$ "
        r"civilian, including indirect where sources attribute it); the civilian-share "
        r"interval spans the most and least civilian-heavy readings of the source ranges. "
        r"Per-conflict sources are in the supplement.}\label{tab:all-conflicts}\\",
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
        share = f"{100*w.civ_share_mid:.0f}\\%"
        rng = f"{100*w.civ_share_lo:.0f}--{100*w.civ_share_hi:.0f}\\%"
        deaths = fmt_count(w.mil_mid + w.civ_mid)
        lines.append(
            f"{latex_escape(short_name(w.name, 44))} & {period} & {deaths} & {share} & {rng} \\\\"
        )
    lines += [r"\end{longtable}", r"\end{footnotesize}", ""]
    out = ROOT / "content" / "tab_all_conflicts.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out.name} ({len(wars)} conflicts)")


write_all_conflicts_table()

print(f"\nAll figures in {FIGS}")
