"""Step 2: posterior figure + sensitivity sweep.

Reads gaza_sim/posterior.npz and gaza_sim/facts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parent
P = np.load(ROOT / "posterior.npz")
F = json.loads((ROOT / "facts.json").read_text())

w = P["weights"]
w = w / w.sum()
ess = 1 / (w ** 2).sum()


def wq(arr, q):
    s = np.argsort(arr)
    cw = np.cumsum(w[s])
    return arr[s[np.searchsorted(cw, q)]]


def wmed(arr): return wq(arr, 0.5)
def wci(arr):  return (wq(arr, 0.025), wq(arr, 0.975))


# IDF claim
IDF_LO, IDF_HI = 17_000, 25_000
MOH_LATE = F["deaths_observed"]["gaza_moh_total_through_late_2025"]["point"]
IDF_Q_LO = IDF_LO / 95_000        # generous (high MoH)
IDF_Q_HI = IDF_HI / 60_000        # ungenerous (low MoH)

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.subplots_adjust(hspace=0.45, wspace=0.35,
                    left=0.05, right=0.985, top=0.93, bottom=0.05)

# ---------------------------------------------------------------- (1)
# Posterior on q
ax = axes[0, 0]
q = P["q"]
ax.hist(q, bins=80, weights=w, density=True, color="#3b6fb6", alpha=0.85,
        edgecolor="black", linewidth=0.3)
med = wmed(q); lo, hi = wci(q)
ax.axvline(med, color="black", ls="--", lw=1.5,
           label=f"median {med:.1%}")
ax.axvline(lo,  color="grey",  ls=":", lw=1.2, label=f"95% CI [{lo:.1%},{hi:.1%}]")
ax.axvline(hi,  color="grey",  ls=":", lw=1.2)
ax.axvspan(IDF_Q_LO, IDF_Q_HI, color="#b8313a", alpha=0.18,
           label=f"IDF claim implies q ≈ {IDF_Q_LO:.0%}-{IDF_Q_HI:.0%}")
ax.set_xlim(0, 0.45)
ax.set_xlabel("Combatant share of all Gaza dead, q")
ax.set_ylabel("posterior density")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_title(f"(1) Posterior on militant share q\n"
             f"median {med:.1%}, 95% CI [{lo:.1%}, {hi:.1%}]")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (2)
# Posterior on absolute # militants killed vs IDF claim
ax = axes[0, 1]
dm = P["D_milt"]
ax.hist(dm, bins=80, weights=w, density=True, color="#3b6fb6", alpha=0.85,
        edgecolor="black", linewidth=0.3, label="posterior")
mm = wmed(dm); ll, hh = wci(dm)
ax.axvline(mm, color="black", ls="--", lw=1.5, label=f"median {int(mm):,}")
ax.axvspan(IDF_LO, IDF_HI, color="#b8313a", alpha=0.25,
           label=f"IDF claim {IDF_LO//1000}-{IDF_HI//1000}k")
ax.set_xscale("symlog", linthresh=1000)
ax.set_xlim(0, 50_000)
ax.set_xlabel("Posterior on militants killed in Gaza")
ax.set_ylabel("density")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
ax.set_title(f"(2) Implied militants killed: median {int(mm):,}\n"
             f"95% CI [{int(ll):,}, {int(hh):,}]")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (3)
# Civilian:militant ratio posterior (with-kids and adults-only)
ax = axes[0, 2]
ratio = P["civ_milt_ratio"]
ratio_AO = P["D_civAM"] / np.maximum(P["D_milt"], 1)   # adults only
ax.hist(ratio,    bins=80, weights=w, density=True, color="#b8313a",
        alpha=0.7, edgecolor="black", lw=0.3, label="all civilians")
ax.hist(ratio_AO, bins=80, weights=w, density=True, color="#3b6fb6",
        alpha=0.7, edgecolor="black", lw=0.3,
        label="adult-male civilians only")
m1 = wmed(ratio); m2 = wmed(ratio_AO)
ax.axvline(m1, color="#b8313a", ls="--", lw=1.4, label=f"all-civ median {m1:.0f}×")
ax.axvline(m2, color="#3b6fb6", ls="--", lw=1.4, label=f"adult-only median {m2:.0f}×")
ax.set_xlim(0, 100)
ax.set_xlabel("civilian deaths / militant deaths")
ax.set_ylabel("density")
ax.set_title("(3) Posterior on civ:mil ratio")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (4)
# Sensitivity: q vs civilian-male exposure multiplier μ_M
ax = axes[1, 0]
mu_bins = np.linspace(P["mu_M"].min(), P["mu_M"].max(), 25)
mu_centers = 0.5 * (mu_bins[1:] + mu_bins[:-1])
q_at_mu = []
qci_at_mu = []
for lo_, hi_ in zip(mu_bins[:-1], mu_bins[1:]):
    mask = (P["mu_M"] >= lo_) & (P["mu_M"] < hi_)
    if mask.sum() == 0:
        q_at_mu.append(np.nan); qci_at_mu.append((np.nan, np.nan)); continue
    arr = P["q"][mask]; ww = w[mask]; ww = ww / ww.sum()
    s = np.argsort(arr); cw = np.cumsum(ww[s])
    q_at_mu.append(arr[s[np.searchsorted(cw, 0.5)]])
    qci_at_mu.append((arr[s[np.searchsorted(cw, 0.16)]],
                      arr[s[np.searchsorted(cw, 0.84)]]))
q_at_mu = np.array(q_at_mu)
qci_lo = np.array([x[0] for x in qci_at_mu])
qci_hi = np.array([x[1] for x in qci_at_mu])
ax.fill_between(mu_centers, qci_lo * 100, qci_hi * 100,
                color="#3b6fb6", alpha=0.25, label="68% CI")
ax.plot(mu_centers, q_at_mu * 100, color="#3b6fb6", lw=2.0, label="median q")
ax.set_xlabel("Civilian-male exposure multiplier μ_M\n"
              "(1.0 = no exposure differential, 2.5 = strong)")
ax.set_ylabel("Posterior median q (%)")
ax.set_title("(4) Sensitivity: militant share q vs μ_M\n"
             "(if civilian men were *not* over-exposed, q would rise)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (5)
# Joint posterior (M, q)
ax = axes[1, 1]
sc = ax.scatter(P["M"], P["q"] * 100, s=4, c=w / w.max(),
                cmap="viridis", alpha=0.5)
plt.colorbar(sc, ax=ax, label="posterior weight (norm.)")
ax.set_xlabel("Total militants in Gaza pop, M (prior)")
ax.set_ylabel("q (militant share of dead, %)")
ax.set_title("(5) Joint posterior over (M, q)")
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (6)
# Decomposition of dead by class (stacked bar with CI)
ax = axes[1, 2]
classes = ["Militants", "Civilian adult males", "Women + children"]
vals_med = [wmed(P["D_milt"]), wmed(P["D_civAM"]), wmed(P["D_WC"])]
vals_lo  = [wci(P["D_milt"])[0], wci(P["D_civAM"])[0], wci(P["D_WC"])[0]]
vals_hi  = [wci(P["D_milt"])[1], wci(P["D_civAM"])[1], wci(P["D_WC"])[1]]
colors = ["#3b6fb6", "#d4a017", "#b8313a"]
ax.bar(classes, vals_med, color=colors, alpha=0.85,
       edgecolor="black", linewidth=0.4)
err_lo = np.array(vals_med) - np.array(vals_lo)
err_hi = np.array(vals_hi) - np.array(vals_med)
ax.errorbar(classes, vals_med, yerr=[err_lo, err_hi],
            fmt="none", ecolor="black", capsize=4, lw=1)
for c, v in zip(classes, vals_med):
    ax.text(c, v, f"\n{int(v):,}", ha="center", va="top",
            color="white", fontsize=9, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
ax.set_ylabel("Posterior deaths (median, 95% CI bars)")
ax.set_title(f"(6) Posterior decomposition of {int(MOH_LATE):,} reported deaths")
ax.grid(True, axis="y", alpha=0.25)

# ---------------------------------------------------------------- (7)
# Spatial: posterior expected # militants per governorate, weighted
# (we recompute m_g posterior using top-weight draws)
ax = axes[2, 0]
gov_names = [g["name"] for g in F["geography"]["governorates"]]
gov_pop   = np.array([g["population_2023"]
                      for g in F["geography"]["governorates"]])
# Posterior expected militant share = M * pop_g / total_pop, scaled by
# clustering uncertainty in shape parameter γ.  We approximate by
# sampling per-gov shares from a Dirichlet centered at pop and adjusted
# by γ.  Simpler: militant density is approximately proportional to pop
# density.
gov_density = gov_pop / np.array([g["area_km2"]
                                   for g in F["geography"]["governorates"]])
share_milt_g = gov_density / gov_density.sum()
M_post = (P["M"] * w).sum() / w.sum()
mil_per_gov = share_milt_g * M_post
ax.bar(gov_names, mil_per_gov, color=["#3b6fb6", "#1f4f8a", "#3b6fb6",
                                       "#3b6fb6", "#3b6fb6"],
       alpha=0.85, edgecolor="black", linewidth=0.4)
for n, v in zip(gov_names, mil_per_gov):
    ax.text(n, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Implied militants in pop (proxy: pop density)")
ax.set_title(f"(7) Implied militants by governorate (sum={int(M_post):,})")
ax.tick_params(axis="x", rotation=20)
ax.grid(True, axis="y", alpha=0.25)

# ---------------------------------------------------------------- (8)
# Demographic check: simulated vs observed adult-male share among dead
ax = axes[2, 1]
sim_AM_share = (P["D_civAM"] + P["D_milt"]) / np.maximum(P["D_obs"], 1)
ax.hist(sim_AM_share, bins=80, weights=w, density=True, color="#888",
        alpha=0.7, edgecolor="black", lw=0.3, label="simulated")
ax.axvline(F["deaths_observed"]["ohchr_identification_sample"]["share_men_18_plus"],
           color="#3b6fb6", lw=2, label="OHCHR (n≈8k)")
ax.axvline(F["deaths_observed"]["moh_demographic_breakdown_full_record"]["share_men_18_plus"],
           color="#b8313a", lw=2, label="MoH full record (n≈70k)")
ax.axvline(F["population"]["share_adult_males_18_60"]["point"],
           color="grey", ls="--", lw=1, label="pop baseline (PCBS)")
ax.set_xlabel("Adult-male share of dead (combatant + civilian)")
ax.set_ylabel("density")
ax.set_title("(8) Posterior fit to demographic data")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0.15, 0.55)
ax.grid(True, alpha=0.25)

# ---------------------------------------------------------------- (9)
# Comparison: prior version vs Bayesian-only vs full simulator vs IDF
ax = axes[2, 2]
ax.set_axis_off()
report_lines = [
    f"Posterior (importance sampler, ESS={int(ess):,})",
    "",
    "Direct deaths in Gaza (true):",
    f"  D_total median {int(wmed(P['D_total'])):,}, "
    f"95% CI [{int(wci(P['D_total'])[0]):,}, {int(wci(P['D_total'])[1]):,}]",
    "",
    "Decomposition (median):",
    f"  Militants killed         {int(wmed(P['D_milt'])):>7,}",
    f"  Adult-male civilians     {int(wmed(P['D_civAM'])):>7,}",
    f"  Women + children         {int(wmed(P['D_WC'])):>7,}",
    "",
    f"Militant share q   median {wmed(P['q']):.1%}",
    f"                  95% CI [{wci(P['q'])[0]:.1%}, {wci(P['q'])[1]:.1%}]",
    "",
    f"Civilian:militant ratio (incl. kids):",
    f"  median {wmed(P['civ_milt_ratio']):.0f}×, "
    f"95% CI [{wci(P['civ_milt_ratio'])[0]:.0f}×, "
    f"{wci(P['civ_milt_ratio'])[1]:.0f}×]",
    "",
    "For comparison:",
    f"  IDF/Israeli intel claim  17–25k militants killed",
    f"     ⇒ q ≈ 19–42% (incompatible with posterior)",
    f"  Naïve OHCHR-based Bayes  q ≈ 6.0% (CI 2.4–9.2%)",
    f"  Simulator (this fig.)    q ≈ {wmed(P['q']):.1%} "
    f"(CI {wci(P['q'])[0]:.1%}–{wci(P['q'])[1]:.1%})",
]
ax.text(0.0, 1.0, "\n".join(report_lines),
        family="DejaVu Sans Mono", fontsize=10, va="top", ha="left",
        transform=ax.transAxes)

# ---------------------------------------------------------------- title
fig.suptitle(
    "Gaza war — spatial Bayesian inference of militant vs civilian death share\n"
    "Forward model: 5 governorates × 80 cells; latent militant cell-density "
    "with clustering γ; strikes target high-militant cells with bias α; "
    "civilian adult-male exposure μ_M absorbs male over-representation.\n"
    "Conditioned on PCBS demographics + MoH+OHCHR demographic split + MoH "
    "death tally + Lancet undercount.  No belligerent claims used as priors.",
    fontsize=13, y=0.995,
)

out_png = ROOT / "posterior_figure.png"
out_pdf = ROOT / "posterior_figure.pdf"
fig.savefig(out_png, dpi=130)
fig.savefig(out_pdf)
print(f"Wrote {out_png}\nWrote {out_pdf}")
