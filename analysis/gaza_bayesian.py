"""Bayesian / demographic estimate of the combatant share of Gaza deaths.

Setup
-----
Let
    w  = fraction of Gaza POPULATION that is "women + children"
         (UN/OHCHR definition: females of any age + males <18).
         PCBS / UN demographic estimates put this at ~73-75% in Gaza.
    ω  = fraction of OBSERVED Gaza dead who are women + children.
         OHCHR's identified-deaths analyses through 2024-25 have found
         this to be ~70% (44% children + 26% women) on a sample of
         several thousand fully-identified deaths.
    q  = fraction of total deaths who were combatants.
    f  = fraction of the total population who are combatants
         (Hamas + PIJ active strength ≈ 30k of ~2.23M ⇒ ~1.3%).

Assumptions
-----------
A1. Combatants are essentially all adult males (>=18).  Female / minor
    combatants are a few %; we sweep this with a parameter `eps`.
A2. Civilian deaths are demographically representative of the
    civilian population (i.e. women+children die roughly in proportion
    to their share of the civilian population).
A3. We treat the OHCHR identified-deaths sample as representative of
    the broader killed population, with binomial sampling noise.

Then the women+children share of civilian deaths is

        w_civ = (w − eps·f) / (1 − f)   ≈ w / (1 − f)   (eps tiny)

and the observed share is

        ω = (1 − q) · w_civ + q · eps_combatant ,
        where eps_combatant ≈ 0  (women+children fraction of combatants).

Solving:

        q = 1 − (ω / w_civ).

We propagate uncertainty in (w, ω, f) by Monte-Carlo to get a posterior
on q.  We then translate q into an implied number of combatants vs
civilians for the current ~72k Gaza-MoH direct-death tally and compare
to Israel's claim (~17–25k combatants killed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RNG  = np.random.default_rng(20260516)

# ----------------------------------------------------------------------------
# Inputs (sourced; see citations dict below).
# ----------------------------------------------------------------------------

# Gaza population women+children share (PCBS + UN: ~47% under 18, ~50% female,
# yielding females-all-ages + males<18 ≈ 73-75%).  We use Beta(α, β) with
# mean ~0.735 and tight sd.
W_MEAN = 0.735
W_SD   = 0.012        # ~1.2 percentage point uncertainty
# Convert mean+sd to Beta(α, β):
def beta_from_mean_sd(m, s):
    v = s * s
    a = m * ((m * (1 - m)) / v - 1)
    b = (1 - m) * ((m * (1 - m)) / v - 1)
    return a, b
W_ALPHA, W_BETA = beta_from_mean_sd(W_MEAN, W_SD)

# OHCHR-style identification analysis: among ~8,000 fully ID'd Gaza dead
# in late-2024 OHCHR analysis, 44% children + 26% women → 70% w+c.  We
# use the sample size as the binomial weight.  (Larger 2025 samples have
# given similar shares; treat as conservative.)
OHCHR_N      = 8_119
OHCHR_W_PLUS = round(0.70 * OHCHR_N)   # women + children count
# Posterior on ω with Jeffreys prior Beta(0.5, 0.5):
W_OBS_ALPHA = OHCHR_W_PLUS + 0.5
W_OBS_BETA  = OHCHR_N - OHCHR_W_PLUS + 0.5

# Combatant share of population.  Hamas+PIJ effective strength is widely
# estimated 25-40k against a Gaza population of ~2.23M ⇒ ~1.3 %.  We
# sweep this 0.5–2.0 % to be safe.
F_LOW, F_HIGH = 0.005, 0.020

# Total reported direct deaths in Gaza (Gaza MoH, mid-2026, ~72k).  We
# bracket because of dispute / undercount estimates.
TOTAL_DEATHS_LOW  = 60_000
TOTAL_DEATHS_HIGH = 90_000          # incl. some indirect / undercount adjustment

# Israel's belligerent claim of Hamas+PIJ combatants killed.
IDF_CLAIM_LOW, IDF_CLAIM_HIGH = 17_000, 25_000

CITATIONS = {
    "PCBS_Gaza_population_2023": "https://www.pcbs.gov.ps/site/lang__en/881/default.aspx",
    "OHCHR_Nov2024_identified_deaths":
        "https://www.ohchr.org/en/press-releases/2024/11/un-human-rights-office-issues-thematic-report-killing-women-and-children-gaza",
    "Gaza_MoH_running_total_2025_2026": "https://www.ochaopt.org/data/casualties",
    "Khatib_Lancet_2024_excess_mortality":
        "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01169-3/fulltext",
    "Hamas_PIJ_strength_estimates": "https://www.cfr.org/backgrounder/who-are-hamas-and-other-palestinian-armed-groups",
}


# ----------------------------------------------------------------------------
# Monte-Carlo posterior on q
# ----------------------------------------------------------------------------

N = 200_000

w     = stats.beta.rvs(W_ALPHA, W_BETA, size=N, random_state=RNG)
omega = stats.beta.rvs(W_OBS_ALPHA, W_OBS_BETA, size=N, random_state=RNG)
f     = stats.uniform.rvs(F_LOW, F_HIGH - F_LOW, size=N, random_state=RNG)

# Civilian-share of women+children
w_civ = w / (1 - f)
q     = 1 - omega / w_civ
q     = np.clip(q, 0, 1)            # negative q means model rejected

# Posterior summaries
q_mean   = float(q.mean())
q_med    = float(np.median(q))
q_lo, q_hi = np.quantile(q, [0.025, 0.975])
print(f"Posterior on combatant share q:")
print(f"  mean   = {q_mean:6.3%}")
print(f"  median = {q_med:6.3%}")
print(f"  95% CI = [{q_lo:6.3%}, {q_hi:6.3%}]")

# Implied combatants / civilians at current Gaza death tallies
total = stats.uniform.rvs(TOTAL_DEATHS_LOW,
                          TOTAL_DEATHS_HIGH - TOTAL_DEATHS_LOW,
                          size=N, random_state=RNG)
combatants = q * total
civilians  = (1 - q) * total

print()
print(f"Implied # combatants killed in Gaza (mid-2026 direct deaths):")
print(f"  median = {int(np.median(combatants)):>7,d}")
print(f"  95% CI = [{int(np.quantile(combatants, 0.025)):>7,d}, "
      f"{int(np.quantile(combatants, 0.975)):>7,d}]")
print(f"  IDF claim = {IDF_CLAIM_LOW:,}–{IDF_CLAIM_HIGH:,}")
print()
print(f"Implied civ:mil ratio (Gaza only, direct deaths):")
print(f"  median = {np.median(civilians/np.maximum(combatants,1)):.1f}×")
print(f"  95% CI = [{np.quantile(civilians/np.maximum(combatants,1), 0.025):.1f}×, "
      f"{np.quantile(civilians/np.maximum(combatants,1), 0.975):.1f}×]")


# ----------------------------------------------------------------------------
# Sensitivity sweep over w and ω
# ----------------------------------------------------------------------------

w_grid     = np.linspace(0.66, 0.80, 80)
omega_grid = np.linspace(0.55, 0.85, 80)
QQ = np.zeros((len(w_grid), len(omega_grid)))
for i, ww in enumerate(w_grid):
    for j, oo in enumerate(omega_grid):
        QQ[i, j] = max(0.0, 1 - oo / (ww / (1 - 0.013)))


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         gridspec_kw={"width_ratios": [1, 1.05, 1.1]})

# --- Panel 1: posterior on q ---
ax = axes[0]
ax.hist(q, bins=80, density=True, color="#3b6fb6", alpha=0.85,
        edgecolor="black", linewidth=0.3)
for v, lbl, c in [(q_med, f"median {q_med:.1%}", "black"),
                  (q_lo,  f"2.5%  {q_lo:.1%}",  "grey"),
                  (q_hi,  f"97.5% {q_hi:.1%}", "grey")]:
    ax.axvline(v, color=c, ls="--", lw=1.2)
# IDF claim band
idf_q_lo = IDF_CLAIM_LOW  / TOTAL_DEATHS_HIGH    # most generous to IDF
idf_q_hi = IDF_CLAIM_HIGH / TOTAL_DEATHS_LOW
ax.axvspan(idf_q_lo, idf_q_hi, color="#b8313a", alpha=0.18,
           label=f"IDF claim implies q ≈ {idf_q_lo:.0%}–{idf_q_hi:.0%}")
ax.set_xlim(0, 0.5)
ax.set_xlabel("Combatant share of Gaza dead (q)")
ax.set_ylabel("posterior density")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_title(f"(1) Posterior on q\nmedian {q_med:.1%}, 95% CI [{q_lo:.1%}, {q_hi:.1%}]")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

# --- Panel 2: sensitivity heatmap ---
ax = axes[1]
im = ax.imshow(QQ * 100, origin="lower",
               extent=(omega_grid[0]*100, omega_grid[-1]*100,
                       w_grid[0]*100,     w_grid[-1]*100),
               aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=40)
cs = ax.contour(omega_grid*100, w_grid*100, QQ*100,
                levels=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                colors="black", linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f%%")
# Mark the OHCHR/PCBS point
ax.scatter([70], [W_MEAN*100], s=180, color="white", edgecolor="black",
           linewidths=1.5, zorder=5)
ax.annotate("OHCHR ω=70%\nPCBS w=73.5%", (70, W_MEAN*100),
            xytext=(8, -22), textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
# Mark the IDF-claim implied point
idf_omega = 100 - 100 * (IDF_CLAIM_LOW + IDF_CLAIM_HIGH) / 2 / \
            ((TOTAL_DEATHS_LOW + TOTAL_DEATHS_HIGH) / 2)
# IDF claim works only if ω is far below observation; show at the bottom
cb = plt.colorbar(im, ax=ax)
cb.set_label("implied combatant share q (%)")
ax.set_xlabel("Observed women+children share among dead, ω (%)")
ax.set_ylabel("Population women+children share, w (%)")
ax.set_title("(2) Sensitivity: q = 1 − ω / w_civ\n(black contours = q in %)")

# --- Panel 3: implied # combatants vs IDF claim ---
ax = axes[2]
ax.hist(combatants, bins=80, density=True, color="#3b6fb6", alpha=0.85,
        edgecolor="black", linewidth=0.3, label="Demographic posterior")
ax.axvspan(IDF_CLAIM_LOW, IDF_CLAIM_HIGH, color="#b8313a", alpha=0.25,
           label=f"IDF/Israeli intel claim: {IDF_CLAIM_LOW//1000}–{IDF_CLAIM_HIGH//1000}k")
med = float(np.median(combatants))
ax.axvline(med, color="black", ls="--", lw=1.5,
           label=f"median ≈ {int(med):,}")
ax.set_xscale("symlog", linthresh=1000)
ax.set_xlim(0, 50_000)
ax.set_xlabel("Implied combatants among Gaza dead (mid-2026)")
ax.set_ylabel("density")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
ax.set_title("(3) Implied combatants vs IDF/Israeli intel claim")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)

fig.suptitle(
    "Gaza: demographic Bayesian inference of combatant share among the dead\n"
    "Inputs: OHCHR ID'd-deaths analysis (~70% women+children), PCBS Gaza pop "
    "(women+children ~73.5%), Hamas+PIJ pop share ~0.5–2%.",
    fontsize=12, y=1.02,
)
fig.tight_layout()

FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
out_png = FIGURES / "gaza_bayesian.png"
out_pdf = FIGURES / "gaza_bayesian.pdf"
fig.savefig(out_png, dpi=140, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"\nWrote {out_png}\nWrote {out_pdf}")


# ----------------------------------------------------------------------------
# Update the Gaza data.json with the Bayesian-corrected combatant share
# ----------------------------------------------------------------------------

data_path = ROOT / "data" / "per_war" / "israel_gaza_war_2023.json"
data = json.loads(data_path.read_text())

bayes_block = {
    "method": "demographic_bayesian (q = 1 − ω/w_civ)",
    "inputs": {
        "w_population_women_plus_children": {
            "mean": W_MEAN, "sd": W_SD,
            "source": "PCBS / UN — Gaza ~47% <18, ~50% female ⇒ w ≈ 0.735"
        },
        "omega_observed_women_plus_children_share": {
            "mean": 0.70, "sample_size": OHCHR_N,
            "source": "UN OHCHR Nov 2024 thematic analysis of identified Gaza deaths"
        },
        "f_combatants_in_population": {
            "low": F_LOW, "high": F_HIGH,
            "source": "Hamas+PIJ effective strength ≈ 25-40k vs Gaza pop ≈ 2.23M"
        },
    },
    "posterior_q": {
        "mean": round(q_mean, 4),
        "median": round(q_med, 4),
        "ci_95": [round(q_lo, 4), round(q_hi, 4)],
    },
    "implied_combatants_killed_gaza_direct": {
        "median": int(np.median(combatants)),
        "ci_95": [int(np.quantile(combatants, 0.025)),
                  int(np.quantile(combatants, 0.975))],
        "vs_idf_claim": [IDF_CLAIM_LOW, IDF_CLAIM_HIGH],
    },
    "implied_civ_to_mil_ratio_direct_only": {
        "median": float(round(np.median(civilians / np.maximum(combatants, 1)), 1)),
        "ci_95": [float(round(np.quantile(civilians / np.maximum(combatants, 1), 0.025), 1)),
                  float(round(np.quantile(civilians / np.maximum(combatants, 1), 0.975), 1))],
    },
    "citations": CITATIONS,
}
data["bayesian_combatant_share"] = bayes_block

# Patch the Hamas-side military_killed to reflect the Bayesian range,
# pushing the IDF claim down to a notes field.
for s in data.get("sides", []):
    if "Hamas" in s.get("name", "") or "Palestinian armed" in s.get("name", ""):
        old = s.get("military_killed") or {}
        s["military_killed"] = {
            "low":  int(np.quantile(combatants, 0.025)),
            "high": int(np.quantile(combatants, 0.975)),
            "point": int(np.median(combatants)),
            "notes": (
                "Bayesian/demographic estimate from OHCHR women+children share "
                "(70%) and PCBS population w+c share (~73.5%): q ≈ "
                f"{q_med:.1%}, 95% CI [{q_lo:.1%}, {q_hi:.1%}].  Israel's "
                f"military-intel claim of {IDF_CLAIM_LOW//1000}–{IDF_CLAIM_HIGH//1000}k "
                "Hamas/PIJ fighters killed implies a combatant share that is "
                "incompatible with the demographic profile of identified dead "
                "and is recorded as a belligerent claim only."
            ),
            "sources": [
                CITATIONS["OHCHR_Nov2024_identified_deaths"],
                CITATIONS["PCBS_Gaza_population_2023"],
            ],
            "belligerent_claim_idf": [IDF_CLAIM_LOW, IDF_CLAIM_HIGH],
        }
        s["civilians_killed_directly"] = s.get("civilians_killed_directly") or {}
        old_cd = s["civilians_killed_directly"]
        # bump civilian count by the difference between IDF-floor and Bayesian-q
        # (those people were civilians under the demographic model).
        # Keep the original 7-Oct civilian-victims figure if present.
        # We don't change civilians_killed_directly under Hamas (that field
        # tracks civilians KILLED BY Hamas, not by Israel).
        old_cd.setdefault("notes", "")

# Recompute totals with Bayesian-corrected military and inflate civilian
# by the same delta on Israel's side.
def _safe(x): return x if x is not None else 0
sides = data.get("sides") or []
isr = next((s for s in sides if s["name"].startswith("Israel")), None)
hamas = next((s for s in sides if "Hamas" in s["name"]), None)
if isr and hamas:
    delta_low  = max(0, IDF_CLAIM_LOW  - hamas["military_killed"]["low"])
    delta_high = max(0, IDF_CLAIM_HIGH - hamas["military_killed"]["high"])
    cd = isr.get("civilians_killed_directly") or {}
    cd["low"]  = _safe(cd.get("low"))  + delta_low
    cd["high"] = _safe(cd.get("high")) + delta_high
    cd["notes"] = (cd.get("notes") or "") + (
        f"  +Bayesian shift: {delta_low:,}–{delta_high:,} 'combatant' deaths "
        "claimed by Israel are reclassified to civilians-killed-directly under "
        "the demographic model."
    )
    isr["civilians_killed_directly"] = cd

# Recompute summary totals
mil_lo = sum(_safe(s.get("military_killed", {}).get("low")) for s in sides)
mil_hi = sum(_safe(s.get("military_killed", {}).get("high")) for s in sides)
civ_lo = sum(_safe(s.get("civilians_killed_directly", {}).get("low"))
             for s in sides)
civ_hi = sum(_safe(s.get("civilians_killed_directly", {}).get("high"))
             for s in sides)
ind_lo = sum(_safe(i.get("low"))  for s in sides for i in s.get("deaths_from_actions") or [])
ind_hi = sum(_safe(i.get("high")) for s in sides for i in s.get("deaths_from_actions") or [])
data["totals"] = {
    "military_low": mil_lo, "military_high": mil_hi,
    "civilian_low": civ_lo + ind_lo, "civilian_high": civ_hi + ind_hi,
    "grand_low":  mil_lo + civ_lo + ind_lo,
    "grand_high": mil_hi + civ_hi + ind_hi,
    "notes": "Totals recomputed after Bayesian/demographic correction of "
             "Hamas combatant-killed bucket; see `bayesian_combatant_share`.",
}

data_path.write_text(json.dumps(data, indent=2))
print(f"\nUpdated {data_path}")
print(f"Recomputed totals: mil {mil_lo:,}–{mil_hi:,}, civ "
      f"{civ_lo+ind_lo:,}–{civ_hi+ind_hi:,}, "
      f"grand {mil_lo+civ_lo+ind_lo:,}–{mil_hi+civ_hi+ind_hi:,}")
