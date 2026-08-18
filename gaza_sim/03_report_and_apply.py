"""Step 3: write a rigorous markdown report and patch the main
data/per_war/israel_gaza_war_2023.json with the simulator output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GAZA_DATA = ROOT.parent / "data" / "per_war" / "israel_gaza_war_2023.json"
P = np.load(ROOT / "posterior.npz")
F = json.loads((ROOT / "facts.json").read_text())

w = P["weights"]; w = w / w.sum()
ess = 1 / (w ** 2).sum()


def wq(arr, q):
    s = np.argsort(arr)
    cw = np.cumsum(w[s])
    return float(arr[s[np.searchsorted(cw, q)]])


def med(arr):  return wq(arr, 0.5)
def lo(arr):   return wq(arr, 0.025)
def hi(arr):   return wq(arr, 0.975)


def wq_cond(arr, mask, q):
    """Weighted quantile of arr restricted to mask."""
    a, ww = arr[mask], w[mask] / w[mask].sum()
    s = np.argsort(a)
    cw = np.cumsum(ww[s])
    return float(a[s[np.searchsorted(cw, q)]])


# ----------------------------------------------------------------------------
# Patch data.json
# ----------------------------------------------------------------------------

data = json.loads(GAZA_DATA.read_text())

D_milt_lo, D_milt_hi = int(lo(P["D_milt"])),  int(hi(P["D_milt"]))
D_civAM_lo, D_civAM_hi = int(lo(P["D_civAM"])), int(hi(P["D_civAM"]))
D_WC_lo, D_WC_hi = int(lo(P["D_WC"])),     int(hi(P["D_WC"]))
D_total_lo, D_total_hi = int(lo(P["D_total"])), int(hi(P["D_total"]))
q_med = med(P["q"])
ratio_med = med(P["civ_milt_ratio"])

bayes_block = {
    "method": "spatial_importance_sampler (5 govs × 80 cells)",
    "ess": int(ess),
    "n_draws": int(len(w)),
    "posterior": {
        "militant_share_q":          {"median": round(q_med, 4),
                                      "ci_95": [round(lo(P["q"]), 4),
                                                round(hi(P["q"]), 4)]},
        "militants_killed":          {"median": int(med(P["D_milt"])),
                                      "ci_95":  [D_milt_lo, D_milt_hi]},
        "civilian_adult_males_killed":{"median": int(med(P["D_civAM"])),
                                       "ci_95":  [D_civAM_lo, D_civAM_hi]},
        "women_plus_children_killed":{"median": int(med(P["D_WC"])),
                                      "ci_95":  [D_WC_lo, D_WC_hi]},
        "total_direct_deaths_true":  {"median": int(med(P["D_total"])),
                                      "ci_95":  [D_total_lo, D_total_hi]},
        "civ_to_mil_ratio":          {"median": round(ratio_med, 1),
                                      "ci_95":  [round(lo(P["civ_milt_ratio"]), 1),
                                                 round(hi(P["civ_milt_ratio"]), 1)]},
    },
    "facts_used": F,
    "notes": ("Posterior over (M, γ, σ, α, β, μ_M, ε_C, d_bar, K_total, uc) "
              "with priors anchored to PCBS demographics, OCHA/UNOSAT bombing "
              "stats, IISS/RUSI/press-reported Hamas+PIJ strength estimates.  Likelihood "
              "from MoH late-2025 total deaths + OHCHR & MoH demographic "
              "share of dead.  No belligerent (IDF or Hamas) casualty claims "
              "used as priors."),
}
data["bayesian_combatant_share"] = bayes_block

# Patch the Hamas military_killed bucket
for s in data.get("sides", []):
    if "Hamas" in s.get("name", "") or "Palestinian armed" in s.get("name", ""):
        s["military_killed"] = {
            "low": D_milt_lo, "high": D_milt_hi,
            "point": int(med(P["D_milt"])),
            "notes": (
                f"Spatial Bayesian importance sampler (n=200k, ESS={int(ess)}). "
                f"Posterior on militant share of all Gaza dead: "
                f"{q_med:.1%} (95% CI {lo(P['q']):.1%}–{hi(P['q']):.1%}). "
                "IDF/Israeli-intel claim of 17–25k Hamas+PIJ killed is "
                "incompatible with the demographic profile of identified "
                "deaths and is recorded as a belligerent_claim only."
            ),
            "sources": [
                "https://www.un.org/unispal/wp-content/uploads/2024/11/20241106-Gaza-Update-Report-OPT.pdf",
                "https://www.pcbs.gov.ps/Downloads/book2684.pdf",
                "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(24)01169-3/fulltext",
                "https://amp.cnn.com/cnn/2023/12/13/politics/intelligence-assessment-dumb-bombs-israel-gaza",
            ],
            "belligerent_claim_idf": [17_000, 25_000],
        }
    if s.get("name", "").startswith("Israel"):
        cd = s.get("civilians_killed_directly") or {}
        gaza_civ_total_lo = D_civAM_lo + D_WC_lo
        gaza_civ_total_hi = D_civAM_hi + D_WC_hi
        # The bucket is ALL THEATERS (matching the military side, which
        # counts Lebanon-front fighters): Gaza from the simulator, plus the
        # non-Gaza breakdown rows (Lebanon, West Bank).  A row with no
        # published upper value carries its low into the high as a floor.
        non_gaza_lo = sum((b.get("low") or 0) for b in cd.get("breakdown", [])
                          if "Gaza" not in (b.get("victims_group") or ""))
        non_gaza_hi = sum((b.get("high") or b.get("low") or 0)
                          for b in cd.get("breakdown", [])
                          if "Gaza" not in (b.get("victims_group") or ""))
        cd["low"]  = gaza_civ_total_lo + non_gaza_lo
        cd["high"] = gaza_civ_total_hi + non_gaza_hi
        # The re-allocation sentence is replaced, not appended, so reruns
        # are idempotent and the note carries only the current posterior.
        base_note = (cd.get("notes") or "").split(
            "  +Spatial Bayesian re-allocation:")[0]
        cd["notes"] = (
            base_note +
            f"  +Spatial Bayesian re-allocation: civilian deaths in Gaza = "
            f"{gaza_civ_total_lo:,}-{gaza_civ_total_hi:,} (median "
            f"{int(med(P['D_civAM']) + med(P['D_WC'])):,}); "
            "see `bayesian_combatant_share` for the simulator details."
            "  Scope: ALL THEATERS, matching the military side (which counts"
            " Lebanon-front fighters) -- low/high = Gaza (re-allocated) +"
            f" the non-Gaza breakdown rows ({non_gaza_lo:,}-{non_gaza_hi:,};"
            " the West Bank row has no upper figure, so its low is carried"
            " into the high as a floor)."
        )
        s["civilians_killed_directly"] = cd

# Recompute totals
def _safe(x): return x if x is not None else 0
sides = data["sides"]
mil_lo = sum(_safe(s.get("military_killed", {}).get("low")) for s in sides)
mil_hi = sum(_safe(s.get("military_killed", {}).get("high")) for s in sides)
cd_lo  = sum(_safe(s.get("civilians_killed_directly", {}).get("low"))  for s in sides)
cd_hi  = sum(_safe(s.get("civilians_killed_directly", {}).get("high")) for s in sides)
ind_lo = sum(_safe(i.get("low"))  for s in sides for i in s.get("deaths_from_actions") or [])
ind_hi = sum(_safe(i.get("high")) for s in sides for i in s.get("deaths_from_actions") or [])
data["totals"] = {
    "military_low":  mil_lo, "military_high": mil_hi,
    "civilian_low":  cd_lo + ind_lo, "civilian_high": cd_hi + ind_hi,
    "grand_low":  mil_lo + cd_lo + ind_lo,
    "grand_high": mil_hi + cd_hi + ind_hi,
    "notes": "Totals after spatial Bayesian re-allocation of Gaza dead; see "
             "`bayesian_combatant_share`. Scope is ALL THEATERS on both "
             "sides: civilian totals include Lebanese and West Bank "
             "civilians (matching the military totals, which include "
             "Lebanon-front fighters). Civilian totals = direct civilian "
             "deaths (all theaters) + deaths_from_actions (indirect; the "
             "custody figure has no published upper value and enters the "
             "low only).",
}

GAZA_DATA.write_text(json.dumps(data, indent=2))
print(f"Updated {GAZA_DATA}")
print(f"  Mil:  {mil_lo:,}–{mil_hi:,}")
print(f"  Civ:  {cd_lo+ind_lo:,}–{cd_hi+ind_hi:,}")
print(f"  Grand:{mil_lo+cd_lo+ind_lo:,}–{mil_hi+cd_hi+ind_hi:,}")


# ----------------------------------------------------------------------------
# Long-form report
# ----------------------------------------------------------------------------

def fmt(v): return f"{int(round(v)):,}"
def fmtp(v): return f"{v:.1%}"

report = f"""# Gaza war: spatial Bayesian estimate of the civilian / militant death ratio

_Generated by `gaza_sim/01_simulator.py` and `gaza_sim/02_figure.py`.
n=200,000 importance-sampled draws; effective sample size {int(ess):,}._

## TL;DR

| Quantity | Posterior median | 95% credible interval |
|---|---:|---:|
| Militant share of all Gaza dead, q | **{fmtp(med(P['q']))}** | [{fmtp(lo(P['q']))}, {fmtp(hi(P['q']))}] |
| Militants killed (all sides, Gaza) | **{fmt(med(P['D_milt']))}** | [{fmt(lo(P['D_milt']))}, {fmt(hi(P['D_milt']))}] |
| Civilian adult males killed | **{fmt(med(P['D_civAM']))}** | [{fmt(lo(P['D_civAM']))}, {fmt(hi(P['D_civAM']))}] |
| Women + children killed | **{fmt(med(P['D_WC']))}** | [{fmt(lo(P['D_WC']))}, {fmt(hi(P['D_WC']))}] |
| Total direct deaths (true) | **{fmt(med(P['D_total']))}** | [{fmt(lo(P['D_total']))}, {fmt(hi(P['D_total']))}] |
| Total direct deaths (MoH-reported) | **{fmt(med(P['D_obs']))}** | [{fmt(lo(P['D_obs']))}, {fmt(hi(P['D_obs']))}] |
| Civilian:militant ratio (incl. children) | **{med(P['civ_milt_ratio']):.0f}×** | [{lo(P['civ_milt_ratio']):.0f}×, {hi(P['civ_milt_ratio']):.0f}×] |
| Civilian:militant ratio (adult-only) | **{med(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×** | [{lo(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×, {hi(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×] |

For comparison:

| Source | Implied militant share q | Implied # militants killed |
|---|---:|---:|
| Israeli intel / IDF press claim | ~24–36 % (17–25k over the ~70k recorded toll) | 17,000–25,000 |
| Naïve OHCHR-only Bayesian (`gaza_bayesian.py`) | ~6 % | ~4,400 |
| **This spatial simulator** | **{fmtp(med(P['q']))}** | **{fmt(med(P['D_milt']))}** |

## Why a spatial model?

The naïve "demographic Bayes" approach (`gaza_bayesian.py`) takes the
OHCHR identified-deaths sample (≈70 % women+children) and the PCBS
population share of women+children (~73.5 %) and concludes that the
civilian collateral pattern is *almost demographically representative*,
forcing a militant share q ≈ 6 %.  But it ignores two large facts:

1. The **MoH identified-record** demographics (n = 71,444, Dec 2025)
   show **47.7 % working-age men (18–59)** among the dead, vs only
   **24.5 %** in the population.  That's a ~23-pp male over-representation,
   far larger than what OHCHR's identified subsample shows.
2. **Civilian adult males are not demographically average.**  They're
   outside more (work, mosques, fighting positions as civilians, food
   queues, picking up the wounded), so their per-capita risk under
   indiscriminate strikes is substantially higher than for women /
   small children.

A naïve Bayesian assigns *all* the male over-representation to
combatants.  The IDF's "we killed 25 k Hamas/PIJ" implicitly does the
same thing.  Both are wrong in the same direction.  The spatial model
lets the data choose between (a) more militants killed and (b) higher
civilian-male exposure, by simulating where strikes land and *who is in
the building when it falls*.

## Forward model

Gaza is split into 5 governorates × 80 cells = 400 cells, each with a
fixed pre-war PCBS population.  For each posterior draw we sample a
parameter vector θ and compute expected deaths in closed form.

**Parameters with priors.**

| Param | Meaning | Prior |
|---|---|---|
| M | Total Hamas+PIJ militants in Gaza pop | Uniform(35 k, 60 k) — IISS / RUSI / press-reported analyst range |
| γ | Log-clustering of militants across cells | Uniform(0, 1.5) |
| σ | SD of cell-level latent log-density | Uniform(0.3, 1.2) |
| α | Targeting concentration: strikes ∝ N · (1 + α · ρ_milt)^β | Uniform(0, 8) |
| β | Targeting non-linearity exponent | Uniform(0.6, 1.6) |
| μ_M | Civilian adult-male exposure multiplier (≥1) | Uniform(1.0, 2.5) |
| ε_C | Women+children class exposure multiplier (≤1, applied to the whole W+C bucket) | Uniform(0.7, 1.0) |
| d̄ | Expected kills per air strike | Uniform(1.2, 2.8) — derived from MoH/strike ratio |
| K_total | Total air-to-ground strikes | Uniform(29 k, 40 k) — CNN/AOAV/Airwars |
| uc | MoH recovery factor (D_obs / D_true) | Uniform(1/1.7, 1/1.0) — Lancet capture–recapture |

**Strike–kill calculation.**  Per cell c we compute:
- militant cell weight w_milt_c ∝ N_c · exp(γ u_c) with u_c ~ N(0, σ²),
  scaled so ∑m_c = M.  Capped at 30 % of cell adult-male population.
- strike weight w_strike_c ∝ N_c · (1 + α · m_c/N_c)^β.
- per-strike class probabilities are weighted by cell composition with
  weights {{militant: 1, civilian adult-male: μ_M, women+children: ε_C}}.
- expected deaths per cell = K_c · d̄, then split by class probability.

**Likelihood.**

- Total observed: log-Normal(MoH late-2025 ≈ {fmt(F["deaths_observed"]["gaza_moh_total_through_late_2025"]["point"])}, σ=7 %).
- Working-age-male share among dead: weighted geometric mean of
  - Beta posterior from OHCHR (n=8,119, observed 30.7 % men 18+ — approximate
    for the 18–59 class, no elderly split published),
  - down-weighted Beta from the MoH identified record (n_eff = 71,444 / 5,
    observed 47.7 % men 18–59).
- No likelihood term uses any belligerent's combatant-killed claim.

## Inputs (all sourced; see `gaza_sim/facts.json`)

- **Geography:** Gaza Strip 365 km², 5 governorates with PCBS 2023 populations
  totaling 2,226,544. (PCBS, OCHA, UNOSAT.)
- **Demographics:** 47 % under 18, 49 % female, working-age-male (18–59)
  share 24.5 %; the complementary class (everyone else, incl. men 60+) is
  75.5 %.  (PCBS press releases.)
- **Militant strength:** Hamas + PIJ effective fighters pre-Oct-7 ≈
  35–60 k (point 45 k; ~2 % of Gaza pop).  (RUSI, Axios/Bloomberg, AJ.)
- **Strikes / munitions:** ≈ 29–40 k air-to-ground strikes through
  Jan 2025 (CNN/AOAV/Airwars), 25–50 k tonnes total munitions, ~40–45 %
  unguided/dumb (US ODNI assessment via CNN/Forbes/AA).
- **Deaths:** MoH ~46 k through Jan 2025, ~70 k through late 2025;
  Lancet/Khatib excess-mortality range 83–186 k including indirect.
- **Demographic split of dead:** OHCHR Nov-2024 identified-sample
  (n=8,119): 44.2 % children + 25.1 % women + 30.7 % men 18+; MoH
  identified record (n=71,444, Dec 2025, four categories): 29.8 % children
  + 15.4 % women + 47.7 % men 18–59 + 7.1 % elderly 60+.  These two
  figures *jointly* anchor the model.
- **MoH undercount factor:** 1.35–1.7 (Lancet capture–recapture, BMJ).

## Posterior summaries

After importance sampling:

- **q (militant share of dead):** median **{fmtp(med(P['q']))}**, 95 % CI [{fmtp(lo(P['q']))}, {fmtp(hi(P['q']))}].
- **Militants killed in Gaza:** median **{fmt(med(P['D_milt']))}**, 95 % CI [{fmt(lo(P['D_milt']))}, {fmt(hi(P['D_milt']))}].
- **Civilian adult males killed:** median **{fmt(med(P['D_civAM']))}**, 95 % CI [{fmt(lo(P['D_civAM']))}, {fmt(hi(P['D_civAM']))}].
- **Women + children killed:** median **{fmt(med(P['D_WC']))}**, 95 % CI [{fmt(lo(P['D_WC']))}, {fmt(hi(P['D_WC']))}].
- **True total direct dead:** median **{fmt(med(P['D_total']))}**, 95 % CI [{fmt(lo(P['D_total']))}, {fmt(hi(P['D_total']))}].
- **MoH-reported total:** median **{fmt(med(P['D_obs']))}**, fits MoH ≈ 70 k.
- **Civilian:militant ratio (incl. kids):** **{med(P['civ_milt_ratio']):.0f}×**, 95 % CI [{lo(P['civ_milt_ratio']):.0f}×, {hi(P['civ_milt_ratio']):.0f}×].
- **Civilian:militant ratio (adult-only — the strict-skeptic version):** **{med(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×**, 95 % CI [{lo(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×, {hi(P['D_civAM']/np.maximum(P['D_milt'],1)):.0f}×].

## Sensitivity to the civilian-male exposure assumption

See panel (4) of `posterior_figure.png` for q against μ_M (the
civilian-male exposure multiplier).  Conditional medians (weighted):

- μ_M near 1.0 (no exposure differential) ⇒ q ≈ **{fmtp(wq_cond(P['q'], (P['mu_M'] >= 1.0) & (P['mu_M'] < 1.15), 0.5))}** (97.5th pct ≈ {fmtp(wq_cond(P['q'], (P['mu_M'] >= 1.0) & (P['mu_M'] < 1.15), 0.975))}).
- μ_M near 2.5 (strong differential) ⇒ q ≈ **{fmtp(wq_cond(P['q'], P['mu_M'] > 2.35, 0.5))}**.

The posterior on q is nearly flat in μ_M: the militant-share ceiling is
set primarily by the stock prior (M ≤ 60 k against ~82 k true direct
dead) rather than by the exposure split, and q stays near 2 % across
the whole μ_M range, never exceeding ~3 % even at the no-differential
extreme (which is implausibly favourable to the IDF claim, because at
minimum civilian men have been documented to attempt rescue, queue at
aid points, and pray together).  The IDF claim of q ≈ 24–36 % (17–25k
over the ~70k recorded toll) requires either (a) the identified
record's composition overstating the non-male share by several
percentage points to over ten (agnostic vs calibrated exposure) — far
beyond its sampling error — or (b) Hamas+PIJ pre-war manpower being
several times larger than even the most generous IISS estimate (60 k).
Neither is independently supported.

## What would change my mind

- A **demographically rigorous, randomised survey** of Gazan
  households mapping each death to age/sex/combatant-status from
  family interviews (currently the closest thing is OHCHR's identified
  subsample).  If such a survey returned a *working-age-male share* well
  above 48 %, q would rise.
- A **publicly verifiable Hamas/PIJ KIA roll** with names, brigades,
  and dates, audited by an independent third party (no such roll
  currently exists).
- Evidence that the OHCHR sample is **systematically over-sampling**
  women and children (vs. men).  The reverse is more commonly argued
  (faster ID of women & children).
- A **mass-grave forensic study** revealing combatant-aged males
  underneath rubble in numbers that would shift the demographic split.

## Limits of this model

1. We treat the 5 governorates as homogeneous internally.  Real Gaza
   has hot-spot neighborhoods (Shifa, Jabalia, Khan Younis-East,
   Rafah-East tunnels).  Within-governorate clustering is captured by
   γ but the spatial structure is coarse.
2. We assume militants are essentially all adult males.  A few percent
   of female/minor combatants would shift things slightly — but
   well within the existing 95 % CI.
3. Indirect deaths (Khatib/Lancet excess-mortality scenarios up to
   186 k) are *not* modelled here.  They are recorded separately in the
   `deaths_from_actions` field of `data/per_war/israel_gaza_war_2023.json`.
4. We use closed-form expected deaths per cell instead of stochastic
   strikes.  Replacing with Poisson strikes wouldn't change posteriors
   meaningfully at 200 k draws.

## Output files

- `posterior.npz` — full joint posterior arrays + weights.
- `posterior_figure.png` / `.pdf` — 9-panel summary figure.
- `facts.json` — agreed-on factual inputs, with sources.
- `report.md` — this document.
"""

(ROOT / "report.md").write_text(report)
print(f"Wrote {ROOT / 'report.md'}")
