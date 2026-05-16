"""Step 1: spatial Bayesian Gaza simulator.

Forward model
-------------
Cells: each Gaza governorate is split into K cells; cell c lives in
governorate g(c) and has a fixed population N_c (from PCBS).  Demographic
fractions (women+children w_c, adult-male share f_M_c) are taken to be
constant across cells = Gaza-wide values.

Latent quantities (sampled per posterior draw):

  M           total Hamas+PIJ militants in Gaza, ~Uniform(35k, 60k)
  γ           clustering log-amplification of militants across cells
                (γ=0 ⇒ uniform per-capita; γ large ⇒ concentrated)
  σ           SD of cell-level latent log-density u_c ~ N(0, σ²)
  α           targeting concentration: per-strike rate ∝ N_c·(1 + α·m_c/N_c)^β
  β           targeting non-linearity exponent
  μ_M         civilian-male exposure multiplier (≥1; civilian men die
                at higher per-capita rate than women+children because
                they're outside more)
  ε_C         child exposure multiplier (≤1; children less exposed than
                adult civilian women)
  d_bar       expected kills per air strike
  K_total     total air-to-ground strikes
  uc          recovery factor: observed MoH ÷ true direct deaths

Strike-kill model (closed-form per cell, no Monte Carlo over strikes):

  weighted_pop_in_cell  Z_c = m_c·η_milt + N_c·f_M_c_civ·μ_M
                            + N_c·(f_W·1 + f_C·ε_C)
       where m_c = militant cell population (assumed all male/young-male)
             f_W = women fraction (~0.49)
             f_C = children-fraction (~0.47), of which boys are partly
                   in f_M_c_civ but we lump kids together.
  P(strike kills a militant       in cell c) = m_c·η_milt        / Z_c
  P(strike kills an adult-male civ in cell c) = (N_c·f_AM - m_c)·μ_M / Z_c
  P(strike kills a women+child civ in cell c) = N_c·(f_W + f_C·ε_C·something)/Z_c

  Expected total deaths in cell c = K_c · d_bar  where K_c ∝ targeting weight.
  Expected per-class deaths in cell c = (K_c · d_bar) · P(class | cell c).

We then sum over cells.  Likelihood:

  D_total      ≈ MoH_late2025 × uc       (Gaussian, σ ~ 5%)
  ω = (D_WC) / (D_total)  ≈ MoH_full_record OR OHCHR_sample
  ω_AM = D_AM / D_total                  (Beta(...|MoH n=70k))

Posterior over (q = D_militant / D_total).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RNG  = np.random.default_rng(20260516)
F    = json.loads((ROOT / "facts.json").read_text())


# ----------------------------------------------------------------------------
# Build cells from governorate-level data
# ----------------------------------------------------------------------------

GOV = F["geography"]["governorates"]
TOTAL_POP_2023 = F["population"]["total_2023"]["point"]
print("Governorates:")
for g in GOV:
    print(f"  {g['name']:14s} area={g['area_km2']:5.1f} km²   "
          f"pop={g['population_2023']:>7,d}")
print(f"  TOTAL pop 2023: {TOTAL_POP_2023:,d}")

CELLS_PER_GOV = 80                      # ⇒ 400 cells total
cells_pop = []
cells_gov = []
for gi, g in enumerate(GOV):
    pop_per_cell = g["population_2023"] / CELLS_PER_GOV
    for _ in range(CELLS_PER_GOV):
        cells_pop.append(pop_per_cell)
        cells_gov.append(gi)
N = np.asarray(cells_pop)               # shape (C,)
G = np.asarray(cells_gov)
C = len(N)
print(f"Cells: {C}  (pop sum = {N.sum():,.0f})")

# Demographics (Gaza-wide, applied uniformly per cell)
W_PLUS_C = F["population"]["share_women_plus_children"]["point"]   # 0.733
F_AM     = F["population"]["share_adult_males_18_60"]["point"]     # 0.255
F_W      = F["population"]["share_female_all_ages"]["low"]         # 0.49
F_U18    = F["population"]["share_under_18"]["low"]                # 0.465
# By definition: women+children = females + males<18 = F_W + (F_U18 * (1-F_W/0.5))
# but we just use the structural quantities below.

# Adult-male population per cell (fixed)
N_AM = N * F_AM
N_WC = N * W_PLUS_C                     # women + children pop per cell
assert np.allclose(N_AM + N_WC, N, rtol=2e-2), "demographic shares inconsistent"


# ----------------------------------------------------------------------------
# Observed targets to condition on
# ----------------------------------------------------------------------------

MOH_LATE2025 = F["deaths_observed"]["gaza_moh_total_through_late_2025"]["point"]
LANCET_LO    = F["deaths_observed"]["lancet_excess_total_estimate"]["low"]
LANCET_HI    = F["deaths_observed"]["lancet_excess_total_estimate"]["high"]

# Demographic shares of dead.  Two sources, with sample weights:
OHCHR = F["deaths_observed"]["ohchr_identification_sample"]
MOH_DEMO = F["deaths_observed"]["moh_demographic_breakdown_full_record"]

# Adult-male share among dead, observed:
omega_AM_ohchr = OHCHR["share_men_18_plus"]                # 0.307
omega_AM_moh   = MOH_DEMO["share_men_18_plus"]             # 0.44

# Use a likelihood that combines them: MoH (n≈70k) is the bigger sample,
# OHCHR (n=8.1k) is more identification-rigorous but biased toward
# easier-to-identify (more kids/women).  We use Beta posteriors.
N_OHCHR = OHCHR["n"]
N_MOH   = MOH_LATE2025                                     # treat MoH-record n ≈ all reported
print(f"\nObserved adult-male share among dead:")
print(f"  OHCHR (n={N_OHCHR}):       {omega_AM_ohchr:.3%}")
print(f"  MoH full record (n={N_MOH}): {omega_AM_moh:.3%}")


# ----------------------------------------------------------------------------
# Forward model — vectorized over posterior draws
# ----------------------------------------------------------------------------

def simulate_batch(M, gamma, sigma, alpha, beta, mu_M, eps_C,
                   d_bar, K_total, uc, latent_seed):
    """Run the spatial forward model for a *batch* of parameter sets.

    All inputs are arrays of length B (batch size).  We use one shared
    set of latent cell-level u_c values per batch element (so each draw
    has its own spatial militant pattern).

    Returns a dict of arrays of length B with expected deaths.
    """
    B = M.shape[0]
    rng = np.random.default_rng(latent_seed)

    # 1. Latent cell-level "militant hot-spot" factor u_c ~ N(0, σ²)
    #    drawn per-batch.  Shape (B, C).
    u = rng.standard_normal((B, C)) * sigma[:, None]
    # cell weight for militants: w_milt_c ∝ N_c · exp(γ·u_c)
    w_milt = N[None, :] * np.exp(gamma[:, None] * u)
    w_milt /= w_milt.sum(axis=1, keepdims=True)
    # m_c: militants per cell, scaled to M.
    m = M[:, None] * w_milt
    # Cap militants at adult-male population per cell (no cell can have
    # more militants than working-age men):
    m = np.minimum(m, N_AM[None, :] * 0.30)            # max 30% of AM are militants
    # Re-scale residual M back to total budget
    m_sum = m.sum(axis=1, keepdims=True)
    scale = (M[:, None] / np.maximum(m_sum, 1))
    m = m * scale

    # 2. Strike targeting weight per cell.
    rho_milt = m / N[None, :]                          # militant density
    w_strike = N[None, :] * np.power(1.0 + alpha[:, None] * rho_milt,
                                     beta[:, None])
    w_strike /= w_strike.sum(axis=1, keepdims=True)
    K_c = K_total[:, None] * w_strike                  # expected strikes per cell

    # 3. Per-strike kill class probabilities, per cell.
    # "Kill weight" per person:
    #   militant: 1.0 (counts only adult males who *are* militants)
    #   civilian adult male: μ_M
    #   civilian women+child: 1.0 (children are still substantial; we
    #     allow eps_C to slightly down-weight)
    # We approximate "women+children" as one bucket with mean exposure 1.
    # More precisely: w (women) + c (children) and we let kids be slightly
    # less exposed by eps_C, but kids are a big share so we keep eps_C in
    # [0.7, 1.0].
    civ_AM = np.maximum(N_AM[None, :] - m, 0.0)
    weight_milt = m * 1.0                              # shape (B, C)
    weight_civAM = civ_AM * mu_M[:, None]
    weight_WC   = N_WC[None, :] * eps_C[:, None]
    Z = weight_milt + weight_civAM + weight_WC

    # Per-cell expected deaths
    deaths_c = K_c * d_bar[:, None]
    P_milt   = weight_milt / Z
    P_civAM  = weight_civAM / Z
    P_WC     = weight_WC / Z

    D_milt  = (deaths_c * P_milt ).sum(axis=1)
    D_civAM = (deaths_c * P_civAM).sum(axis=1)
    D_WC    = (deaths_c * P_WC   ).sum(axis=1)
    D_total = D_milt + D_civAM + D_WC                  # true direct deaths
    D_obs   = D_total * uc                             # MoH-reported total

    return dict(D_milt=D_milt, D_civAM=D_civAM, D_WC=D_WC,
                D_total=D_total, D_obs=D_obs)


# ----------------------------------------------------------------------------
# Prior + likelihood
# ----------------------------------------------------------------------------

PRIOR = {
    "M":       (35_000, 60_000),                       # militants (uniform)
    "gamma":   (0.0,    1.5),                          # clustering
    "sigma":   (0.3,    1.2),                          # spatial SD
    "alpha":   (0.0,    8.0),                          # strike-targeting bias
    "beta":    (0.6,    1.6),                          # nonlinearity
    "mu_M":    (1.0,    2.5),                          # civ-male exposure
    "eps_C":   (0.7,    1.0),                          # child exposure
    "d_bar":   (1.2,    2.8),                          # killed per strike
    "K_total": (29_000, 40_000),                       # strikes
    "uc":      (1/1.7,  1/1.0),                        # MoH undercount fraction
                                                       # (true_dead = MoH/uc)
}


def sample_prior(B, rng):
    out = {}
    for k, (lo, hi) in PRIOR.items():
        out[k] = rng.uniform(lo, hi, size=B)
    return out


def log_likelihood(D_obs, D_milt, D_civAM, D_WC):
    """Likelihood of params given (D_obs ≈ MoH_late2025) and the
    adult-male share among dead, using both OHCHR (small but identified)
    and MoH full-record (large but cruder) shares."""
    # 1. Total observed dead — Gaussian on log scale, sd ≈ 7%
    log_lk_total = stats.norm.logpdf(np.log(D_obs),
                                     loc=np.log(MOH_LATE2025), scale=0.07)

    # 2. Adult-male share of dead
    sim_AM = D_civAM / np.maximum(D_obs, 1)
    sim_milt_AM = (D_civAM + D_milt) / np.maximum(D_obs, 1)
    # The data records adult-MALES (combatant or civilian) jointly;
    # OHCHR reports 30.7% men 18+, MoH full-record 44%.  These mix
    # combatants (all assumed adult-male) with adult-male civilians.
    # So compare sim_milt_AM to those observations.
    log_lk_ohchr = stats.beta.logpdf(np.clip(sim_milt_AM, 1e-3, 1-1e-3),
                                     OHCHR["share_men_18_plus"] * N_OHCHR + 1,
                                     (1 - OHCHR["share_men_18_plus"]) * N_OHCHR + 1)
    # MoH full-record carries more weight but with fatter-tail likelihood
    # because demographic categorization is cruder.  Down-weight n by 5x.
    n_eff_moh = N_MOH / 5.0
    log_lk_moh = stats.beta.logpdf(np.clip(sim_milt_AM, 1e-3, 1-1e-3),
                                   MOH_DEMO["share_men_18_plus"] * n_eff_moh + 1,
                                   (1 - MOH_DEMO["share_men_18_plus"]) * n_eff_moh + 1)
    # Geometric mean of the two demographic likelihoods (pretend they're
    # two independent observations of the same parameter, no correlation)
    log_lk_demo = 0.5 * (log_lk_ohchr + log_lk_moh)

    return log_lk_total + log_lk_demo


# ----------------------------------------------------------------------------
# Importance sampling
# ----------------------------------------------------------------------------

B = 200_000
print(f"\nSampling {B:,} prior draws...")
t0 = time.time()
rng = np.random.default_rng(42)
theta = sample_prior(B, rng)
sim = simulate_batch(latent_seed=43, **theta)
log_w = log_likelihood(sim["D_obs"], sim["D_milt"], sim["D_civAM"], sim["D_WC"])
log_w -= log_w.max()
w = np.exp(log_w)
ess = (w.sum() ** 2) / (w ** 2).sum()
print(f"  draws={B:,}, ESS={ess:,.0f} ({100*ess/B:.1f}%), "
      f"time={time.time()-t0:.1f}s")


def post_quantile(arr, q):
    s = np.argsort(arr)
    cw = np.cumsum(w[s])
    cw /= cw[-1]
    idx = np.searchsorted(cw, q)
    return arr[s[idx]]


def post_summary(arr, name):
    med = post_quantile(arr, 0.5)
    lo  = post_quantile(arr, 0.025)
    hi  = post_quantile(arr, 0.975)
    return {"name": name, "median": float(med), "lo": float(lo), "hi": float(hi)}


q_arr = sim["D_milt"] / np.maximum(sim["D_obs"], 1)
ratio = sim["D_civAM"] / np.maximum(sim["D_milt"], 1)
ratio_total = (sim["D_civAM"] + sim["D_WC"]) / np.maximum(sim["D_milt"], 1)

print()
print("=== Posterior summaries ===")
for name, arr in [
    ("M (total militants)",                   theta["M"]),
    ("gamma (clustering)",                    theta["gamma"]),
    ("sigma (spatial SD)",                    theta["sigma"]),
    ("alpha (targeting concentration)",       theta["alpha"]),
    ("mu_M (civilian-male exposure)",         theta["mu_M"]),
    ("d_bar (kills per strike)",              theta["d_bar"]),
    ("uc (MoH undercount)",                   1 / theta["uc"]),
    ("D_total (true direct dead)",            sim["D_total"]),
    ("D_obs   (MoH-reported)",                sim["D_obs"]),
    ("D_militants killed",                    sim["D_milt"]),
    ("D_civilian adult-male killed",          sim["D_civAM"]),
    ("D_women+children killed",               sim["D_WC"]),
    ("q = militant share of dead",            q_arr),
    ("civilian:militant ratio (total)",       ratio_total),
    ("civilian:militant ratio (excl. kids)",  ratio),
]:
    s = post_summary(arr, name)
    if "share" in name or "factor" in name:
        print(f"  {name:42s}  {s['median']:8.3f}  "
              f"[{s['lo']:7.3f}, {s['hi']:7.3f}]")
    else:
        print(f"  {name:42s}  {s['median']:>10,.0f}  "
              f"[{s['lo']:>10,.0f}, {s['hi']:>10,.0f}]")


# Save posterior arrays for the plotting/report step
out = ROOT / "posterior.npz"
np.savez_compressed(
    out,
    weights=w,
    **{k: v for k, v in theta.items()},
    D_total=sim["D_total"], D_obs=sim["D_obs"],
    D_milt=sim["D_milt"], D_civAM=sim["D_civAM"], D_WC=sim["D_WC"],
    q=q_arr, civ_milt_ratio=ratio_total,
)
print(f"\nSaved {out}")
