"""Validation and number-generation script for the manuscript.

Purpose
-------
Every quantitative claim in the paper (bounds, contradiction radii,
sensitivity grids, convergence table) is computed HERE, not typed by hand.
The script:

  1. runs normalization / consistency checks on the demographic inputs
     and the Gaza data file (shares sum to 1, totals consistent, CCR <-> q
     conversions invertible);
  2. implements the paper's formulas once, with unit tests against
     hand-derivable cases;
  3. reproduces every number quoted in Sections 3, 4, 6 and the
     abstract, asserting agreement within stated tolerance;
  4. emits LaTeX tables (sensitivity grid + convergence-of-methods) that
     the manuscript \\input{}s directly, plus a JSON report.

Exit code is nonzero if ANY check fails.  Run:

    python3 analysis/validate_bounds.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "paper" / "content"
FAILURES: list[str] = []


def check(name: str, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {got:.6g}, expected {expected:.6g} (tol {tol:g})")
    if not ok:
        FAILURES.append(name)
    return ok


def check_true(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{': ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)
    return cond


# ===========================================================================
# Core formulas (single source of truth, mirrors the paper's equations)
# ===========================================================================

def q_of_mu(omega: float, w: float, a: float, f: float, mu: float) -> float:
    """Theorem 'Sharp identified set', eq:qmu:  q(mu) = 1 - omega*(w + mu*(a-f))/w.

    Decreasing in mu.  Valid when combatants are inside the AM class and
    f <= a.  NOT clipped to [0,1]; caller clips.
    """
    return 1.0 - omega * (w + mu * (a - f)) / w


def identified_set(omega, w, a, f, mu_lo, mu_hi, M=None, D=None):
    """[q(mu_hi), q(mu_lo)] ∩ [0,1] ∩ [0, M/D].  Returns None when the
    intersection is empty (jointly infeasible inputs), rather than a
    degenerate pseudo-interval."""
    lo = max(0.0, q_of_mu(omega, w, a, f, mu_hi))
    hi = min(1.0, q_of_mu(omega, w, a, f, mu_lo))
    if M is not None and D is not None:
        hi = min(hi, M / D)
    if hi < lo - 1e-12:
        return None
    return (lo, max(lo, hi))


def omega_needed(q: float, w: float, a: float, f: float, mu: float) -> float:
    """Invert eq:qmu for omega: the demographic composition a claim requires."""
    return (1.0 - q) * w / (w + mu * (a - f))


def mu_needed(q: float, omega: float, w: float, a: float, f: float) -> float:
    """Invert eq:qmu for mu: the exposure ratio a claim requires.

    q = 1 - omega*(w + mu*(a-f))/w  =>  mu = [w*((1-q)/omega - 1)] / (a-f).
    Values < 1 mean the claim is infeasible for any physical exposure ratio
    (civilian men would have to be LESS exposed than women and children).
    """
    return w * ((1.0 - q) / omega - 1.0) / (a - f)


def ccr_of_q(q: float) -> float:
    return (1.0 - q) / q


def q_of_ccr(ccr: float) -> float:
    return 1.0 / (1.0 + ccr)


def rho_omega(q_claim, omega_hat, sigma_omega, w, a, f, mu_lo, mu_hi):
    """Contradiction radius on the omega axis, per Definition 'radius':
    the claim may pick the MOST favourable mu in [mu_lo, mu_hi], so the
    feasible omega interval is [omega_needed(mu_hi), omega_needed(mu_lo)]
    (omega_needed is decreasing in mu).  rho is the standardized distance
    from omega_hat to that interval (0 if inside).

    NOTE: the most-favourable-mu grant is part of Definition 4.2; evaluating
    omega_needed at a single mu endpoint would overstate rho for high-q
    claims.
    """
    om_hi = omega_needed(q_claim, w, a, f, mu_lo)   # largest feasible omega
    om_lo = omega_needed(q_claim, w, a, f, mu_hi)   # smallest feasible omega
    if om_lo <= omega_hat <= om_hi:
        return 0.0
    return min(abs(omega_hat - om_hi), abs(omega_hat - om_lo)) / sigma_omega


def z_uniformity(omega_hat, w, n):
    """Discrepancy of omega_hat from w in binomial-SE units (descriptive:
    the OHCHR sample is verified/selected, not a probability sample)."""
    se = math.sqrt(w * (1 - w) / n)
    return (omega_hat - w) / se


def norm_sf(z: float) -> float:
    """One-sided upper tail of the standard normal (no scipy dependency)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ===========================================================================
# 0. Unit tests of the formulas against hand-derivable cases
# ===========================================================================

print("== Unit tests ==")
# If omega == w and mu=1, a=1-w, f=0: q = 1 - w*(w+(1-w))/w = 0 exactly.
check("q=0 when dead mirror population (mu=1,f=0)",
      q_of_mu(omega=0.7, w=0.7, a=0.3, f=0.0, mu=1.0), 0.0)
# If omega = 0 (no women/children among dead), q(1) = 1.
check("q=1 when no W deaths", q_of_mu(0.0, 0.7, 0.3, 0.0, 1.0), 1.0)
# Round trips.
_q = q_of_mu(0.56, 0.733, 0.267, 0.02, 1.7)
check("omega_needed round trip", omega_needed(_q, 0.733, 0.267, 0.02, 1.7), 0.56, 1e-12)
check("mu_needed round trip", mu_needed(_q, 0.56, 0.733, 0.267, 0.02), 1.7, 1e-12)
check("ccr/q round trip", q_of_ccr(ccr_of_q(0.025)), 0.025, 1e-12)
check("ccr(q=0.5)=1", ccr_of_q(0.5), 1.0)

# ===========================================================================
# 1. Inputs (with provenance) and normalization checks
# ===========================================================================

print("\n== Normalization and input consistency ==")

# Demographic classes.  Convention (matching the paper, Table 1): AM =
# working-age males 18-59, matching the MoH record's own category boundary
# (its "elderly" category is defined as 60+); W = everyone else (females of
# all ages + boys under 18 + men 60+), so the classes partition the
# population.  PCBS males 18+ = 0.267; elderly 60+ ~ 5.0% of population
# with a roughly even sex split -> males 60+ ~ 0.022, so a = 0.245 and
# w = 1 - a = 0.755.
W_POP = 0.755            # everyone outside the working-age-male class
A_POP = 1.0 - W_POP      # males 18-59 (0.245)
F_STOCK = 0.020          # combatant stock / population (IISS/RUSI mid)
A_MINUS_F = A_POP - F_STOCK

check("classes sum to 1", W_POP + A_POP, 1.0, 1e-12)
check_true("f <= a", F_STOCK <= A_POP, f"f={F_STOCK}, a={A_POP}")

# gaza_sim/facts.json carries share_adult_males_18_60 = [0.245, 0.27]
# (point 0.255).  a = 0.245 is the value consistent with males 18+ = 0.267
# and an elderly (60+) population share of ~5% with an even-to-slightly-
# female sex split; the field's 0.255 point would require the elderly to
# be ~76% female and is carried as a sensitivity, not the anchor.
A_SENS_18_60 = 0.255

# omega anchors: share of the DEAD outside the working-age-male class.
# Primary: the MoH four-category breakdown as published by OCHA (Reported
# Impact Snapshot, 20 May 2026) for the 71,444 identified fatalities as of
# 31 Dec 2025 -- natively at the paper's D ~ 70k window:
#   men 34,078 / children 21,283 / women 10,983 / elderly 5,100.
# omega = 1 - men/identified = 1 - 34,078/71,444 = 0.523.  The MoH "men"
# category is 18-59 (elderly = 60+), so this is the working-age complement
# with no re-splitting of any sex-mixed bucket.
OCHA_MEN, OCHA_CHILDREN, OCHA_WOMEN, OCHA_ELDERLY = 34_078, 21_283, 10_983, 5_100
N_IDENTIFIED = 71_444
check("OCHA four-category counts sum to the identified total",
      OCHA_MEN + OCHA_CHILDREN + OCHA_WOMEN + OCHA_ELDERLY, N_IDENTIFIED, 0)
OMEGA_MOH = round(1.0 - OCHA_MEN / N_IDENTIFIED, 3)
check("omega reconstructed from published category counts",
      OMEGA_MOH, 0.523, 5e-4)
OMEGA_OHCHR = 0.693      # OHCHR verified sample (residential-strike stratum).
                         # OHCHR publishes no elderly split, so under the
                         # working-age convention this anchor is approximate
                         # (a slight understatement of omega' -- direction
                         # known, so every rejection on it holds a fortiori).
OMEGA_BLEND = round(math.sqrt(OMEGA_MOH * OMEGA_OHCHR), 3)  # geometric blend
SIGMA_OMEGA = 0.01       # conservative SE (systematic; binomial SE is smaller)
N_OHCHR = 8119
N_MOH = N_IDENTIFIED

# Option-A convention sensitivity: males 18+ (a = 0.267, w = 0.733) on the
# SAME Dec-2025 record (vintage-matched, so the comparison isolates the
# class boundary).  The record's elderly category is sex-mixed, so this
# convention requires an imputed elderly sex split; we use the documented
# 61.7%-male split (Epstein / Washington Institute):
#   omega_A = (children + women + 0.383*elderly) / identified = 0.479.
# The class boundary is NOT immaterial: q(1) moves 32.1% -> 36.0% and the
# 25k endpoint's required mu moves from 0.77 to ~1.06 (arithmetic edge).
# The calibrated rejection holds under either convention (q(2) = 19.8%
# under men-18+, still below the claim's lower endpoint).
ELDERLY_MALE_SHARE = 0.617
OMEGA_A_CONV = round((OCHA_CHILDREN + OCHA_WOMEN
                      + (1 - ELDERLY_MALE_SHARE) * OCHA_ELDERLY)
                     / N_IDENTIFIED, 3)
W_A_CONV = 0.733
check("men-18+ convention omega on the Dec-2025 record ~ 0.479",
      OMEGA_A_CONV, 0.479, 5e-4)

# binomial SEs are below the conservative sigma
se_moh = math.sqrt(OMEGA_MOH * (1 - OMEGA_MOH) / N_IDENTIFIED)
se_ohchr = math.sqrt(OMEGA_OHCHR * (1 - OMEGA_OHCHR) / N_OHCHR)
check_true("sigma_omega conservative vs binomial",
           SIGMA_OMEGA >= max(se_moh, se_ohchr),
           f"binomial SEs: MoH {se_moh:.4f}, OHCHR {se_ohchr:.4f}")

# Totals / claims.  The anchor-window denominator is the record's OWN
# identified count (71,444 as of 31 Dec 2025): omega is measured on that
# count, so every claim conversion at the anchor window uses the same
# denominator (an identified count cannot exceed the total it is drawn
# from, so D = 70,000 alongside n = 71,444 would be incoherent).  The
# round "~70k" appears in prose only as an informal description of scale,
# never in arithmetic.
D_MOH = N_IDENTIFIED     # 71,444
# GMS survey (Lancet Glob Health 2026): survey estimate 75,200 violent
# deaths vs 49,090 recorded by MoH over the same window, i.e. the MoH
# record captures ~65% of the survey estimate (a ~35% shortfall relative
# to the survey), so the correction multiplies the recorded toll by
# 75,200/49,090 ~= 1.53.
GMS_SURVEY_EST = 75_200
GMS_MOH_SAME_WINDOW = 49_090
GMS_UNDERCOUNT = GMS_SURVEY_EST / GMS_MOH_SAME_WINDOW       # ~1.532
D_CORRECTED = round(D_MOH * GMS_UNDERCOUNT / 1000) * 1000   # ~109k
MISSING = 12_200         # missing persons, presumed under rubble
# D_MAX treats the missing as additional to the survey correction; this
# assumes the survey estimate does not already include them, which is why
# both corrected scenarios (D_CORRECTED and D_MAX) are reported.
D_MAX = D_CORRECTED + MISSING                               # ~121k
M_STOCK = 45_000         # IISS/RUSI/press-reported pre-war Hamas+PIJ mid
IDF_LO, IDF_HI = 17_000, 25_000

# Named-combatant evidence (Aman internal database, Guardian/+972 Aug 2025)
AMAN_NAMED = 8_900       # named Hamas/PIJ dead or probably dead, May 2025
AMAN_D_AT_TIME = 53_000  # MoH toll at the same date

# Gaza data file consistency (per-war JSON totals vs sides)
gaza = json.loads((ROOT / "data" / "per_war" / "israel_gaza_war_2023.json").read_text())
tot = gaza["totals"]
check("gaza grand_low = mil+civ low", tot["grand_low"],
      tot["military_low"] + tot["civilian_low"], 0)
check("gaza grand_high = mil+civ high", tot["grand_high"],
      tot["military_high"] + tot["civilian_high"], 0)

# ===========================================================================
# 2. Reproduce the paper's quoted numbers (regression tests on prose)
# ===========================================================================

print("\n== Prose regression checks ==")

# Uniformity z-test (Section 6, Step 1; OHCHR sample, ONE-SIDED p:
# the alternative is omega < w, i.e. adult males over-represented among
# dead).  The OHCHR omega is carried at the men-18+ value (no elderly
# split is published), so the comparison is made at the MATCHED men-18+
# population share w = 0.733, not the working-age 0.755 -- differencing
# an 18+ omega against an 18-59 w would inflate the discrepancy.
z = z_uniformity(OMEGA_OHCHR, W_A_CONV, N_OHCHR)
p_one = norm_sf(-z)          # z is negative; upper tail of |z|
check("uniformity z (OHCHR, matched men-18+ convention)", z, -8.1, 0.1)
check_true("uniformity p astronomically small (one-sided)", p_one < 1e-12,
           f"p={p_one:.3g}")

# Blended-anchor mu=1 endpoint (Table 3 sensitivity row: q(1) ~ 22%)
q1_blend = q_of_mu(OMEGA_BLEND, W_POP, A_POP, F_STOCK, 1.0)
check("q(1) blend ~ 0.22", q1_blend, 0.219, 0.005)

# IDF claim as q over the identified anchor-window denominator
q_idf_lo, q_idf_hi = IDF_LO / D_MOH, IDF_HI / D_MOH
check("q_IDF low  ~ 23.8%", q_idf_lo, 0.2379, 0.001)
check("q_IDF high ~ 35.0%", q_idf_hi, 0.3499, 0.001)
print(f"  [INFO] IDF claim implies CCR "
      f"{ccr_of_q(q_idf_hi):.1f}:1 - {ccr_of_q(q_idf_lo):.1f}:1")

# ===========================================================================
# 3. Headline numbers (MoH anchor primary)
# ===========================================================================

print("\n== Headline computations (MoH anchor) ==")

# Identified set on the MoH anchor at benchmark mu ranges
set_moh_1_25 = identified_set(OMEGA_MOH, W_POP, A_POP, F_STOCK, 1.0, 2.5,
                              M_STOCK, D_MOH)
set_moh_2_35 = identified_set(OMEGA_MOH, W_POP, A_POP, F_STOCK, 2.0, 3.5,
                              M_STOCK, D_MOH)
q1_moh = q_of_mu(OMEGA_MOH, W_POP, A_POP, F_STOCK, 1.0)
print(f"  MoH anchor, mu in [1, 2.5]: q in [{set_moh_1_25[0]:.3f}, {set_moh_1_25[1]:.3f}]")
print(f"  MoH anchor, mu in [2, 3.5]: q in [{set_moh_2_35[0]:.3f}, {set_moh_2_35[1]:.3f}]")
check("q(1) MoH ~ 32.1%", q1_moh, 0.3211, 0.002)
q2_moh = q_of_mu(OMEGA_MOH, W_POP, A_POP, F_STOCK, 2.0)
check("q(2) MoH ~ 16.5%", q2_moh, 0.1653, 0.002)
check("calibrated ceiling CCR ~ 5.05 (quoted as 5)",
      ccr_of_q(q2_moh), 5.05, 0.1)

# Convention dependence (vintage-matched, Dec-2025 record): the men-18+
# convention moves the mu=1 upper endpoint moderately (32.1% -> 36.0%) and
# puts the 25k endpoint's required mu at the arithmetic edge (~1.06);
# the calibrated rejection holds under either convention.
q1_a_conv = q_of_mu(OMEGA_A_CONV, W_A_CONV, 1 - W_A_CONV, F_STOCK, 1.0)
q2_a_conv = q_of_mu(OMEGA_A_CONV, W_A_CONV, 1 - W_A_CONV, F_STOCK, 2.0)
check("q(1) under the men-18+ convention ~ 36.0%", q1_a_conv, 0.360, 0.005)
check("q(2) under the men-18+ convention ~ 19.8%", q2_a_conv, 0.198, 0.005)

# mu required to rationalise each IDF endpoint on the MoH anchor
mu_req_lo = mu_needed(q_idf_lo, OMEGA_MOH, W_POP, A_POP, F_STOCK)
mu_req_hi = mu_needed(q_idf_hi, OMEGA_MOH, W_POP, A_POP, F_STOCK)
check("mu required for IDF 17k ~ 1.53", mu_req_lo, 1.534, 0.01)
check_true("IDF 25k infeasible for any mu >= 1 (working-age convention)",
           mu_req_hi < 1.0, f"mu_needed={mu_req_hi:.3f} < 1")
# The same endpoint under the men-18+ convention: at the arithmetic edge.
mu_req_hi_a = mu_needed(q_idf_hi, OMEGA_A_CONV, W_A_CONV, 1 - W_A_CONV,
                        F_STOCK)
check("IDF 25k under men-18+ convention needs mu ~ 1.06 (arithmetic edge)",
      mu_req_hi_a, 1.06, 0.01)
check_true("calibrated rejection is convention-robust (q2 < claim lower"
           " endpoint under both conventions)",
           max(q2_moh, q2_a_conv) < q_idf_lo,
           f"max({q2_moh:.3f}, {q2_a_conv:.3f}) < {q_idf_lo:.3f}")

# Contradiction radii per the paper's Definition (claim picks the most
# favourable mu in the range).  Two exposure regimes:
#   agnostic   mu in [1, 2.5]  (no calibration assumed)
#   calibrated mu in [2, 3.5]  (Frost/B'tselem historical calibration)
# The agnostic regime is quoted in prose as "any mu >= 1"; the 2.5 upper
# grid endpoint is non-binding for every claim scored here, because both
# tested claims sit above the set and the most favourable mu is mu_lo = 1
# (q(mu) is decreasing), so widening the grid upward cannot change rho.
print("  rho_omega, agnostic exposure mu in [1,2.5]:")
rho_ag = {}
for lbl, om in [("MoH", OMEGA_MOH), ("blend", OMEGA_BLEND), ("OHCHR", OMEGA_OHCHR)]:
    r17 = rho_omega(q_idf_lo, om, SIGMA_OMEGA, W_POP, A_POP, F_STOCK, 1.0, 2.5)
    r25 = rho_omega(q_idf_hi, om, SIGMA_OMEGA, W_POP, A_POP, F_STOCK, 1.0, 2.5)
    rho_ag[lbl] = (r17, r25)
    print(f"    {lbl:6s} anchor: IDF 17k -> {r17:5.1f} SE, IDF 25k -> {r25:5.1f} SE")
print("  rho_omega, calibrated exposure mu in [2,3.5]:")
rho_cal = {}
for lbl, om in [("MoH", OMEGA_MOH), ("blend", OMEGA_BLEND), ("OHCHR", OMEGA_OHCHR)]:
    r17 = rho_omega(q_idf_lo, om, SIGMA_OMEGA, W_POP, A_POP, F_STOCK, 2.0, 3.5)
    r25 = rho_omega(q_idf_hi, om, SIGMA_OMEGA, W_POP, A_POP, F_STOCK, 2.0, 3.5)
    rho_cal[lbl] = (r17, r25)
    print(f"    {lbl:6s} anchor: IDF 17k -> {r17:5.1f} SE, IDF 25k -> {r25:5.1f} SE")

# Regression checks on the numbers quoted in Section 6
check("rho MoH/17k agnostic = 0 (feasible at mu~1.53)", rho_ag["MoH"][0], 0.0)
check("rho MoH/25k agnostic ~ 2.2 SE", rho_ag["MoH"][1], 2.22, 0.1)
check("rho MoH/17k calibrated ~ 4.6 SE", rho_cal["MoH"][0], 4.55, 0.1)
check("rho MoH/25k calibrated ~ 11.6 SE", rho_cal["MoH"][1], 11.57, 0.15)
check("rho OHCHR/25k agnostic ~ 19 SE", rho_ag["OHCHR"][1], 19.2, 0.2)

# --- The other two axes of Definition 4.2 (single-input rescues, mu = 1
# granted, accounting identities a' = 1 - w' and f' = M'/N respected) ---
N_POP_FOR_AXES = 2_250_000
# Census axis: q = 1 - omega*(1-f)/w'  =>  w' = omega*(1-f)/(1-q).
w_rescue = OMEGA_MOH * (1 - F_STOCK) / (1 - q_idf_hi)
check("w-axis rescue of 25k endpoint needs w' ~ 0.788", w_rescue, 0.7884, 0.002)
check_true("w-axis rescue is a >= 3 pp census error",
           w_rescue - W_POP >= 0.03, f"{w_rescue - W_POP:.3f}")
# Manpower axis: omega = (1-q)w/(w + a - f')  =>  f' = 1 - (1-q)w/omega.
f_rescue = 1 - (1 - q_idf_hi) * W_POP / OMEGA_MOH
M_rescue = f_rescue * N_POP_FOR_AXES
check("M-axis rescue of 25k endpoint needs M' ~ 138k", M_rescue, 138_500, 2_000)
check_true("M-axis rescue is ~3.1x the institutional stock estimate",
           2.8 < M_rescue / M_STOCK < 3.4, f"{M_rescue/M_STOCK:.1f}x")

# Standardised per-axis radii (Definition 'radius'): sigma_w = 0.005 is the
# census scale stated in Section 6; sigma_M = 12,500 is the half-width of
# the institutional pre-war-strength range [35k, 60k].
SIGMA_W = 0.005
SIGMA_M = 12_500
rho_w_25k = (w_rescue - W_POP) / SIGMA_W
rho_M_25k = (M_rescue - M_STOCK) / SIGMA_M
check("rho_w of 25k endpoint ~ 6.7 sigma_w", rho_w_25k, 6.69, 0.3)
check("rho_M of 25k endpoint ~ 7.5 sigma_M", rho_M_25k, 7.48, 0.3)
check_true("rho = min over axes is attained on the omega axis",
           rho_ag["MoH"][1] < min(rho_w_25k, rho_M_25k),
           f"{rho_ag['MoH'][1]:.1f} < min({rho_w_25k:.1f}, {rho_M_25k:.1f})")

# --- Joint (coordinated) radius illustration for Remark 'joint':
# simultaneous moves of omega and w (a'=1-w'), sigma_w = 0.005, mu = 1.
# Constraint: omega' = (1-q) w'/(1-f).  Minimise the max standardised move.
SIGMA_W_ILLUST = 0.005
coef = (1 - q_idf_hi) / (1 - F_STOCK)
# Equalise the two standardised displacements:
w_joint = (OMEGA_MOH + (SIGMA_OMEGA / SIGMA_W_ILLUST) * W_POP) / \
          (coef + SIGMA_OMEGA / SIGMA_W_ILLUST)
rho_joint = (w_joint - W_POP) / SIGMA_W_ILLUST
check("joint (omega,w) radius of 25k endpoint ~ 1.7", rho_joint, 1.67, 0.1)
check_true("joint radius below single-axis rho_omega = 2.2",
           rho_joint < rho_ag["MoH"][1], f"{rho_joint:.2f} < {rho_ag['MoH'][1]:.2f}")

# --- f-sensitivity of the mu=1 endpoint (Section 6, Step 2 footnote) ---
q1_f_lo = q_of_mu(OMEGA_MOH, W_POP, A_POP, 0.0157, 1.0)
q1_f_hi = q_of_mu(OMEGA_MOH, W_POP, A_POP, 0.0269, 1.0)
check("q(1) at f=0.0157 ~ 31.8%", q1_f_lo, 0.3182, 0.001)
check("q(1) at f=0.0269 ~ 32.6%", q1_f_hi, 0.3259, 0.001)
check("17k claim kills 28% of the 60k top-of-range stock",
      17_000 / 60_000, 0.2833, 0.001)

# --- identified_set empty-set semantics (unit test) ---
check_true("OHCHR anchor at calibrated exposure -> empty set (None)",
           identified_set(OMEGA_OHCHR, W_POP, A_POP, F_STOCK, 2.0, 3.5) is None,
           "empty intersection correctly reported")

# Symmetric test: q = 0.  On the demographic axis alone, q=0 needs
# mu = mu_star; the honest statement is that q=0 is instead rejected by
# documented named combatant deaths (D_M >= names > 0).
mu_star_q0 = mu_needed(0.0, OMEGA_MOH, W_POP, A_POP, F_STOCK)
check("q=0 on MoH anchor requires mu ~ 3.06", mu_star_q0, 3.06, 0.01)
print(f"  q=0 on MoH anchor requires mu = {mu_star_q0:.2f} "
      f"(inside plausible range -> demographic axis alone cannot reject q=0;"
      f" rejection of q=0 comes from named combatant dead)")

# Aman database consistency with the mu=1 bound
q_aman = AMAN_NAMED / AMAN_D_AT_TIME
check("Aman DB q ~ 16.8%", q_aman, 0.168, 0.002)
check_true("Aman q inside [0, q(1)] on MoH anchor", 0 <= q_aman <= q1_moh,
           f"{q_aman:.3f} <= {q1_moh:.3f}")

# Manpower ledger: the IDF's own stock figures (paper, Section 6).
# Pre-war official estimate 30,000 (Jerusalem Post, first official figure);
# post-ceasefire assessment: >22,000 killed AND ~20,000 fielded today (ITIC,
# citing Times of Israel Oct 21, 2025).  Jointly consistent only with
# killed + remaining - pre_war >= 12,000 recruited during the war.
IDF_PREWAR_STOCK = 30_000
IDF_CLAIM_KILLED = 22_000
IDF_REMAINING = 20_000
recruits_implied = IDF_CLAIM_KILLED + IDF_REMAINING - IDF_PREWAR_STOCK
check("ledger: implied recruits = 12,000", recruits_implied, 12_000, 0)
check_true("ledger consistent only via recruitment concession",
           recruits_implied > 0,
           f"killed+remaining exceeds pre-war stock by {recruits_implied:,}")

# ===========================================================================
# 3b. The adult-male death budget (three-component decomposition, Fig. budget)
# ===========================================================================
# The demographic record fixes D_WC = omega*D and D_men = (1-omega)*D.
# Every hypothesis about combatant deaths D_C allocates the men's budget:
# D_C combatants, D_men - D_C civilian men.  Per-capita rates then follow
# from the population decomposition N = N_WC + N_AMciv + M.
# Stock convention: M = 45,000 (IISS/RUSI central, f = M/N = 0.020 exactly,
# N = 2.25M), the choice most favourable to the claim (a larger stock lowers
# the implied killed fraction and selectivity).  The IDF's own 30,000 stock
# is reported alongside for the claim rows (the manpower ledger).

print("\n== Adult-male death budget ==")
# N is rounded to 2.25M so that f = M/N = 45k/2.25M = 0.020 holds exactly;
# the sourced PCBS total is 2,226,544 (gaza_sim/facts.json), a <1.1%
# difference that moves no derived figure by more than 1%.
N_POP = 2_250_000
check("f consistency: M/N = 0.020", M_STOCK / N_POP, F_STOCK, 1e-12)
N_WC = W_POP * N_POP                   # 1,698,750 outside the AM class
N_AM = A_POP * N_POP                   # 551,250 working-age men (18-59)
N_AMCIV = N_AM - M_STOCK               # 506,250 civilian working-age men

# The budget uses the record's own published category counts (native,
# not omega-rounded): men 34,078; everyone else 37,366.
D_WC = N_IDENTIFIED - OCHA_MEN         # 37,366 W-class deaths
D_MEN = OCHA_MEN                       # 34,078 working-age men among dead
check("budget: W-class dead = 37,366", D_WC, 37_366, 0)
check("budget: working-age-male dead = 34,078", D_MEN, 34_078, 0)
check("budget: native counts consistent with rounded omega",
      D_WC / D_MOH, OMEGA_MOH, 1e-3)
# The caption's uncertainty note: sigma_omega shifts the men's budget by ~700.
check("budget: sigma_omega shifts budget by ~700", SIGMA_OMEGA * D_MOH, 714, 1)

RATE_WC = D_WC / N_WC                  # 2.20%, fixed by the record
q_cal_hi = q_of_mu(OMEGA_MOH, W_POP, A_POP, F_STOCK, 2.0)   # 16.5%

def budget_row(label, d_c, stock=M_STOCK, split_wc=None):
    """Decompose the toll and compute per-capita rates for one hypothesis."""
    d_wc = split_wc if split_wc is not None else D_WC
    d_men = D_MOH - d_wc
    d_amciv = d_men - d_c
    rate_c = d_c / stock
    rate_amciv = d_amciv / N_AMCIV if stock == M_STOCK else float("nan")
    mu_implied = rate_amciv / (d_wc / N_WC) if stock == M_STOCK else float("nan")
    return {
        "label": label, "d_combatant": d_c, "d_amciv": d_amciv, "d_wc": d_wc,
        "stock": stock, "rate_combatant": rate_c, "rate_amciv": rate_amciv,
        "rate_wc": d_wc / N_WC, "mu_implied": mu_implied,
        "selectivity": rate_c / rate_amciv if rate_amciv else float("inf"),
    }

# Zero-targeting benchmark: a uniform draw from the population kills every
# class at rate D/N, so combatants die at their population share.  It also
# predicts omega = w = 75.5%, which the record rejects (Step 1); the row is
# the zero-care reference, not an allocation of the observed record.
row_random = budget_row("random violence", F_STOCK * D_MOH,
                        split_wc=W_POP * D_MOH)
rows_budget = [
    row_random,
    budget_row("calibrated upper end (mu=2)", q_cal_hi * D_MOH),
    budget_row("equal exposure (mu=1)", q1_moh * D_MOH),
    budget_row("IDF claim, 17k", IDF_LO),
    budget_row("IDF claim, 25k", IDF_HI),
    budget_row("all working-age men counted as combatants", D_MEN, stock=N_AM),
]
for r in rows_budget:
    print(f"  {r['label']:38s} D_C={r['d_combatant']:>7,.0f}  "
          f"rate_C={r['rate_combatant']:6.1%}  rate_AMciv={r['rate_amciv']:6.2%}  "
          f"mu={r['mu_implied']:.2f}  sel={r['selectivity']:.1f}x")

# Checks: the derived rates reproduce the paper's mu-required numbers.
check("budget: random q = 2.0%", row_random["d_combatant"] / D_MOH, 0.020, 1e-9)
check_true("random q inside calibrated set",
           row_random["d_combatant"] / D_MOH <= q_cal_hi,
           f"2.0% <= {q_cal_hi:.1%}")
check("budget: mu=1 row implies mu = 1.00 (native counts vs rounded omega)",
      rows_budget[2]["mu_implied"], 1.0, 0.01)
check("budget: IDF 17k implies mu ~ 1.53", rows_budget[3]["mu_implied"],
      mu_req_lo, 0.01)
check("budget: IDF 25k implies mu ~ 0.82", rows_budget[4]["mu_implied"],
      mu_req_hi, 0.01)
check("budget: IDF 17k kills 38% of institutional stock",
      rows_budget[3]["rate_combatant"], 17_000 / 45_000, 1e-9)
check("budget: IDF 25k kills 56% of institutional stock",
      rows_budget[4]["rate_combatant"], 25_000 / 45_000, 1e-9)
check("budget: IDF 17k selectivity ~ 11x", rows_budget[3]["selectivity"], 11.2, 0.2)
check("budget: IDF 25k selectivity ~ 31x", rows_budget[4]["selectivity"], 31.0, 0.5)
check("budget: IDF own stock -> 57-83% killed",
      IDF_LO / IDF_PREWAR_STOCK, 0.567, 0.001)
check("budget: all-men ceiling q = 47.7%", D_MEN / D_MOH, 0.477, 5e-4)
check_true("all-men stock (551k) contradicts claimed stock 18x over",
           N_AM / IDF_PREWAR_STOCK > 18, f"{N_AM/IDF_PREWAR_STOCK:.1f}x")
check_true("claim < all-men budget: concedes 9.1-17.1k dead men civilian",
           IDF_HI < D_MEN, f"{IDF_HI:,} < {D_MEN:,.0f}")

# GMS D-sensitivity: what the IDF claim implies as D grows.
# Caveat carried into the paper: applying the MoH omega to corrected D
# assumes the unrecorded dead share the recorded demographic mix, which
# GMS supports (their demographic breakdown matches the MoH record).
print("\n== D-sensitivity ==")
D_GRID = [("MoH confirmed", D_MOH),
          ("GMS undercount-corrected", D_CORRECTED),
          ("corrected + missing", D_MAX)]
d_rows = []
for label, D in D_GRID:
    ql, qh = IDF_LO / D, IDF_HI / D
    mul_ = mu_needed(ql, OMEGA_MOH, W_POP, A_POP, F_STOCK)
    muh_ = mu_needed(qh, OMEGA_MOH, W_POP, A_POP, F_STOCK)
    d_rows.append((label, D, ql, qh, mul_, muh_))
    print(f"  D={D:>7,d} ({label:25s}): q_IDF {ql:.1%}-{qh:.1%}; "
          f"mu needed {mul_:.2f} / {muh_:.2f}")
# The 25k endpoint stays below the calibrated range mu in [2, 3.5] at every
# supported denominator; the 17k endpoint's required mu enters the calibrated
# range once the GMS correction is applied (mu ~ 2.06 at D~109k), so the
# demographic axis alone no longer rejects 17k at corrected denominators --
# at the price of 84-104k implied civilian deaths on the claim's own ledger.
check_true("IDF 25k needs mu < 2 (below calibrated) even at D~121k",
           d_rows[-1][5] < 2.0, f"mu={d_rows[-1][5]:.3f}")
check("IDF 17k at D~109k needs mu ~ 2.06 (enters calibrated range)",
      d_rows[1][4], 2.06, 0.01)
check("IDF 17k at D~121k needs mu ~ 2.16", d_rows[-1][4], 2.16, 0.01)

# Force math on the civilian ledger: the claimed combatant count is fixed,
# so the entire undercount correction lands on civilians (paper, Step 3).
print("  civilian ledger under the fixed claim:")
for label, DD in D_GRID:
    civ_lo, civ_hi = DD - IDF_HI, DD - IDF_LO
    ccr_lo, ccr_hi = civ_lo / IDF_HI, civ_hi / IDF_LO
    print(f"    D={DD:>7,d}: implied civilians {civ_lo:,.0f}-{civ_hi:,.0f} "
          f"(CCR {ccr_lo:.1f}:1-{ccr_hi:.1f}:1)")
check("ledger: civilians at D=71,444, 25k claim ~ 46k", D_MOH - IDF_HI, 46_444, 0)
check("ledger: civilians at D=71,444, 17k claim ~ 54k", D_MOH - IDF_LO, 54_444, 0)
check("ledger: civilians at D=109k, 25k claim = 84k", D_CORRECTED - IDF_HI, 84_000, 0)
check("ledger: civilians at D=109k, 17k claim = 92k", D_CORRECTED - IDF_LO, 92_000, 0)
check("ledger: civilians at D=121.2k = 96.2-104.2k", D_MAX - IDF_HI, 96_200, 0)
check("ledger: CCR at D=109k, 25k claim ~ 3.4:1",
      (D_CORRECTED - IDF_HI) / IDF_HI, 3.36, 0.05)
check("ledger: CCR at D=109k, 17k claim ~ 5.4:1",
      (D_CORRECTED - IDF_LO) / IDF_LO, 5.41, 0.05)
check("ledger: CCR at D=121.2k, 25k claim ~ 3.8:1",
      (D_MAX - IDF_HI) / IDF_HI, 3.85, 0.05)
check("ledger: CCR at D=121.2k, 17k claim ~ 6.1:1",
      (D_MAX - IDF_LO) / IDF_LO, 6.13, 0.05)
# The combatant side cannot absorb the correction: 25k + 37k exceeds even
# the 60k top of the institutional pre-war-strength range.
check_true("absorbing the correction would exceed the full strength range",
           IDF_HI + (D_CORRECTED - D_MOH) > 60_000,
           f"{IDF_HI + D_CORRECTED - D_MOH:,} > 60,000")

# ===========================================================================
# 4. Sensitivity grid (omega anchor x mu) -> LaTeX table
# ===========================================================================

print("\n== Sensitivity grid ==")
ANCHORS = [("MoH (four-category record, Dec 2025)", OMEGA_MOH, W_POP),
           ("Blend (geometric)", OMEGA_BLEND, W_POP),
           ("OHCHR (residential-strike stratum)", OMEGA_OHCHR, W_POP),
           ("MoH, men-18$+$ convention (Dec 2025)", OMEGA_A_CONV, W_A_CONV)]
MU_GRID = [1.0, 1.5, 2.0, 2.5, 3.5, 5.0]

grid = {}
lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{\textbf{Sensitivity of the identified-set upper endpoint.} Each cell is"
    r" $q(\mu)=1-\omega\,(w+\mu(a-f))/w$ evaluated at $w=0.755$, $a=0.245$,"
    r" $f=0.020$ (all rows except the last):"
    r" the largest combatant share consistent with anchor $\omega$ if civilian"
    r" working-age men die at exactly $\mu$ times the per-capita rate of the rest of"
    r" the population. The last row is the same Dec-2025 record under the men-18$+$"
    r" class convention ($w=0.733$, $a=0.267$; the record's sex-mixed elderly"
    r" category re-split at the 61.7\%-male share documented on the May-2025"
    r" identified list \citep{epstein2025}, an earlier vintage, so the row is"
    r" approximate): the class boundary"
    rf" is not innocuous---the upper endpoint at $\mu=1$ moves from {100*q1_moh:.1f}\% to"
    rf" {100*q1_a_conv:.1f}\%---but the calibrated bound stays below the disputed"
    r" claim's lower endpoint under either convention."
    r" The OHCHR anchor is approximate under the working-age convention (no elderly"
    r" split is published; the true $\omega'$ is slightly higher, so rejections on"
    r" it hold a fortiori)."
    r" The identified set for exposure range $[\mu_{\mathrm{lo}},\mu_{\mathrm{hi}}]$ is"
    r" $[q(\mu_{\mathrm{hi}}),\,q(\mu_{\mathrm{lo}})]\cap[0,1]$. Dashes: $q(\mu)\le 0$"
    r" at that exact exposure ratio, i.e.\ the anchor admits no positive combatant"
    r" share there; an exposure \emph{range} whose lower end still gives $q\ge 0$"
    r" has its lower endpoint clipped at zero (zero combatants remains consistent),"
    r" while a range lying entirely in the dashed region yields an empty set: that"
    r" anchor and that exposure range are jointly inconsistent."
    r" Exposure ratios calibrated on conflicts with recorded combatant status are"
    r" $\mu\gtrsim 2$ \citep{frost2026}; airstrike-specific data suggest a smaller"
    r" differential \citep{cockerill2024}.}",
    r"\label{tab:sensitivity}",
    r"\begin{small}",
    r"\renewcommand{\arraystretch}{1.3}",
    r"\begin{tabular}{@{}lcccccc@{}}",
    r"\toprule",
    r"Anchor $\omega$ (share of dead) & $\mu{=}1$ & $\mu{=}1.5$ & $\mu{=}2$ &"
    r" $\mu{=}2.5$ & $\mu{=}3.5$ & $\mu{=}5$ \\",
    r"\midrule",
]
for label, om, w_row in ANCHORS:
    a_row = 1.0 - w_row
    cells = []
    for mu in MU_GRID:
        qv = q_of_mu(om, w_row, a_row, F_STOCK, mu)
        grid[(label, mu)] = qv
        cells.append(f"{100*qv:.1f}\\%" if qv > 0 else "--")
    name = label.split(" (")[0]
    note = label[label.find("("):] if "(" in label else ""
    lines.append(f"{name} {note}, $\\omega={om:.3f}$ & " + " & ".join(cells) + r" \\")
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{small}",
    r"\end{table}",
]
if CONTENT.is_dir():
    (CONTENT / "tab_sensitivity.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {CONTENT / 'tab_sensitivity.tex'}")
else:
    print("  skipped tab_sensitivity.tex (no paper/content/)")

# ===========================================================================
# 5. Convergence-of-methods table -> LaTeX
# ===========================================================================

# 95% CI from gaza_sim/posterior.npz (seed 42; estimand q = D_milt /
# D_total with the working-age 18-59 class partition): q = 2.0%
# [1.4%, 2.8%], civilian-to-combatant ratio 49:1 [35, 72].
SPATIAL_Q, SPATIAL_Q_LO, SPATIAL_Q_HI = 0.020, 0.014, 0.028
SPATIAL_CCR, SPATIAL_CCR_LO, SPATIAL_CCR_HI = 49, 35, 72

conv_rows = [
    # (method, source, q_lo, q_hi, note)
    ("IDF public claim (17--25k of 71,444)", r"\citep{idfclaim}",
     q_idf_lo, q_idf_hi, "object of the test"),
    # Exposure-agnostic set: any mu >= 1; q(mu) <= 0 beyond mu* ~ 3.06,
    # so the set is [0, q(1)] (lower endpoint clipped at zero), matching
    # Step 2's statement -- not the display-grid range [q(2.5), q(1)].
    ("Identified set, MoH anchor, $\\mu\\ge 1$", "this paper",
     0.0, q1_moh, "weakest assumption ($\\mu\\ge 1$)"),
    ("Identified set, MoH anchor, $\\mu\\in[2,3.5]$", "this paper",
     set_moh_2_35[0], set_moh_2_35[1], "Frost-calibrated exposure"),
    ("Aman named-militant database", r"\citep{guardian972db2025}",
     q_aman, q_aman, "named fraction: 8,900 of 53,000 (May 2025); coverage unknown"),
    ("Male-bias model, MoH record", r"\citep{frost2026}",
     q_of_ccr(8.0), q_of_ccr(4.9), "B'tselem-calibrated"),
    ("Demographic decomposition", r"\citep{cockerill2024}",
     q_of_ccr(9.6), q_of_ccr(2.8), "range of male-bias assumptions"),
    ("Spatial Bayesian posterior", r"this paper, App.~\ref{app:spatial}",
     SPATIAL_Q_LO, SPATIAL_Q_HI, "prior-dependent; one point in the set"),
]
lines = [
    r"\begin{table}[ht]",
    r"\centering",
    rf"\caption{{\textbf{{Convergence of methods on the Gaza combatant"
    rf" share.}} Every method's range lies within $q\in[0,{{\sim}}{math.ceil(100*q1_moh):d}\%]$, with"
    r" point estimates and calibrated bounds concentrated at the low end"
    r" (in civilian-to-combatant terms, the loosest bound concedes"
    rf" ${{\sim}}{ccr_of_q(q1_moh):.0f}{{:}}1$ or more, the calibrated bounds"
    rf" at least ${{\sim}}{ccr_of_q(q2_moh):.0f}{{:}}1$, and the model-based"
    rf" estimate centres at ${SPATIAL_CCR:d}{{:}}1$, on direct deaths). The upper end of the IDF claim (${100*q_idf_hi:.1f}\%$) lies above"
    rf" every range; its lower end (${100*q_idf_lo:.1f}\%$) is admitted only at exposure"
    rf" ratios below the calibrated range ($\mu\le{mu_needed(q_idf_lo, OMEGA_MOH, W_POP, A_POP, F_STOCK):.2f}$) and lies above every"
    r" calibrated bound and point estimate, the one exception being the extreme"
    rf" low-male-bias end of the decomposition range (${100*q_of_ccr(2.8):.1f}\%$). The methods"
    r" share some inputs (MoH record, PCBS census) and are convergent"
    r" cross-checks, not statistically independent replicates; in particular,"
    r" the calibrated identified-set row imports its exposure range from the"
    r" male-bias calibration underlying the \citet{frost2026} row, so those two"
    r" rows share the calibration itself, not just the data.}",
    r"\label{tab:convergence}",
    r"\begin{footnotesize}",
    r"\renewcommand{\arraystretch}{1.3}",
    r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xll"
    r">{\raggedright\arraybackslash}X@{}}",
    r"\toprule",
    r"Method & Source & Implied $q$ & Note \\",
    r"\midrule",
]
for i, (method, src, qlo, qhi, note) in enumerate(conv_rows):
    qs = f"{100*qlo:.1f}\\%" if abs(qhi - qlo) < 5e-4 else \
         f"{100*qlo:.1f}--{100*qhi:.1f}\\%"
    lines.append(f"{method} & {src} & {qs} & {note} \\\\")
    if i == 0:  # rule below the tested claim, separating it from the methods
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabularx}", r"\end{footnotesize}", r"\end{table}"]
if CONTENT.is_dir():
    (CONTENT / "tab_convergence.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {CONTENT / 'tab_convergence.tex'}")
else:
    print("  skipped tab_convergence.tex (no paper/content/)")

# ===========================================================================
# 6. Huber bias-robust inflation of the primary bound (Prop 'robust')
# ===========================================================================

print("\n== Huber inflation ==")
# R_i = diameter of the identified set for q with source i removed
# (Prop 'robust'): the set is [0, q_max], so R_i = q_max after removal.
# Convention: the removed source's information is replaced by a background
# envelope where one exists (the census case below); retained inputs stay
# at their measured central values (e.g. M = 45k in R_demo).
# Removing the demographic source entirely: q in [0, M/D].
R_demo = M_STOCK / D_MOH
# Removing the census: w ranges over a generous global envelope [0.72, 0.78]
# (the working-age-complement analogue of the [0.70, 0.76] women-and-
# children envelope, shifted up by the elderly-men population share) with
# a = 1 - w; the exposure-agnostic set is [0, max_w q(1; w)], attained
# at w = 0.78.
CENSUS_ENV_LO, CENSUS_ENV_HI = 0.72, 0.78
R_census = max(q_of_mu(OMEGA_MOH, wv, 1 - wv, F_STOCK, 1.0)
               for wv in (CENSUS_ENV_LO, CENSUS_ENV_HI))
# Removing the manpower source frees f up to its logical cap f <= a, at
# which q(1) reaches the all-male ceiling 1 - omega.
R_M = 1.0 - OMEGA_MOH
EPS = 0.05
# The removal diameters are computed on the EXPOSURE-AGNOSTIC set and then
# applied to both exposure regimes.  For the calibrated endpoint this
# reuses the larger agnostic census diameter -- a deliberately conservative
# (wider) inflation: recomputing R_census within mu in [2,3.5] would give
# ~0.209 and a calibrated cap of 23.1% instead of 23.8%.  Stated in the
# prose (Section 5).
inflation = EPS * (R_demo + R_census + R_M)
q1_moh_robust = min(1.0, q1_moh + inflation)
q2_moh_robust = min(1.0, q2_moh + inflation)
print(f"  R_demo={R_demo:.3f}, R_census={R_census:.3f}, R_M={R_M:.3f}; "
      f"eps={EPS} -> inflation {inflation:.3f}")
print(f"  bias-robust upper bounds: q(1) {q1_moh:.3f} -> {q1_moh_robust:.3f}; "
      f"q(2) {q2_moh:.3f} -> {q2_moh_robust:.3f}")
# Under the contamination allowance the exposure-agnostic cap (39.4%) no
# longer sits below the 25k endpoint (35.0%); the calibrated inflated cap
# still rejects the 25k endpoint, while at the 17k endpoint the inflated
# cap and the claim's lower endpoint essentially coincide (23.777% vs
# 23.795%) -- a 0.018 pp margin on which no rejection claim is made in
# either direction (the band is a first-order statement in any case).
check_true("robust agnostic q(1) exceeds IDF 25k (agnostic rejection does"
           " not survive contamination)", q1_moh_robust > q_idf_hi,
           f"{q1_moh_robust:.3f} > {q_idf_hi:.3f}")
check_true("robust calibrated q(2) still rejects IDF 25k",
           q2_moh_robust < q_idf_hi,
           f"{q2_moh_robust:.3f} < {q_idf_hi:.3f}")
check_true("robust calibrated q(2) essentially coincides with IDF 17k"
           " (knife edge; no rejection claimed either way)",
           abs(q2_moh_robust - q_idf_lo) < 0.001,
           f"|{q2_moh_robust:.4f} - {q_idf_lo:.4f}| < 0.001")

# ===========================================================================
# 6b. Prose macros -> paper/content/numbers.tex
#     Every derived value quoted in the prose is emitted as a LaTeX macro,
#     so the prose always renders exactly what this script computes.
#     Raw measured inputs (0.523, 0.755, 71,444, 17-25k, 45k, ...) stay as
#     literals in the .tex files; only derived quantities live here.
# ===========================================================================


def fmt_pct(x, nd=1):
    """x as a percentage with nd decimals, digits only (caller adds \\%)."""
    return f"{100 * x:.{nd}f}"


def fmt_int(x):
    return f"{round(x):d}"


def fmt_grp(n):
    """Integer with math-mode thousands separators, e.g. 39{,}200."""
    return f"{round(n):,d}".replace(",", "{,}")


# Quantities quoted in the prose but not bound to a name above.
q_onefive_moh = q_of_mu(OMEGA_MOH, W_POP, A_POP, F_STOCK, 1.5)
q1_ohchr = q_of_mu(OMEGA_OHCHR, W_POP, A_POP, F_STOCK, 1.0)
mu_star_ceil = math.ceil(mu_star_q0 * 100) / 100      # q(mu) <= 0 for mu >= this
q1_count_k = q1_moh * D_MOH / 1000                    # mu=1 cap in thousands of dead
d_need_cal_k = round(IDF_LO / q2_moh / 10_000) * 10   # D making 17k calibrated-feasible
d_need_cal_hi_k = round(IDF_HI / q2_moh / 10_000) * 10  # D making 25k calibrated-feasible
corr_add = D_CORRECTED - D_MOH                        # deaths the GMS correction adds
corr_add_missing = round((D_MAX - D_MOH) / 1000) * 1000
absorb_needed_k = round((IDF_HI + corr_add) / 1000)   # combatant side absorbing it all
p_mant, p_exp = f"{p_one:.1e}".split("e")             # one-sided uniformity p
row_17k, row_25k = rows_budget[3], rows_budget[4]

MACRO_GROUPS = [
    ("Identified set and its sensitivities (MoH anchor unless noted)", [
        ("nQOne", fmt_pct(q1_moh)),                       # 32.1
        ("nQOneRound", fmt_int(100 * q1_moh)),            # 32
        ("nQOneCeil", f"{math.ceil(100 * q1_moh):d}"),    # 33
        ("nQOnePointFive", fmt_pct(q_onefive_moh)),       # 24.3
        ("nQTwo", fmt_pct(q2_moh)),                       # 16.5
        ("nQTwoRound", fmt_int(100 * q2_moh)),            # 17
        ("nQOneOhchr", fmt_pct(q1_ohchr)),                # 10.0
        ("nQOneFLo", fmt_pct(q1_f_lo)),                   # 31.8 (f = 0.0157)
        ("nQOneFHi", fmt_pct(q1_f_hi)),                   # 32.6 (f = 0.0269)
        ("nQOneCount", f"{q1_count_k:.1f}"),              # 22.9 (k of identified dead)
        ("nMuStar", f"{mu_star_q0:.2f}"),                 # 3.06 (root of q(mu)=0)
        ("nMuStarOneDp", f"{mu_star_q0:.1f}"),            # 3.1
        ("nMuStarCeil", f"{mu_star_ceil:.2f}"),           # 3.07 (q(mu)<=0 beyond it)
        ("nMOverD", f"{M_STOCK / D_MOH:.2f}"),            # 0.63 (manpower bound)
        ("nBlend", f"{OMEGA_BLEND:.3f}"),                 # 0.602 (geometric blend)
        ("nOmegaAConv", f"{OMEGA_A_CONV:.3f}"),           # 0.479 (men-18+, Dec 2025)
        ("nQOneAConv", fmt_pct(q1_a_conv)),               # 36.0 (men-18+ convention)
        ("nQTwoAConv", fmt_pct(q2_a_conv)),               # 19.8 (men-18+, calibrated)
        ("nMuIdfHiAConv", f"{mu_req_hi_a:.2f}"),          # 1.06 (25k under men-18+)
        ("nCcrAtQOne", fmt_int(ccr_of_q(q1_moh))),        # 2 (CCR at set's upper end)
        ("nCcrCalibratedFloor", fmt_int(ccr_of_q(q2_moh))),  # 5 (CCR at calibrated ceiling)
    ]),
    ("Uniformity discrepancy (OHCHR sample vs population, descriptive;"
     " matched men-18+ convention)", [
        ("nZUnif", f"{z:.1f}"),                                     # -8.1
        ("nPUnif", f"{p_mant}\\times 10^{{{int(p_exp)}}}"),         # ~2e-16
    ]),
    ("IDF claim converted to shares, and required exposure ratios", [
        ("nIdfQLo", fmt_pct(q_idf_lo)),                   # 23.8 (17k / 71,444)
        ("nIdfQHi", fmt_pct(q_idf_hi)),                   # 35.0 (25k / 71,444)
        ("nIdfQLoRound", fmt_int(100 * q_idf_lo)),        # 24
        ("nIdfQHiRound", fmt_int(100 * q_idf_hi)),        # 35
        ("nMuIdfLo", f"{mu_req_lo:.2f}"),                 # 1.53 (rationalises 17k)
        ("nMuIdfHi", f"{mu_req_hi:.2f}"),                 # 0.82 (rationalises 25k)
        ("nIdfCcrAtHi", fmt_int(ccr_of_q(q_idf_hi))),     # 2 (CCR at 25k endpoint)
        ("nIdfCcrAtLo", fmt_int(ccr_of_q(q_idf_lo))),     # 3 (CCR at 17k endpoint)
    ]),
    ("Contradiction radii (units of the stated sigma scales)", [
        ("nRhoOmegaAg", f"{rho_ag['MoH'][1]:.1f}"),       # 2.2 (25k, agnostic)
        ("nRhoOmegaAgRound", fmt_int(rho_ag["MoH"][1])),  # 2
        ("nRhoOmegaCal", f"{rho_cal['MoH'][1]:.1f}"),     # 11.6 (25k, calibrated)
        ("nRhoIdfLoCal", f"{rho_cal['MoH'][0]:.1f}"),     # 4.6 (17k, calibrated)
        ("nRhoIdfLoCalRound", fmt_int(rho_cal["MoH"][0])),  # 5
        ("nRhoOhchrAgRound", fmt_int(rho_ag["OHCHR"][1])),  # 19 (25k, OHCHR anchor)
        ("nRhoOhchrCalRound", fmt_int(rho_cal["OHCHR"][1])),  # 29 (empty-set config)
        ("nWRescue", f"{w_rescue:.3f}"),                  # 0.788 (census rescue w')
        ("nWRescuePp", f"{100 * (w_rescue - W_POP):.1f}"),  # 3.3 (pp census error)
        ("nRhoW", fmt_int(rho_w_25k)),                    # 7
        ("nMRescue", fmt_grp(round(M_rescue / 1000) * 1000)),  # 138{,}000
        ("nMRescueRatio", f"{M_rescue / M_STOCK:.1f}"),   # 3.1 (times the stock)
        ("nRhoM", fmt_int(rho_M_25k)),                    # 7
        ("nRhoJoint", f"{rho_joint:.1f}"),                # 1.7 (joint omega-w move)
    ]),
    ("Huber removal diameters and bias-robust band", [
        ("nRDemo", f"{R_demo:.2f}"),                      # 0.63
        ("nRCensus", f"{R_census:.2f}"),                  # 0.34
        ("nRCensusSet", f"{R_census:.3f}"),               # 0.343 (set endpoint)
        ("nRM", f"{R_M:.2f}"),                            # 0.48
        ("nHuberPP", fmt_pct(inflation)),                 # 7.2 (percentage points)
        ("nHuberInflated", fmt_pct(q1_moh_robust)),       # 39.4
        ("nHuberInflatedCal", fmt_pct(q2_moh_robust)),    # 23.8 (calibrated cap)
    ]),
    ("D-sensitivity: corrected denominators and the civilian ledger", [
        ("nGmsRatio", f"{GMS_UNDERCOUNT:.2f}"),           # 1.53 (survey/recorded)
        ("nDCorr", fmt_int(D_CORRECTED / 1000)),          # 109 (k)
        ("nDMax", fmt_int(D_MAX / 1000)),                 # 121 (k)
        ("nIdfQCorrLo", fmt_int(100 * IDF_LO / D_CORRECTED)),  # 16
        ("nIdfQCorrHi", fmt_int(100 * IDF_HI / D_CORRECTED)),  # 23
        ("nIdfQMaxLo", fmt_int(100 * IDF_LO / D_MAX)),    # 14
        ("nIdfQMaxHi", fmt_int(100 * IDF_HI / D_MAX)),    # 21
        ("nMuMaxSeventeen", f"{d_rows[-1][4]:.2f}"),      # 2.16 (17k at D~121k)
        ("nMuMaxTwentyFive", f"{d_rows[-1][5]:.2f}"),     # 1.74 (25k at D~121k)
        ("nMuCorrSeventeen", f"{d_rows[1][4]:.2f}"),      # 2.06 (17k at D~109k)
        ("nMuCorrTwentyFive", f"{d_rows[1][5]:.2f}"),     # 1.59 (25k at D~109k)
        ("nCivMohLo", fmt_int((D_MOH - IDF_HI) / 1000)),  # 46 (k)
        ("nCivMohHi", fmt_int((D_MOH - IDF_LO) / 1000)),  # 54 (k)
        ("nCivCorrLo", fmt_int((D_CORRECTED - IDF_HI) / 1000)),  # 84 (k)
        ("nCivCorrHi", fmt_int((D_CORRECTED - IDF_LO) / 1000)),  # 92 (k)
        ("nCivMaxLo", fmt_int((D_MAX - IDF_HI) / 1000)),  # 96 (k)
        ("nCivMaxHi", fmt_int((D_MAX - IDF_LO) / 1000)),  # 104 (k)
        ("nCcrCorrLo", f"{(D_CORRECTED - IDF_HI) / IDF_HI:.1f}"),  # 3.4
        ("nCcrCorrHi", f"{(D_CORRECTED - IDF_LO) / IDF_LO:.1f}"),  # 5.4
        ("nCcrMaxLo", f"{(D_MAX - IDF_HI) / IDF_HI:.1f}"),         # 3.8
        ("nCcrMaxHi", f"{(D_MAX - IDF_LO) / IDF_LO:.1f}"),         # 6.1
        ("nCorrAdd", fmt_grp(round(corr_add / 1000) * 1000)),      # 38{,}000
        ("nCorrAddMissing", fmt_grp(corr_add_missing)),   # 50{,}000
        ("nAbsorbNeeded", fmt_int(absorb_needed_k)),      # 63 (k)
        ("nDNeeded", fmt_int(d_need_cal_k)),              # 100 (k)
        ("nDNeededHi", fmt_int(d_need_cal_hi_k)),         # 150 (k)
    ]),
    ("Working-age-male death budget", [
        ("nBudgetWc", fmt_grp(D_WC)),                     # 37{,}366
        ("nBudgetMen", fmt_grp(D_MEN)),                   # 34{,}078
        ("nBudgetShift", fmt_grp(round(SIGMA_OMEGA * D_MOH / 100) * 100)),  # 700
        ("nRateWc", fmt_pct(RATE_WC)),                    # 2.2
        ("nRateRandom", fmt_pct(D_MOH / N_POP)),          # 3.2
        ("nRandomCombatants", fmt_grp(round(row_random["d_combatant"]
                                            / 100) * 100)),         # 1{,}400
        ("nRandomQ", fmt_pct(row_random["d_combatant"] / D_MOH)),   # 2.0
        ("nShareAmCiv", fmt_pct(A_POP - F_STOCK)),        # 22.5 (of population)
        ("nPopWc", fmt_grp(N_WC)),                        # 1{,}698{,}750
        ("nPopAmCiv", fmt_grp(N_AMCIV)),                  # 506{,}250
        ("nKillFracSeventeenIiss", fmt_int(100 * IDF_LO / M_STOCK)),          # 38
        ("nKillFracTwentyFiveIiss", fmt_int(100 * IDF_HI / M_STOCK)),         # 56
        ("nKillFracSeventeenIdf", fmt_int(100 * IDF_LO / IDF_PREWAR_STOCK)),  # 57
        ("nKillFracTwentyFiveIdf", fmt_int(100 * IDF_HI / IDF_PREWAR_STOCK)),  # 83
        ("nKillFracSeventeenTop", fmt_int(100 * IDF_LO / 60_000)),            # 28
        ("nRateAmcivSeventeen", fmt_pct(row_17k["rate_amciv"])),   # 3.4
        ("nRateAmcivTwentyFive", fmt_pct(row_25k["rate_amciv"])),  # 1.8
        ("nSelSeventeen", fmt_int(row_17k["selectivity"])),        # 11
        ("nSelTwentyFive", fmt_int(row_25k["selectivity"])),       # 31
        ("nAllMenQ", fmt_pct(D_MEN / D_MOH)),             # 47.7
        ("nAllMenStock", fmt_grp(N_AM)),                  # 551{,}250
        ("nAllMenStockRatio", fmt_int(N_AM / IDF_PREWAR_STOCK)),   # 18
        ("nConcedeLo", fmt_grp(D_MEN - IDF_HI)),          # 9{,}078
        ("nConcedeHi", fmt_grp(D_MEN - IDF_LO)),          # 17{,}078
    ]),
    ("Named-combatant cross-check and manpower ledger", [
        ("nAmanQ", fmt_pct(q_aman)),                      # 16.8
        ("nLedgerRecruits", fmt_grp(recruits_implied)),   # 12{,}000
    ]),
    ("Spatial Bayesian posterior (gaza_sim, seed 42)", [
        ("nSpatialQ", fmt_pct(SPATIAL_Q)),                # 2.0
        ("nSpatialQRound", fmt_int(100 * SPATIAL_Q)),     # 2
        ("nSpatialQLo", fmt_pct(SPATIAL_Q_LO)),           # 1.4
        ("nSpatialQHi", fmt_pct(SPATIAL_Q_HI)),           # 2.8
        ("nSpatialCcr", f"{SPATIAL_CCR:d}"),              # 49
        ("nSpatialCcrLo", f"{SPATIAL_CCR_LO:d}"),         # 35
        ("nSpatialCcrHi", f"{SPATIAL_CCR_HI:d}"),         # 72
    ]),
]

ntex_lines = [
    "% =====================================================================",
    "% numbers.tex --- MACHINE-GENERATED by analysis/validate_bounds.py.",
    "% DO NOT EDIT BY HAND; rerun the script to regenerate.",
    "% Each macro holds one derived quantity, formatted exactly as the prose",
    "% renders it (percent macros carry digits only; the caller adds \\%).",
    "% Convention: macro bodies carry no trailing-space guard, so call with",
    "% empty braces --- \\nQOne{} --- wherever a letter or space follows.",
    "% =====================================================================",
]
for group_label, entries in MACRO_GROUPS:
    ntex_lines.append(f"% --- {group_label} ---")
    for name, val in entries:
        ntex_lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
if CONTENT.is_dir():
    (CONTENT / "numbers.tex").write_text("\n".join(ntex_lines) + "\n")
    print(f"\nWrote {CONTENT / 'numbers.tex'} "
      f"({sum(len(e) for _, e in MACRO_GROUPS)} macros)")
else:
    print("skipped numbers.tex (no paper/content/)")

# ===========================================================================
# 7. Report
# ===========================================================================

report = {
    "inputs": {
        "w_pop": W_POP, "a_pop": A_POP, "f_stock": F_STOCK,
        "omega_moh": OMEGA_MOH, "omega_ohchr": OMEGA_OHCHR,
        "omega_blend": OMEGA_BLEND, "sigma_omega": SIGMA_OMEGA,
        "D_moh": D_MOH, "M_stock": M_STOCK,
        "idf_claim": [IDF_LO, IDF_HI],
    },
    "headline": {
        "q1_moh": q1_moh, "q2_moh": q2_moh,
        "set_moh_mu_1_25": set_moh_1_25, "set_moh_mu_2_35": set_moh_2_35,
        "mu_needed_idf17k": mu_req_lo, "mu_needed_idf25k": mu_req_hi,
        "rho_agnostic_mu_1_25": rho_ag, "rho_calibrated_mu_2_35": rho_cal,
        "mu_star_q0": mu_star_q0,
        "q_aman": q_aman,
        "q_frost": [q_of_ccr(8.0), q_of_ccr(4.9)],
        "q_cockerill": [q_of_ccr(9.6), q_of_ccr(2.8)],
        "uniformity_z": z, "uniformity_p_one_sided": p_one,
        "huber": {"R_demo": R_demo, "R_census": R_census, "R_M": R_M, "eps": EPS,
                  "inflation": inflation, "q1_robust": q1_moh_robust,
                  "q2_robust": q2_moh_robust},
    },
    "budget": {
        "N_pop": N_POP, "N_wc": N_WC, "N_am": N_AM, "N_amciv": N_AMCIV,
        "M_stock": M_STOCK, "idf_prewar_stock": IDF_PREWAR_STOCK,
        "D": D_MOH, "D_wc": D_WC, "D_men": D_MEN, "rate_wc": RATE_WC,
        "rows": rows_budget,
    },
    "d_sensitivity": [
        {"label": l, "D": D, "q_idf": [ql, qh], "mu_needed": [ml, mh]}
        for l, D, ql, qh, ml, mh in d_rows
    ],
    "failures": FAILURES,
}
out = ROOT / "analysis" / "validation_report.json"
out.write_text(json.dumps(report, indent=2))
print(f"\nWrote {out}")

if FAILURES:
    print(f"\n*** {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("\nAll checks passed.")
