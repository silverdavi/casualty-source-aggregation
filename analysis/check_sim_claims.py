"""Machine-check of Appendix B's behavioral claims about the Gaza simulator.

Purpose
-------
The "Sensitivities and limitations" paragraph of Appendix B
(paper/content/09_appendix.tex) makes several checkable statements about
what gaza_sim/01_simulator.py computes: the prior-alone q, the
full-likelihood posterior, the MoH-only and OHCHR-only re-anchored
posteriors, the 1/2 weighting of the two demographic likelihood terms,
the importance-sampling ESS, the civilian-to-combatant ratio, and the
maximum effective male-to-W/C exposure ratio implied by the priors.

This script executes the simulator module (seed 42, 200,000 draws),
re-derives every one of those statements from the module's own arrays and
likelihood function, and asserts each against the appendix's quoted
values.  Any drift between the code and the appendix text fails the run.

Exit code is nonzero if ANY check fails.  Run:

    python3 analysis/check_sim_claims.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def check_range(name: str, got: float, lo: float, hi: float):
    ok = lo < got < hi
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {got:.6g}, required in ({lo:g}, {hi:g})")
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
# 0. Load the simulator module (it runs its full pipeline at import)
# ===========================================================================

print("== Loading gaza_sim/01_simulator.py (runs the full pipeline, ~10s) ==")
spec = importlib.util.spec_from_file_location(
    "gaza_simulator", ROOT / "gaza_sim" / "01_simulator.py")
mod = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(mod)
print(f"  loaded: B={len(mod.w):,} draws, module ESS={mod.ess:,.0f}")

sim, theta = mod.sim, mod.theta
q_arr = sim["D_milt"] / np.maximum(sim["D_total"], 1)


def wq(arr, weights, q):
    """Weighted quantile: sort, cumsum of weights, searchsorted."""
    s = np.argsort(arr)
    cw = np.cumsum(weights[s])
    cw /= cw[-1]
    return arr[s[np.searchsorted(cw, q)]]


def summary(arr, weights):
    return (wq(arr, weights, 0.5), wq(arr, weights, 0.025),
            wq(arr, weights, 0.975))


# ===========================================================================
# 1. Reconstruct the likelihood components exactly as the module does
# ===========================================================================
# Appendix B, "Likelihood": (i) total recorded deaths, log-Gaussian at 7%
# scale; (ii) adult-male share vs OHCHR at its sample size; (iii) the same
# share vs the MoH record at effective n/5.  These are recomputed here from
# the module's own arrays and constants so the re-anchored variants below
# use the identical functional forms.

share_AM = (sim["D_civAM"] + sim["D_milt"]) / np.maximum(sim["D_total"], 1)
share_AM_c = np.clip(share_AM, 1e-3, 1 - 1e-3)

lk_total = stats.norm.logpdf(np.log(sim["D_obs"]),
                             loc=np.log(mod.MOH_LATE2025), scale=0.07)
lk_ohchr = stats.beta.logpdf(
    share_AM_c,
    mod.omega_AM_ohchr * mod.N_OHCHR + 1,
    (1 - mod.omega_AM_ohchr) * mod.N_OHCHR + 1)
n_eff_moh = mod.N_MOH / 5.0
lk_moh = stats.beta.logpdf(
    share_AM_c,
    mod.omega_AM_moh * n_eff_moh + 1,
    (1 - mod.omega_AM_moh) * n_eff_moh + 1)


def weights_from(log_w):
    return np.exp(log_w - log_w.max())


w_full = weights_from(lk_total + 0.5 * (lk_ohchr + lk_moh))
w_moh_only = weights_from(lk_total + lk_moh)      # no OHCHR, no 1/2 factor
w_ohchr_only = weights_from(lk_total + lk_ohchr)  # no MoH, no 1/2 factor
w_prior = np.ones_like(q_arr)                     # prior alone

# ===========================================================================
# Claim 5 (structural, checked first since the variants depend on it):
# Appendix B states the two demographic log-likelihoods enter with weight
# 1/2 each.  The module's log_likelihood must equal
# lk_total + 0.5*(lk_moh + lk_ohchr); this pins that sentence to the code.
# ===========================================================================

print("\n== Claim 5: demographic terms carry weight 1/2 each ==")
lk_module = mod.log_likelihood(sim["D_obs"], sim["D_milt"],
                               sim["D_civAM"], sim["D_WC"])
check_true("module log_likelihood == lk_total + 0.5*(lk_moh + lk_ohchr)",
           np.allclose(lk_module, lk_total + 0.5 * (lk_moh + lk_ohchr)),
           f"max |diff| = {np.abs(lk_module - (lk_total + 0.5 * (lk_ohchr + lk_moh))).max():.3g}")
# The reconstructed full weights must reproduce the module's own weights.
check_true("reconstructed full weights match module weights",
           np.allclose(w_full, mod.w), "same normalization (max-shifted exp)")

# ===========================================================================
# Claim 1: prior-alone q.  Appendix B quotes "2.1% [1.4%, 3.0%]".
# ===========================================================================

print("\n== Claim 1: prior-alone q ==")
med, lo, hi = summary(q_arr, w_prior)
print(f"  prior-alone q: median {med:.4%}, 95% [{lo:.4%}, {hi:.4%}]")
check_range("prior-alone median in (1.9%, 2.2%)", med, 0.019, 0.022)
check_range("prior-alone lo in (1.2%, 1.6%)", lo, 0.012, 0.016)
check_range("prior-alone hi in (2.8%, 3.2%)", hi, 0.028, 0.032)

# ===========================================================================
# Claim 2: full-likelihood posterior.  Appendix B quotes "2.0% [1.4%, 2.8%]".
# ===========================================================================

print("\n== Claim 2: full-likelihood posterior q ==")
med_full, lo_full, hi_full = summary(q_arr, w_full)
print(f"  full posterior q: median {med_full:.4%}, 95% [{lo_full:.4%}, {hi_full:.4%}]")
check_range("full median in (1.9%, 2.05%)", med_full, 0.019, 0.0205)
check_range("full lo consistent with quoted 1.4% (±0.1pp)",
            lo_full, 0.013, 0.015)
check_range("full hi consistent with quoted 2.8% (±0.1pp)",
            hi_full, 0.027, 0.029)

# ===========================================================================
# Claim 3: MoH-only re-anchoring.  Appendix B states this moves the
# posterior slightly DOWN, to "2.0% [1.4%, 2.7%]".
# ===========================================================================

print("\n== Claim 3: MoH-only re-anchored posterior ==")
med_moh, lo_moh, hi_moh = summary(q_arr, w_moh_only)
print(f"  MoH-only q: median {med_moh:.4%}, 95% [{lo_moh:.4%}, {hi_moh:.4%}]")
check_true("MoH-only median below full-likelihood median",
           med_moh < med_full, f"{med_moh:.4%} < {med_full:.4%}")
check_range("MoH-only median in (1.9%, 2.05%)", med_moh, 0.019, 0.0205)
check_range("MoH-only lo within ±0.15pp of quoted 1.4%", lo_moh, 0.0125, 0.0155)
check_range("MoH-only hi within ±0.15pp of quoted 2.7%", hi_moh, 0.0255, 0.0285)

# ===========================================================================
# Claim 4: OHCHR-only re-anchoring.  Appendix B states this moves the
# posterior UP, to "2.3% [1.6%, 3.2%]".
# ===========================================================================

print("\n== Claim 4: OHCHR-only re-anchored posterior ==")
med_oh, lo_oh, hi_oh = summary(q_arr, w_ohchr_only)
print(f"  OHCHR-only q: median {med_oh:.4%}, 95% [{lo_oh:.4%}, {hi_oh:.4%}]")
check_true("OHCHR-only median above full-likelihood median",
           med_oh > med_full, f"{med_oh:.4%} > {med_full:.4%}")
check_range("OHCHR-only median in (2.2%, 2.45%)", med_oh, 0.022, 0.0245)
check_range("OHCHR-only lo within ±0.15pp of quoted 1.6%", lo_oh, 0.0145, 0.0175)
check_range("OHCHR-only hi within ±0.15pp of quoted 3.2%", hi_oh, 0.0305, 0.0335)

# ===========================================================================
# Claim 6: Appendix B states ESS ≈ 4,300 under the normalized full weights.
# ===========================================================================

print("\n== Claim 6: effective sample size ==")
ess = (w_full.sum() ** 2) / (w_full ** 2).sum()
check_range("ESS in (3,800, 4,700)", ess, 3_800, 4_700)
check_true("recomputed ESS matches the module's own",
           abs(ess - mod.ess) < 1.0, f"{ess:.1f} vs {mod.ess:.1f}")

# ===========================================================================
# Claim 7: Appendix B quotes a civilian-to-combatant ratio of 49:1 [35, 72];
# the ratio is (1-q)/q draw-by-draw under the full weights.
# ===========================================================================

print("\n== Claim 7: civilian-to-combatant ratio ==")
ratio = (1 - q_arr) / q_arr
med_r, lo_r, hi_r = summary(ratio, w_full)
print(f"  ratio: median {med_r:.2f}, 95% [{lo_r:.2f}, {hi_r:.2f}]")
check_range("ratio median in (47, 51)", med_r, 47, 51)
check_range("ratio lo in (33, 37)", lo_r, 33, 37)
check_range("ratio hi in (68, 76)", hi_r, 68, 76)

# ===========================================================================
# Claim 8: Appendix B states the effective male-to-W/C exposure ratio can
# reach ≈3.6, i.e. max(mu_M)/min(eps_C) = 2.5/0.7 from the prior bounds.
# ===========================================================================

print("\n== Claim 8: maximum effective male-to-W/C exposure ratio ==")
mu_M_hi = mod.PRIOR["mu_M"][1]
eps_C_lo = mod.PRIOR["eps_C"][0]
max_ratio = mu_M_hi / eps_C_lo
print(f"  prior bounds: mu_M hi = {mu_M_hi}, eps_C lo = {eps_C_lo}, "
      f"ratio = {max_ratio:.4f}")
check_true("prior-bound ratio equals 2.5/0.7 within 1%",
           abs(max_ratio - 2.5 / 0.7) <= 0.01 * (2.5 / 0.7),
           f"{max_ratio:.4f} vs {2.5 / 0.7:.4f}")
check_true("ratio rounds to the quoted 3.6", round(max_ratio, 1) == 3.6,
           f"round({max_ratio:.4f}, 1) = {round(max_ratio, 1)}")
# The sampled arrays must respect the same bounds (uniform priors).
check_true("sampled mu_M max approaches prior bound",
           abs(theta["mu_M"].max() - mu_M_hi) < 1e-3,
           f"max sampled = {theta['mu_M'].max():.5f}")
check_true("sampled eps_C min approaches prior bound",
           abs(theta["eps_C"].min() - eps_C_lo) < 1e-3,
           f"min sampled = {theta['eps_C'].min():.5f}")

# ===========================================================================

if FAILURES:
    print(f"\n*** {len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("\nAll simulator-claim checks passed.")
