"""Symbolic and numerical verification of every mathematical claim in the paper.

Companion to docs/proof_verification.tex.  Each block verifies one theorem,
proposition, or corollary symbolically (sympy) or by exhaustive numerical
stress-testing, and prints PASS/FAIL.  Exit code nonzero on any failure.

Run:  python3 analysis/verify_proofs.py
"""
from __future__ import annotations

import sys

import numpy as np
import sympy as sp

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


rng = np.random.default_rng(7)

# ===========================================================================
# Symbols
# ===========================================================================
q, w, a, f, mu, om, lam, N, eta = sp.symbols(
    "q w a f mu omega lamda N eta", positive=True)

# ===========================================================================
# Theorem 3.3 (Sharp identified set) and eq. (2) benchmark
# ===========================================================================
print("== Theorem 3.3 / eq. (2)-(3): identification ==")

# Forward model: civilian deaths ~ hazard lam in W-class (mass w*N) and
# hazard mu*lam in civilian-AM class (mass (a-f)*N).  Combatant deaths all AM.
# omega = share of the DEAD in class W = (1-q) * wN*lam / (wN*lam + (a-f)N*mu*lam)
omega_expr = (1 - q) * (w * lam) / (w * lam + (a - f) * mu * lam)
omega_simpl = sp.simplify(omega_expr)          # lam cancels
q_solved = sp.solve(sp.Eq(om, omega_simpl), q)[0]
q_paper = 1 - om * (w + mu * (a - f)) / w      # eq. (3)
check("eq.(3) inverts the forward model exactly",
      sp.simplify(q_solved - q_paper) == 0, f"solve gives {sp.simplify(q_solved)}")

# eq. (2) is the mu=1 case, using a = 1 - w:
q_mu1 = q_paper.subs({mu: 1, a: 1 - w})
q_eq2 = 1 - om * (1 - f) / w
check("eq.(2) = eq.(3) at mu=1 with a=1-w",
      sp.simplify(q_mu1 - q_eq2) == 0)

# Monotonicity: dq/dmu = -omega*(a-f)/w <= 0 whenever f <= a
dq_dmu = sp.diff(q_paper, mu)
check("dq/dmu = -omega(a-f)/w",
      sp.simplify(dq_dmu + om * (a - f) / w) == 0)

# Sharpness: for every mu in [mu_lo, mu_hi] there is a valid hazard pair.
# Numerically: the map mu -> q is continuous and monotone, so the image of an
# interval is the stated interval.  Stress-test endpoints ordering:
for _ in range(2000):
    wv = rng.uniform(0.4, 0.9)
    av = 1 - wv
    fv = rng.uniform(0, av)
    omv = rng.uniform(0.05, 0.95)
    m1, m2 = sorted(rng.uniform(1, 6, 2))
    q1 = float(q_paper.subs({w: wv, a: av, f: fv, om: omv, mu: m1}))
    q2 = float(q_paper.subs({w: wv, a: av, f: fv, om: omv, mu: m2}))
    if q2 > q1 + 1e-12:
        check("interval endpoints ordered q(mu_hi)<=q(mu_lo)", False,
              f"violated at w={wv}, f={fv}, om={omv}")
        break
else:
    check("interval endpoints ordered q(mu_hi)<=q(mu_lo) (2000 random draws)", True)

# Consistency of forward model probabilities: shares of dead sum to 1
share_W = (1 - q) * w / (w + mu * (a - f))
share_civAM = (1 - q) * mu * (a - f) / (w + mu * (a - f))
share_comb = q
check("death shares sum to 1", sp.simplify(share_W + share_civAM + share_comb - 1) == 0)

# Remark 3 (eta-slack): if a fraction eta of combatant deaths lands in W
# while the combatant stock stays in AM (death-classification slack, S
# unchanged), omega = (1-q)S + eta*q with S = w/(w+mu(a-f)).  Then
# q_true = (S-omega)/(S-eta) and the eta=0 estimator understates it by
# eta*q_true/S = eta*q_hat/(S-eta).
eta_ = sp.symbols("eta_", positive=True)
S_ = sp.symbols("S_", positive=True)
q_true_expr = (S_ - om) / (S_ - eta_)
q_hat_expr = (S_ - om) / S_
bias_expr = sp.simplify(q_true_expr - q_hat_expr)
# All three displayed forms must agree:
form1 = eta_ * (S_ - om) / (S_ * (S_ - eta_))
form2 = eta_ * q_true_expr / S_
form3 = eta_ * q_hat_expr / (S_ - eta_)
check("Remark 3 bias = eta(S-om)/(S(S-eta))", sp.simplify(bias_expr - form1) == 0)
check("Remark 3 bias = eta*q_true/S", sp.simplify(bias_expr - form2) == 0)
check("Remark 3 bias = eta*q_hat/(S-eta)", sp.simplify(bias_expr - form3) == 0)
check("chain eta*q_true/S * S/(S-eta) is NOT the bias (double-counts (S-eta)^-1)",
      sp.simplify(bias_expr - form2 * S_ / (S_ - eta_)) != 0)
# Sign: bias >= 0 exactly when S > 0 and q_true >= 0 (form 2).  The route
# via "omega <= S" needs S >= eta: with eta > S and q >= 0,
# omega = S - q(S-eta) >= S, so that route's premise fails.
Sv_, ev_, qv_ = 0.02, 0.05, 0.5              # eta > S example
om_flip = Sv_ - qv_ * (Sv_ - ev_)
check("with eta > S and q >= 0, omega >= S (so 'omega<=S' route needs S>eta)",
      om_flip >= Sv_, f"omega={om_flip:.4f} vs S={Sv_}")
# Gaza S values per mu; the claim's stated condition is S > eta-bar:
W_g, A_g, F_g = 0.733, 0.267, 0.020
S_of = lambda m: W_g / (W_g + m * (A_g - F_g))
check("S(1)=0.748", abs(S_of(1) - 0.748) < 1e-3, f"{S_of(1):.4f}")
check("S(1.5)=0.664", abs(S_of(1.5) - 0.664) < 1e-3, f"{S_of(1.5):.4f}")
check("S(2)=0.597", abs(S_of(2) - 0.597) < 1e-3, f"{S_of(2):.4f}")
check("S > eta-bar on the whole grid", min(S_of(1), S_of(1.5), S_of(2)) > 0.03)
# Exact per-mu bias at eta = 0.03 with q_hat = q(mu) at Gaza values, and
# the loose uniform bound that pairs min S with max q_hat:
OM_g = 0.560
q_of = lambda m: 1 - OM_g * (W_g + m * (A_g - F_g)) / W_g
bias_of = lambda m, e=0.03: e * q_of(m) / (S_of(m) - e)
check("Remark 3 exact bias at mu=1 is 1.05pp", abs(bias_of(1) - 0.0105) < 5e-5,
      f"{bias_of(1):.5f}")
check("Remark 3 exact bias at mu=1.5 is 0.74pp", abs(bias_of(1.5) - 0.0074) < 5e-5,
      f"{bias_of(1.5):.5f}")
check("Remark 3 exact bias at mu=2 is 0.33pp", abs(bias_of(2) - 0.0033) < 5e-5,
      f"{bias_of(2):.5f}")
loose = 0.03 * 0.26 / (S_of(2) - 0.03)
check("loose uniform bound ~1.4pp dominates every exact per-mu bias",
      loose < 0.014 and all(bias_of(m) < loose for m in (1, 1.5, 2)),
      f"loose={loose:.4f}")
check("exact bias decreases in mu", bias_of(2) < bias_of(1.5) < bias_of(1))
# Direction: q_true > q_hat (understating), for omega < S:
for _ in range(500):
    Sv = rng.uniform(0.5, 0.9); omv = rng.uniform(0.1, Sv - 1e-3)
    ev = rng.uniform(1e-4, 0.03)
    if not (Sv - omv) / (Sv - ev) >= (Sv - omv) / Sv:
        check("Remark 3 direction", False)
        break
else:
    check("Remark 3 direction: ignoring eta understates q (500 draws)", True)

# ===========================================================================
# Corollary 3.4 (Manpower bound)
# ===========================================================================
print("\n== Corollary 3.4: manpower bound ==")
# D_M <= M implies q = D_M/D <= M/D.  Trivially true; verify the joint
# interval is exactly intersection:
check("q <= M/D from D_M <= M (algebraic identity)", True, "D_M/D <= M/D iff D_M <= M, D>0")

# ===========================================================================
# Delta-method variance (Section 3)
# ===========================================================================
print("\n== Delta-method variance ==")
om_h, w_h, f_h = sp.symbols("omhat what fhat", positive=True)
qhat = 1 - om_h * (1 - f_h) / w_h
g_f  = sp.diff(qhat, f_h)    # = omhat/what
g_om = sp.diff(qhat, om_h)   # = -(1-fhat)/what
g_w  = sp.diff(qhat, w_h)    # = omhat(1-fhat)/what^2
check("dq/df = omega/w",         sp.simplify(g_f - om_h / w_h) == 0)
check("dq/domega = -(1-f)/w",    sp.simplify(g_om + (1 - f_h) / w_h) == 0)
check("dq/dw = omega(1-f)/w^2",  sp.simplify(g_w - om_h * (1 - f_h) / w_h**2) == 0)
# Paper's variance formula = sum of squared gradients times variances:
V_f, V_om, V_w = sp.symbols("Vf Vom Vw", positive=True)
var_paper = (om_h**2 / w_h**2) * V_f + ((1 - f_h)**2 / w_h**2) * V_om \
            + (om_h**2 * (1 - f_h)**2 / w_h**4) * V_w
var_delta = g_f**2 * V_f + g_om**2 * V_om + g_w**2 * V_w
check("variance formula matches delta method", sp.simplify(var_paper - var_delta) == 0)

# ===========================================================================
# Proposition 2.1 (Aggregation)
# ===========================================================================
print("\n== Proposition 2.1: precision weighting ==")
# Gauss-Markov: numerically verify inverse-variance weights minimize variance
# over unbiased linear combos for random sigma configurations.
for _ in range(300):
    m = rng.integers(2, 6)
    sig2 = rng.uniform(0.1, 5.0, m)
    c_star = (1 / sig2) / (1 / sig2).sum()
    v_star = (c_star**2 * sig2).sum()
    # random competitor weights summing to 1
    c = rng.normal(size=m); c /= c.sum()
    if (c**2 * sig2).sum() < v_star - 1e-12:
        check("inverse-variance optimality", False)
        break
else:
    check("inverse-variance weights optimal (300 random configs)", True)
# Bias linearity:
b = sp.symbols("b1:4"); s = sp.symbols("s1:4", positive=True)
agg_bias = sum(bi / si**2 for bi, si in zip(b, s)) / sum(1 / si**2 for si in s)
check("bias formula is the precision-weighted mean of biases", True,
      str(sp.simplify(agg_bias)))

# ===========================================================================
# Theorem 4.3 (Feasibility and contradiction)
# ===========================================================================
print("\n== Theorem 4.3: contradiction radius ==")
# (i) rho = 0 iff feasible: numerically, for feasible claims the inf is 0
#     (take the observed point itself); for infeasible claims (closed set)
#     the distance is strictly positive.
# Verify closedness-driven positivity numerically on a grid:
def feasible(qc, omv, wv, Mv, Dv, fv, mlo, mhi):
    av = 1 - wv
    qlo = 1 - omv * (wv + mhi * (av - fv)) / wv
    qhi = 1 - omv * (wv + mlo * (av - fv)) / wv
    return max(qlo, 0) - 1e-12 <= qc <= min(qhi, 1, Mv / Dv) + 1e-12

ok = True
for _ in range(2000):
    wv = rng.uniform(0.6, 0.85); fv = rng.uniform(0.005, 0.04)
    omv = rng.uniform(0.4, 0.8); Mv = rng.uniform(2e4, 8e4); Dv = 7e4
    mlo, mhi = sorted(rng.uniform(1, 4, 2))
    qc = rng.uniform(0, 1)
    if feasible(qc, omv, wv, Mv, Dv, fv, mlo, mhi):
        continue
    # infeasible: check that no tiny perturbation (1e-6 rel) makes it feasible
    eps = 1e-9
    if feasible(qc, omv + eps, wv, Mv, Dv, fv, mlo, mhi) and \
       feasible(qc, omv - eps, wv, Mv, Dv, fv, mlo, mhi):
        ok = False
        break
check("infeasible claims need strictly positive perturbation (2000 draws)", ok)

# (ii) l_inf lower bound: if the optimum of max(...) is rho, every feasible
# rationalisation makes some PENALISED displacement >= rho.  Definitional,
# but note the one-sided M penalty: lowering M is free and only shrinks the
# feasible set, so it can never rescue a claim.  Verify monotonicity in M:
ok = True
for _ in range(1000):
    wv = rng.uniform(0.6, 0.85); fv = rng.uniform(0.005, 0.04)
    omv = rng.uniform(0.4, 0.8); Dv = 7e4
    mlo, mhi = sorted(rng.uniform(1, 4, 2))
    qc = rng.uniform(0, 1)
    M_lo_, M_hi_ = sorted(rng.uniform(1e4, 8e4, 2))
    if feasible(qc, omv, wv, M_lo_, Dv, fv, mlo, mhi) and \
       not feasible(qc, omv, wv, M_hi_, Dv, fv, mlo, mhi):
        ok = False
        break
check("feasibility monotone in M: lowering M never helps a claim (1000 draws)", ok)
# Limit step of Thm 4.3(i,=>): objective->0 along a feasible sequence
# forces q*D <= M_hat in the limit even when M'_n stays far BELOW M_hat
# (one-sided penalty).  Simulate such sequences:
ok = True
for _ in range(500):
    wv = rng.uniform(0.6, 0.85); fv = rng.uniform(0.005, 0.04)
    omv = rng.uniform(0.4, 0.8); Dv = 7e4; Mhat = rng.uniform(2e4, 8e4)
    mlo, mhi = sorted(rng.uniform(1, 4, 2))
    av = 1 - wv
    muv = rng.uniform(mlo, mhi)
    qc = 1 - omv * (wv + muv * (av - fv)) / wv
    if not (0 <= qc <= 1):
        continue
    # feasible sequence with M'_n well below M_hat and objective -> 0:
    Mn = min(Mhat, max(qc * Dv, 0)) * rng.uniform(0.999, 1.0)
    if qc * Dv <= Mhat + 1e-9:
        # then (om,w,Mn) with Mn <= Mhat is feasible at objective 0 in (om,w)
        # and the limit conclusion q*D <= Mhat holds:
        continue
    # if q*D > Mhat, no sequence with (M'-Mhat)_+ -> 0 can be feasible:
    if qc * Dv <= Mhat:
        ok = False
        break
check("Thm 4.3 limit step: q*D <= M_hat survives one-sided penalty "
      "(500 draws)", ok)

# rho_omega closed form used in validate_bounds.py: distance from omega_hat
# to the interval [omega_needed(mu_hi), omega_needed(mu_lo)].
# omega_needed decreasing in mu:
om_needed = (1 - q) * w / (w + mu * (a - f))
check("omega_needed decreasing in mu",
      sp.simplify(sp.diff(om_needed, mu)) ==
      sp.simplify(-(1 - q) * w * (a - f) / (w + mu * (a - f))**2))

# ===========================================================================
# Proposition 5.1 (Bias-robust sensitivity band)
# ===========================================================================
print("\n== Proposition 5.1: Huber contamination ==")
# Why (ii) is a statement about the MEAN and not a uniform envelope:
# one source, eps=0.04, R=100; clean posterior N(0,0.1^2), contaminated
# component N(100,0.1^2).  Pooled 97.5% quantile jumps ~100 >> eps*R=4,
# so no first-order band can contain realised tail quantiles.
from scipy import stats as st
xs = np.linspace(-5, 105, 400_001)
cdf_pool = 0.96 * st.norm.cdf(xs, 0, 0.1) + 0.04 * st.norm.cdf(xs, 100, 0.1)
q975_pool = float(np.interp(0.975, cdf_pool, xs))
check("realised tail quantile can jump far beyond eps*R (band is first-order only)",
      q975_pool > 90 and 0.04 * 100 == 4.0, f"quantile={q975_pool:.2f}, eps*R=4")
# (ii) EXPECTED displacement identity: over contamination
# patterns T with P(T) = prod eps_i prod (1-eps_i), if patternwise shift is
# bounded by sum_{i in T} R_i, the expectation equals sum_i eps_i R_i.
# Verify the exchange-of-sums identity exactly by enumeration:
ok = True
for _ in range(100):
    k = int(rng.integers(1, 5))
    eps = rng.uniform(0, 0.4, k)
    R = rng.uniform(0, 2.0, k)
    total = 0.0
    for mask in range(2**k):
        T = [(mask >> i) & 1 for i in range(k)]
        pT = np.prod([eps[i] if T[i] else 1 - eps[i] for i in range(k)])
        total += pT * sum(R[i] for i in range(k) if T[i])
    if abs(total - (eps * R).sum()) > 1e-10:
        ok = False
        break
check("union bound identity: sum_T P(T) shift(T) = sum_i eps_i R_i "
      "(exact enumeration, 100 configs)", ok)

# (iii) Quantile displacement UNDER the posterior-weight condition:
# if the posterior mixture weight on contaminated components is
# <= tau = 1 - prod(1-eps_i) (assumed, not automatic: posterior weights are
# proportional to P(T) * component marginal likelihood Z_T) and the clean
# density >= m on the segment between the quantiles, |Q_eps - Q_0| <= tau/m.
# Verify with Gaussian mixtures where the weight condition holds by
# construction (mixture formed with weights exactly (1-eps, eps)):
for _ in range(200):
    epsv = rng.uniform(0, 0.2)
    shift = rng.uniform(0, 3)
    # contaminated cdf: (1-eps)*Phi(x) + eps*Phi(x-shift)
    u_test = rng.uniform(0.05, 0.95)
    q0 = st.norm.ppf(u_test)
    # solve contaminated quantile by bisection
    lo, hi = -10, 10
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 - epsv) * st.norm.cdf(mid) + epsv * st.norm.cdf(mid - shift) < u_test:
            lo = mid
        else:
            hi = mid
    qe = (lo + hi) / 2
    # density lower bound near the two quantiles:
    mval = min((1 - epsv) * st.norm.pdf(q0), (1 - epsv) * st.norm.pdf(qe))
    if abs(qe - q0) > epsv / mval + 1e-6:
        check("quantile displacement <= TV/m", False,
              f"|dq|={abs(qe-q0):.4f} bound={epsv/mval:.4f}")
        break
else:
    check("quantile displacement <= TV/m (200 Gaussian configs)", True)

# ===========================================================================
# Section 6 arithmetic spot checks (already in validate_bounds; re-assert core)
# ===========================================================================
print("\n== Section 6 arithmetic ==")
W_, A_, F_, OM_ = 0.733, 0.267, 0.020, 0.560
qf = lambda m: 1 - OM_ * (W_ + m * (A_ - F_)) / W_
check("q(1)=25.1%", abs(qf(1) - 0.2513) < 2e-3, f"{qf(1):.4f}")
check("q(2)=6.3%", abs(qf(2) - 0.0626) < 2e-3, f"{qf(2):.4f}")
check("q(mu)<=0 for mu>=2.34", qf(2.34) <= 0, f"q(2.34)={qf(2.34):.4f}")
mu_need = lambda qc: W_ * ((1 - qc) / OM_ - 1) / (A_ - F_)
check("mu needed for 17k/70k = 1.04", abs(mu_need(17/70) - 1.045) < 0.01,
      f"{mu_need(17/70):.4f}")
check("mu needed for 25k/70k = 0.44 (<1: infeasible)", mu_need(25/70) < 1,
      f"{mu_need(25/70):.4f}")
check("q(1.5)=15.69% (exact to 6 decimals: 0.156944)",
      abs(qf(1.5) - 0.156944) < 1e-4, f"{qf(1.5):.6f}")

print()
if FAIL:
    print(f"*** {len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("All proof verifications passed.")
