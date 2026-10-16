# Machine-checked proofs (Lean 4 / mathlib)

Formal verification of the core algebraic claims in *Bounds and Estimators
for the Civilian-to-Combatant Death Ratio from Aggregation of Conflicting
Sources*. Every theorem in [`CasualtyProofs.lean`](CasualtyProofs.lean) is
checked by the Lean 4 kernel; the development contains **no `sorry`s**.

This is the strictest layer of the paper's three-level verification
pipeline:

1. `analysis/validate_bounds.py` — recomputes every applied number in the
   paper from raw inputs (fails on any discrepancy);
2. `analysis/verify_proofs.py` — sympy symbolic checks and Monte-Carlo
   stress tests of every theorem;
3. **this project** — kernel-checked proofs of the identification algebra,
   over exact rationals where numbers are involved.

## What is formalised

| Lean theorem | Paper claim |
|---|---|
| `claim1_benchmark_identification` | Eq. (2): benchmark identification `q = 1 − ω(1−f)/w` under equal hazard |
| `claim1_identification_partition` | Eq. (2) derived from the raw denominator and the exact partition `w + a = 1` |
| `claim2_inversion`, `claim2_inversion_iff` | Theorem 3.3(a): the forward model inverts to `q(μ) = 1 − ω(w+μ(a−f))/w` (full equivalence) |
| `claim2_monotone`, `claim2_deriv` | Theorem 3.3(b): `q(μ)` is antitone in `μ`, with `∂q/∂μ = −ω(a−f)/w` as a `HasDerivAt` statement |
| `claim3_qtrue` | Remark 3: `q_true = (S−ω)/(S−η)` under η-slack |
| `claim3_bias_form1/2/3` | Remark 3: the three equal closed forms of the slack bias |
| `claim3_bias_nonneg_of_qtrue` | Remark 3: bias ≥ 0 exactly under `S > 0`, `η ≥ 0`, `q_true ≥ 0` — ignoring the slack *understates* q |
| `claim3_bias_nonneg` | Remark 3: the same under the stronger hypotheses `η < S`, `ω ≤ S` |
| `claim3_spurious_chain_refuted` | Formal refutation of a superficially plausible bias chain that double-counts the `(S−η)⁻¹` correction |
| `claim4_manpower_bound` | Corollary 3.4: `D_M ≤ M ⇒ q ≤ M/D` |
| `claim5_grad_f`, `claim5_grad_om`, `claim5_grad_w` | Section 3: the three delta-method gradients of `g(ω,f,w) = 1 − ω(1−f)/w` |
| `claim6_weights_sum`, `claim6_optimality` | Proposition 2.1: precision weights sum to 1 and are minimum-variance among unbiased linear combinations (Cauchy–Schwarz) |
| `claim7_feasibility_monotone_in_M` | Theorem 4.3 ingredient: lowering the manpower ceiling never helps a claim (soundness of the one-sided penalty) |
| `claim7_omega_needed_antitone` | Theorem 4.3 / code reduction: the "needed ω" for a target claim is antitone in μ |
| `claim7_radius_lower`, `claim7_radius_zero_of_feasible` | Theorem 4.3 (i,⇐) and (ii) in abstract form: the radius is a lower bound on every rationalisation's penalty, and is 0 when the face-value point is feasible |
| `partition_exact` | Section 6: the demographic classes partition the population (`w + a = 1` exactly) |
| `q_at_1`, `q_at_1_bounds` | Section 6: `q(1) = 67511651/210130090` exactly; rounds to 32.13% |
| `q_at_15_bounds`, `q_at_2_bounds` | Section 6: `q(1.5) ∈ (24.34%, 24.35%)`, `q(2) ∈ (16.55%, 16.56%)` |
| `q_grid_ordered`, `q_crossing` | Section 6: grid ordering and the `q = 0` crossing below `μ = 3.07` |
| `muStar`, `q_root_exact`, `q_root_bounds` | Section 6: the exact root of `q(μ) = 0`, rounding to 3.06 |
| `muNeeded_inverts`, `mu_needed_17k`, `mu_needed_25k` | Section 6: exposure needed for the claim endpoints — `μ ≈ 1.54 ≥ 1` for 17k/71,444 (feasible), `μ ≈ 0.82 < 1` for 25k/71,444 (infeasible) |
| `manpower_nonbinding` | Section 6: `M/D = 0.630` exceeds `q(1)`, so the manpower bound never binds |
| `S_exceeds_eta` | Remark 3 condition: `S(μ) > η` throughout the reported grid |
| `bias_at_1_bounds`, `bias_at_15_bounds`, `bias_at_2_bounds` | Remark 3: exact η-slack bias at `μ = 1, 1.5, 2` (1.30, 1.10, 0.83 percentage points) |
| `bias_loose_uniform_bound`, `bias_grid_ordered` | Remark 3: the 1.7-pp uniform bound is loose and dominates every exact per-μ bias, which decreases in μ |

The application arithmetic is proved over `ℚ` (exact rationals), so there
is no floating-point in the verified statements.

## What is *not* formalised

The set-containment and multilinear-expansion statements of Proposition
5.1, and the topological limit step of Theorem 4.3(i,⇒) beyond its
abstract infimum and monotonicity ingredients, are verified symbolically
and numerically in `analysis/verify_proofs.py` and proved in
`docs/proof_verification.pdf`, but are not kernel-checked. The formal
layer covers every claim on which the paper's headline numbers rest,
including the delta-method gradients, Gauss–Markov optimality, and the
exact application arithmetic.

## Build

Requires [elan](https://github.com/leanprover/elan) (Lean toolchain
manager). Then:

```bash
cd formal/casualty_proofs
lake build            # fetches mathlib (cached), checks all proofs
```

A successful build *is* the verification: Lean's kernel accepts every
theorem or the build fails. To audit, confirm the absence of `sorry`/
`admit` in `CasualtyProofs.lean` and read the theorem statements — the
proofs themselves need not be trusted, only the statements and the kernel.

Toolchain: Lean 4 (pinned in `lean-toolchain`), mathlib v4.31.0 (pinned in
`lakefile.toml`).
