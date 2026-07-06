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
| `claim2_inversion` | Theorem 3.3(a): the forward model inverts to `q(μ) = 1 − ω(w+μ(a−f))/w` |
| `claim2_monotone` | Theorem 3.3(b): `q(μ)` is antitone in `μ` |
| `claim3_qtrue` | Remark 3: `q_true = (S−ω)/(S−η)` under η-slack |
| `claim3_bias_form1/2/3` | Remark 3: the three equal closed forms of the slack bias |
| `claim3_bias_nonneg_of_qtrue` | Remark 3: bias ≥ 0 exactly under `S > 0`, `η ≥ 0`, `q_true ≥ 0` — ignoring the slack *understates* q |
| `claim3_bias_nonneg` | Remark 3: the same under the stronger hypotheses `η < S`, `ω ≤ S` |
| `claim3_spurious_chain_refuted` | Formal refutation of a superficially plausible bias chain that double-counts the `(S−η)⁻¹` correction |
| `claim4_manpower_bound` | Corollary 3.4: `D_M ≤ M ⇒ q ≤ M/D` |
| `claim7_feasibility_monotone_in_M` | Theorem 4.3 ingredient: lowering the manpower ceiling never helps a claim (soundness of the one-sided penalty) |
| `partition_exact` | Section 6: the demographic classes partition the population (`w + a = 1` exactly) |
| `q_at_1`, `q_at_1_bounds` | Section 6: `q(1) = 921/3665` exactly; rounds to 25.13% |
| `q_at_15_bounds`, `q_at_2_bounds` | Section 6: `q(1.5) ∈ (15.69%, 15.70%)`, `q(2) ∈ (6.25%, 6.26%)` |
| `q_grid_ordered`, `q_crossing` | Section 6: grid ordering and the `q = 0` crossing below `μ = 2.34` |
| `S_exceeds_eta` | Remark 3 condition: `S(μ) > η` throughout the reported grid |
| `bias_at_1_bounds`, `bias_at_15_bounds`, `bias_at_2_bounds` | Remark 3: exact η-slack bias at `μ = 1, 1.5, 2` (1.05, 0.74, 0.33 percentage points) |
| `bias_loose_uniform_bound`, `bias_grid_ordered` | Remark 3: the 1.4-pp uniform bound is loose and dominates every exact per-μ bias, which decreases in μ |

The application arithmetic is proved over `ℚ` (exact rationals), so there
is no floating-point in the verified statements.

## What is *not* formalised

The statistical results (delta-method variance, Gauss–Markov optimality,
the Huber-contamination sensitivity band, and the topological closure step
of Theorem 4.3) are verified symbolically and numerically in
`analysis/verify_proofs.py` and proved in `docs/proof_verification.pdf`,
but are not kernel-checked: they involve measure-theoretic machinery whose
formalisation cost is out of proportion to the algebra actually at risk.
The formal layer deliberately covers the claims on which the paper's
headline numbers rest.

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
