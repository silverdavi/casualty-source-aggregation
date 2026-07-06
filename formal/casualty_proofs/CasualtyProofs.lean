/-
Formal (machine-checked) proofs of the core algebraic claims in
"Bounds and Estimators for the Civilian-to-Combatant Death Ratio
from Aggregation of Conflicting Sources".

Companion to docs/proof_verification.tex and analysis/verify_proofs.py.
Every theorem here is checked by the Lean 4 kernel; there are no
`sorry`s.  Build with `lake build` in this directory.

Notation (matching the paper):
  w  : population share of class W (women + male minors)
  a  : population share of class AM (males 18+), with w + a = 1
  f  : combatant stock / population
  mu : civilian adult-male exposure multiplier
  om : share of the dead in class W (omega)
  q  : combatant share of the dead
  S  : civilian share of deaths landing in W, S = w / (w + mu*(a-f))
  eta: fraction of combatant deaths landing in W (slack)
-/
import Mathlib

namespace CasualtyProofs

/-! ## Claim 1: benchmark identification (paper eq. 2)
If om = (1-q) * w/(1-f)  (the mu = 1 forward model with a = 1-w),
then q = 1 - om*(1-f)/w. -/

theorem claim1_benchmark_identification
    (w f q om : ℝ) (hw : w ≠ 0) (hf : (1 : ℝ) - f ≠ 0)
    (h : om = (1 - q) * (w / (1 - f))) :
    q = 1 - om * (1 - f) / w := by
  subst h
  field_simp
  ring

/-! ## Claim 2: sharp identified set (paper Theorem 3.3)
Part (a): the forward model om = (1-q) * w/(w + mu*(a-f)) inverts to
q(mu) = 1 - om*(w + mu*(a-f))/w. -/

theorem claim2_inversion
    (w a f mu q om : ℝ) (hw : w ≠ 0) (hden : w + mu * (a - f) ≠ 0)
    (h : om = (1 - q) * (w / (w + mu * (a - f)))) :
    q = 1 - om * (w + mu * (a - f)) / w := by
  subst h
  field_simp
  ring

/-- Part (b): q(mu) is antitone in mu when om ≥ 0, f ≤ a, w > 0. -/
theorem claim2_monotone
    (w a f om : ℝ) (hw : 0 < w) (hom : 0 ≤ om) (haf : f ≤ a)
    {mu1 mu2 : ℝ} (hmu : mu1 ≤ mu2) :
    1 - om * (w + mu2 * (a - f)) / w ≤ 1 - om * (w + mu1 * (a - f)) / w := by
  have h1 : om * (w + mu1 * (a - f)) ≤ om * (w + mu2 * (a - f)) := by
    nlinarith [mul_nonneg (mul_nonneg hom (sub_nonneg.mpr hmu))
      (sub_nonneg.mpr haf)]
  have h2 : om * (w + mu1 * (a - f)) / w ≤ om * (w + mu2 * (a - f)) / w := by
    gcongr
  linarith

/-! ## Claim 3: eta-slack (paper Remark 3)
With om = (1-q)S + eta*q: q_true = (S-om)/(S-eta), q0 = (S-om)/S,
and the bias admits three equal closed forms. -/

theorem claim3_qtrue
    (S eta q om : ℝ) (hSeta : S - eta ≠ 0)
    (h : om = (1 - q) * S + eta * q) :
    q = (S - om) / (S - eta) := by
  subst h
  field_simp
  ring

/-- Bias identity, form 1: q_true - q0 = eta*(S-om)/(S*(S-eta)). -/
theorem claim3_bias_form1
    (S eta om : ℝ) (hS : S ≠ 0) (hSeta : S - eta ≠ 0) :
    (S - om) / (S - eta) - (S - om) / S = eta * (S - om) / (S * (S - eta)) := by
  field_simp
  ring

/-- Bias identity, form 2: q_true - q0 = eta * q_true / S. -/
theorem claim3_bias_form2
    (S eta om : ℝ) (hS : S ≠ 0) (hSeta : S - eta ≠ 0) :
    (S - om) / (S - eta) - (S - om) / S = eta * ((S - om) / (S - eta)) / S := by
  field_simp
  ring

/-- Bias identity, form 3: q_true - q0 = eta * q0 / (S - eta). -/
theorem claim3_bias_form3
    (S eta om : ℝ) (hS : S ≠ 0) (hSeta : S - eta ≠ 0) :
    (S - om) / (S - eta) - (S - om) / S = eta * ((S - om) / S) / (S - eta) := by
  field_simp
  ring

/-- The bias is nonnegative when 0 ≤ eta < S and om ≤ S:
ignoring the slack UNDERSTATES q. -/
theorem claim3_bias_nonneg
    (S eta om : ℝ) (hS : 0 < S) (heta : 0 ≤ eta) (hetaS : eta < S)
    (homS : om ≤ S) :
    (0 : ℝ) ≤ (S - om) / (S - eta) - (S - om) / S := by
  rw [claim3_bias_form1 S eta om (ne_of_gt hS) (by linarith)]
  apply div_nonneg
  · nlinarith
  · nlinarith

/-- The OLD dossier chain eta*q_true/S * S/(S-eta) is NOT the bias:
concrete refutation at S = 1, eta = 1/2, om = 0 (chain gives 2, bias is 1). -/
theorem claim3_old_chain_wrong :
    ∃ S eta om : ℝ, S ≠ 0 ∧ S - eta ≠ 0 ∧
      eta * ((S - om) / (S - eta)) / S * (S / (S - eta)) ≠
        (S - om) / (S - eta) - (S - om) / S := by
  refine ⟨1, 1 / 2, 0, one_ne_zero, by norm_num, by norm_num⟩

/-! ## Claim 4: manpower bound (paper Corollary 3.4)
D_M ≤ M and D > 0 imply q = D_M/D ≤ M/D. -/

theorem claim4_manpower_bound
    (DM M D : ℝ) (hD : 0 < D) (h : DM ≤ M) :
    DM / D ≤ M / D := by
  gcongr

/-! ## Claim 7 ingredient: feasibility is monotone in M
(lowering the manpower ceiling never helps a claim), the fact that makes
the one-sided M-penalty in the contradiction radius sound. -/

theorem claim7_feasibility_monotone_in_M
    (q D M M' : ℝ) (hMM' : M ≤ M') (h : q * D ≤ M) :
    q * D ≤ M' := le_trans h hMM'

/-! ## Section 6 arithmetic, exactly over ℚ
q(mu) = 1 - om*(w + mu*(a-f))/w at
(w, a, f, om) = (0.733, 0.267, 0.020, 0.560). -/

section Arithmetic

def w0 : ℚ := 733 / 1000
def a0 : ℚ := 267 / 1000
def f0 : ℚ := 20 / 1000
def om0 : ℚ := 560 / 1000

def qmu (mu : ℚ) : ℚ := 1 - om0 * (w0 + mu * (a0 - f0)) / w0

/-- Classes partition the population exactly. -/
theorem partition_exact : w0 + a0 = 1 := by norm_num [w0, a0]

/-- q(1) = 25.1296...%: exact value 921/3665. -/
theorem q_at_1 : qmu 1 = 921 / 3665 := by
  norm_num [qmu, w0, a0, f0, om0]

/-- q(1) rounds to 25.13%: 0.2512 < q(1) < 0.2513. -/
theorem q_at_1_bounds : 2512 / 10000 < qmu 1 ∧ qmu 1 < 2513 / 10000 := by
  constructor <;> norm_num [qmu, w0, a0, f0, om0]

/-- q(3/2) < 0.157 and q(3/2) > 0.1569: the dossier's misprinted 15.72%
is refuted, the correct rounding is 15.69%. -/
theorem q_at_15_bounds :
    1569 / 10000 < qmu (3 / 2) ∧ qmu (3 / 2) < 1570 / 10000 := by
  constructor <;> norm_num [qmu, w0, a0, f0, om0]

/-- q(2) rounds to 6.26%. -/
theorem q_at_2_bounds : 625 / 10000 < qmu 2 ∧ qmu 2 < 626 / 10000 := by
  constructor <;> norm_num [qmu, w0, a0, f0, om0]

/-- The identified-set upper endpoint is decreasing on the grid. -/
theorem q_grid_ordered : qmu 2 < qmu (3 / 2) ∧ qmu (3 / 2) < qmu 1 := by
  constructor <;> norm_num [qmu, w0, a0, f0, om0]

/-- q(mu) ≤ 0 at mu = 2.34 (the q = 0 crossing is below 2.34). -/
theorem q_crossing : qmu (234 / 100) < 0 := by
  norm_num [qmu, w0, a0, f0, om0]

end Arithmetic

end CasualtyProofs
