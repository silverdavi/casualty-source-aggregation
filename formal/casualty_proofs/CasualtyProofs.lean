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

/-- Variant that derives eq. 2 from the raw denominator w + a - f and the
exact partition w + a = 1, rather than pre-substituting a = 1 - w. -/
theorem claim1_identification_partition
    (w a f q om : ℝ) (hw : w ≠ 0) (hpart : w + a = 1)
    (hden : w + a - f ≠ 0)
    (hmodel : om = (1 - q) * w / (w + a - f)) :
    q = 1 - om * (1 - f) / w := by
  have ha : a = 1 - w := by linarith
  subst ha
  rw [hmodel]
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

/-- The forward model and the inverted curve are equivalent (full iff). -/
theorem claim2_inversion_iff
    (w a f mu q om : ℝ) (hw : w ≠ 0) (hden : w + mu * (a - f) ≠ 0) :
    om = (1 - q) * (w / (w + mu * (a - f))) ↔
      q = 1 - om * (w + mu * (a - f)) / w := by
  constructor
  · exact claim2_inversion w a f mu q om hw hden
  · intro h
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

/-- Part (b), derivative form: d q(mu) / d mu = -om*(a-f)/w. -/
theorem claim2_deriv
    (w a f om mu : ℝ) :
    HasDerivAt (fun mu => 1 - om * (w + mu * (a - f)) / w)
      (-(om * (a - f) / w)) mu := by
  have h : HasDerivAt (fun mu : ℝ => 1 - om * (w + mu * (a - f)) / w)
      (0 - om * (0 + 1 * (a - f)) / w) mu := by
    apply HasDerivAt.sub (hasDerivAt_const _ _)
    apply HasDerivAt.div_const
    apply HasDerivAt.const_mul
    apply HasDerivAt.const_add
    simpa using (hasDerivAt_id mu).mul_const (a - f)
  convert h using 1; ring

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

/-- The bias is nonnegative exactly under S > 0, eta ≥ 0, q_true ≥ 0
(via form 2: bias = eta * q_true / S): ignoring the slack UNDERSTATES q.
No hypothesis om ≤ S is needed -- that route would additionally require
S ≥ eta, since om = S - q(S-eta) gives om ≥ S when eta > S and q ≥ 0. -/
theorem claim3_bias_nonneg_of_qtrue
    (S eta om : ℝ) (hS : 0 < S) (heta : 0 ≤ eta) (hSeta : S - eta ≠ 0)
    (hq : 0 ≤ (S - om) / (S - eta)) :
    (0 : ℝ) ≤ (S - om) / (S - eta) - (S - om) / S := by
  rw [claim3_bias_form2 S eta om (ne_of_gt hS) hSeta]
  exact div_nonneg (mul_nonneg heta hq) hS.le

/-- Under the stronger hypotheses 0 ≤ eta < S and om ≤ S the bias is
likewise nonnegative (special case of the above). -/
theorem claim3_bias_nonneg
    (S eta om : ℝ) (hS : 0 < S) (heta : 0 ≤ eta) (hetaS : eta < S)
    (homS : om ≤ S) :
    (0 : ℝ) ≤ (S - om) / (S - eta) - (S - om) / S := by
  refine claim3_bias_nonneg_of_qtrue S eta om hS heta (by linarith) ?_
  exact div_nonneg (by linarith) (by linarith)

/-- The superficially plausible chain eta*q_true/S * S/(S-eta) is NOT the
bias (it double-counts the (S-eta)⁻¹ correction): concrete refutation at
S = 1, eta = 1/2, om = 0 (chain gives 2, bias is 1). -/
theorem claim3_spurious_chain_refuted :
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

/-! ## Claim 5: delta-method partial derivatives (paper Section 3)
The three gradients of g(om, f, w) = 1 - om*(1-f)/w that assemble the
delta-method variance formula. -/

/-- dg/df = om/w. -/
theorem claim5_grad_f (om w f : ℝ) :
    HasDerivAt (fun f => 1 - om * (1 - f) / w) (om / w) f := by
  have hf : HasDerivAt (fun f : ℝ => (1 : ℝ) - f) (-1) f :=
    (hasDerivAt_id f).const_sub 1
  have h := ((hf.const_mul om).div_const w).const_sub 1
  rw [show om / w = -(om * -1 / w) from by ring]
  exact h

/-- dg/dom = -(1-f)/w. -/
theorem claim5_grad_om (om w f : ℝ) :
    HasDerivAt (fun om => 1 - om * (1 - f) / w) (-(1 - f) / w) om := by
  have h := (((hasDerivAt_id om).mul_const (1 - f)).div_const w).const_sub 1
  rw [show -(1 - f) / w = -(1 * (1 - f) / w) from by ring]
  exact h

/-- dg/dw = om*(1-f)/w^2. -/
theorem claim5_grad_w (om w f : ℝ) (hw : w ≠ 0) :
    HasDerivAt (fun w => 1 - om * (1 - f) / w) (om * (1 - f) / w ^ 2) w := by
  have h1 : HasDerivAt (fun w : ℝ => om * (1 - f) / w)
      (om * (1 - f) * (-(w ^ 2)⁻¹)) w := by
    have := (hasDerivAt_inv hw).const_mul (om * (1 - f))
    simpa [div_eq_mul_inv, mul_comm, mul_assoc] using this
  have h2 := h1.const_sub 1
  rw [show om * (1 - f) / w ^ 2 = -(om * (1 - f) * (-(w ^ 2)⁻¹)) from by
    field_simp]
  exact h2

/-! ## Claim 6: precision-weighted aggregation (paper Proposition 2.1)
Gauss-Markov in one dimension: inverse-variance weights are unbiased and
minimum-variance among linear unbiased combinations. -/

/-- The precision weights c_i ∝ 1/σ_i² sum to one (unbiasedness). -/
theorem claim6_weights_sum {ι : Type*}
    (s : Finset ι) (σ2 : ι → ℝ)
    (hne : (∑ j ∈ s, 1 / σ2 j) ≠ 0) :
    (∑ i ∈ s, (1 / σ2 i) / (∑ j ∈ s, 1 / σ2 j)) = 1 := by
  rw [← Finset.sum_div, div_self hne]

/-- Minimum-variance optimality via Cauchy-Schwarz: any unbiased linear
combination ∑ c_i = 1 has variance ∑ c_i² σ_i² ≥ 1/(∑ 1/σ_i²), the value
attained by the precision weights. -/
theorem claim6_optimality {ι : Type*}
    (s : Finset ι) (σ2 c : ι → ℝ) (hpos : ∀ i ∈ s, 0 < σ2 i)
    (hsum : ∑ i ∈ s, c i = 1) (hne : 0 < ∑ j ∈ s, 1 / σ2 j) :
    1 / (∑ j ∈ s, 1 / σ2 j) ≤ ∑ i ∈ s, (c i) ^ 2 * σ2 i := by
  have key := Finset.sum_mul_sq_le_sq_mul_sq s
    (fun i => c i * Real.sqrt (σ2 i)) (fun i => 1 / Real.sqrt (σ2 i))
  have e1 : (∑ i ∈ s, (c i * Real.sqrt (σ2 i)) * (1 / Real.sqrt (σ2 i)))
      = ∑ i ∈ s, c i := by
    apply Finset.sum_congr rfl
    intro i hi
    have hsq : Real.sqrt (σ2 i) ≠ 0 := by have := hpos i hi; positivity
    field_simp
  have e2 : (∑ i ∈ s, (c i * Real.sqrt (σ2 i)) ^ 2)
      = ∑ i ∈ s, (c i) ^ 2 * σ2 i := by
    apply Finset.sum_congr rfl
    intro i hi
    have := hpos i hi
    rw [mul_pow, Real.sq_sqrt this.le]
  have e3 : (∑ i ∈ s, (1 / Real.sqrt (σ2 i)) ^ 2) = ∑ i ∈ s, 1 / σ2 i := by
    apply Finset.sum_congr rfl
    intro i hi
    have := hpos i hi
    rw [div_pow, one_pow, Real.sq_sqrt this.le]
  rw [e1, e2, e3, hsum] at key
  rw [div_le_iff₀ hne]
  nlinarith [key]

/-! ## Claim 7 ingredient: feasibility is monotone in M
(lowering the manpower ceiling never helps a claim), the fact that makes
the one-sided M-penalty in the contradiction radius sound. -/

theorem claim7_feasibility_monotone_in_M
    (q D M M' : ℝ) (hMM' : M ≤ M') (h : q * D ≤ M) :
    q * D ≤ M' := le_trans h hMM'

/-- The "needed omega" for a fixed target q ≤ 1,
omega_needed(mu) = (1-q)w/(w+mu(a-f)), is antitone in mu -- the fact
justifying the radius reduction used in analysis/validate_bounds.py. -/
theorem claim7_omega_needed_antitone
    (w a f q : ℝ) (hw : 0 < w) (haf : 0 < a - f) (hq : q ≤ 1)
    (mu1 mu2 : ℝ) (hmu : 0 ≤ mu1) (h : mu1 ≤ mu2) :
    (1 - q) * w / (w + mu2 * (a - f)) ≤ (1 - q) * w / (w + mu1 * (a - f)) := by
  have h1q : 0 ≤ 1 - q := by linarith
  have hd1 : 0 < w + mu1 * (a - f) := by nlinarith
  gcongr

/-- Contradiction radius, part (ii), abstract form: the radius
ρ = inf of the penalty over the rationalisation set C is a lower bound,
so every rationalisation has penalty ≥ ρ. -/
theorem claim7_radius_lower
    {α : Type*} (g : α → ℝ) (C : Set α)
    (hbdd : ∀ p ∈ C, 0 ≤ g p) :
    ∀ p ∈ C, sInf (g '' C) ≤ g p := by
  intro p hp
  apply csInf_le
  · exact ⟨0, by rintro _ ⟨x, hx, rfl⟩; exact hbdd x hx⟩
  · exact ⟨p, hp, rfl⟩

/-- Contradiction radius, part (i, ⇐), abstract form: if the face-value
point is feasible (penalty 0), the radius is 0. -/
theorem claim7_radius_zero_of_feasible
    {α : Type*} (g : α → ℝ) (C : Set α)
    (hbdd : ∀ p ∈ C, 0 ≤ g p) (p0 : α) (hp0 : p0 ∈ C) (hg0 : g p0 = 0) :
    sInf (g '' C) = 0 := by
  have hbd : BddBelow (g '' C) := ⟨0, by rintro _ ⟨x, hx, rfl⟩; exact hbdd x hx⟩
  have hmem : g p0 ∈ g '' C := ⟨p0, hp0, rfl⟩
  apply le_antisymm
  · calc sInf (g '' C) ≤ g p0 := csInf_le hbd hmem
       _ = 0 := hg0
  · apply le_csInf ⟨g p0, hmem⟩
    rintro _ ⟨x, hx, rfl⟩; exact hbdd x hx

/-! ## Claim 8 kernel: why Proposition 5.1(ii) is a first-order band
in expectation and NOT a uniform quantile envelope.
A single source contaminated with probability ε by a value displaced by R:
the displacement is 0 with probability 1-ε and R with probability ε. -/

/-- CDF of the two-point displacement distribution. -/
noncomputable def contamCDF (eps R x : ℝ) : ℝ :=
    (if (0 : ℝ) ≤ x then 1 - eps else 0) + (if R ≤ x then eps else 0)

/-- The MEAN displacement equals ε·R: the first-order sensitivity band is
exact in expectation. -/
theorem claim8_mean_displacement (eps R : ℝ) :
    (1 - eps) * 0 + eps * R = eps * R := by ring

/-- Below R the CDF stays ≤ 1-ε, so for any level u > 1-ε (e.g. u = 0.975
with ε = 0.04) the u-quantile is ≥ R: the realised tail quantile moves by
the full R, not by ε·R. -/
theorem claim8_quantile_stuck_below
    (eps R : ℝ) (heps : eps ≤ 1) (x : ℝ) (hx : x < R) :
    contamCDF eps R x ≤ 1 - eps := by
  unfold contamCDF
  rw [if_neg (not_le.mpr hx), add_zero]
  split <;> linarith

/-- At x = R the CDF reaches 1, so the quantile equals R exactly. -/
theorem claim8_quantile_at_R (eps R : ℝ) (hR : 0 ≤ R) :
    contamCDF eps R R = 1 := by
  unfold contamCDF
  rw [if_pos hR, if_pos le_rfl]
  ring

/-- Quantitatively at the dossier's numbers: the quantile shift R = 100 far
exceeds the first-order band ε·R = 4. -/
theorem claim8_band_not_envelope : (0.04 : ℝ) * 100 < 100 := by norm_num

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

/-- q(3/2) rounds to 15.69%: 0.1569 < q(3/2) < 0.1570
(in particular q(3/2) < 0.1572, refuting any 15.72% reading). -/
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

/-- The exact root of q(mu) = 0. -/
def muStar : ℚ := w0 * (1 - om0) / (om0 * (a0 - f0))

theorem q_root_exact : qmu muStar = 0 := by
  norm_num [qmu, muStar, w0, a0, f0, om0]

/-- The root rounds to 2.33. -/
theorem q_root_bounds : 233 / 100 < muStar ∧ muStar < 234 / 100 := by
  constructor <;> norm_num [muStar, w0, a0, f0, om0]

/-- Exposure needed to rationalise a target combatant share q. -/
def muNeeded (q : ℚ) : ℚ := ((1 - q) * w0 / om0 - w0) / (a0 - f0)

/-- muNeeded inverts qmu: q(muNeeded(q)) = q. -/
theorem muNeeded_inverts (q : ℚ) : qmu (muNeeded q) = q := by
  norm_num [qmu, muNeeded, w0, a0, f0, om0]
  ring

/-- The 17k/70k claim endpoint needs mu ≈ 1.04 ≥ 1: arithmetically
feasible under agnostic exposure. -/
theorem mu_needed_17k : 1 ≤ muNeeded (17 / 70) ∧
    104 / 100 < muNeeded (17 / 70) ∧ muNeeded (17 / 70) < 105 / 100 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [muNeeded, w0, a0, f0, om0]

/-- The 25k/70k claim endpoint needs mu ≈ 0.44 < 1: infeasible for every
admissible mu ≥ 1. -/
theorem mu_needed_25k : muNeeded (25 / 70) < 1 ∧
    43 / 100 < muNeeded (25 / 70) ∧ muNeeded (25 / 70) < 44 / 100 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [muNeeded, w0, a0, f0, om0]

/-- The manpower bound M/D = 0.643 is non-binding: it exceeds q(1). -/
theorem manpower_nonbinding : qmu 1 < 643 / 1000 := by
  norm_num [qmu, w0, a0, f0, om0]

/-! ### Remark 3 eta-slack bias, exact per-mu values
bias(mu) = eta * q_{eta=0}(mu) / (S(mu) - eta) with eta = 0.03 and
S(mu) = w/(w + mu*(a-f)); note qmu mu = (S - om)/S at S = S0 mu. -/

def eta0 : ℚ := 3 / 100
def S0 (mu : ℚ) : ℚ := w0 / (w0 + mu * (a0 - f0))
def bias0 (mu : ℚ) : ℚ := eta0 * qmu mu / (S0 mu - eta0)

/-- S > eta throughout the reported grid (the condition under which the
slack bound is stated). -/
theorem S_exceeds_eta : eta0 < S0 2 ∧ S0 2 < S0 (3 / 2) ∧ S0 (3 / 2) < S0 1 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [S0, eta0, w0, a0, f0]

/-- Exact bias at mu = 1 rounds to 1.05 percentage points. -/
theorem bias_at_1_bounds : 105 / 10000 < bias0 1 ∧ bias0 1 < 106 / 10000 := by
  constructor <;> norm_num [bias0, qmu, S0, eta0, w0, a0, f0, om0]

/-- Exact bias at mu = 3/2 rounds to 0.74 percentage points. -/
theorem bias_at_15_bounds :
    74 / 10000 < bias0 (3 / 2) ∧ bias0 (3 / 2) < 75 / 10000 := by
  constructor <;> norm_num [bias0, qmu, S0, eta0, w0, a0, f0, om0]

/-- Exact bias at mu = 2 rounds to 0.33 percentage points. -/
theorem bias_at_2_bounds : 33 / 10000 < bias0 2 ∧ bias0 2 < 34 / 10000 := by
  constructor <;> norm_num [bias0, qmu, S0, eta0, w0, a0, f0, om0]

/-- The loose uniform bound 1.4 pp dominates every exact per-mu bias:
each bias is below eta * 0.26 / (S(2) - eta) < 0.014. -/
theorem bias_loose_uniform_bound :
    bias0 1 < 14 / 1000 ∧ bias0 (3 / 2) < 14 / 1000 ∧ bias0 2 < 14 / 1000 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [bias0, qmu, S0, eta0, w0, a0, f0, om0]

/-- The exact bias decreases along the grid (q_{eta=0} falls faster
than S). -/
theorem bias_grid_ordered : bias0 2 < bias0 (3 / 2) ∧ bias0 (3 / 2) < bias0 1 := by
  constructor <;> norm_num [bias0, qmu, S0, eta0, w0, a0, f0, om0]

end Arithmetic

end CasualtyProofs
