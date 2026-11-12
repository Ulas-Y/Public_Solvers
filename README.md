# 🧠 Coupled Nonlinear Cognitive Dynamics

### *(Or: How to Accidentally Reinvent Bigeometric Calculus From Scratch While Catching a D on Your Midterm)*

![Status](https://img.shields.io/badge/GPA-Structurally_Compromised-red?style=flat-square)
![Propellant](https://img.shields.io/badge/Primary_Fuel-Weaponized_Spite-blueviolet?style=flat-square)
![Scope Creep](https://img.shields.io/badge/Scope_Creep-Exponential-orange?style=flat-square)
![Numerical Drift](https://img.shields.io/badge/Dimensionless_Drift-0.0e0-brightgreen?style=flat-square)

A log-space Euler solver for a deeply non-standard learning function $f(\text{effort}) = \text{result}$ where **syllabus compliance is zero**, **bandwidth is self-amplifying**, and **scope creep is an unpatched exponential runaway**.

If your brain treats a standard 3-credit university course like a side-quest while burning 72 sleepless hours deriving first-principles research out of pure pettiness, congratulations: you are not broken, you just have a poorly damped coupled ODE system.

---

## ⚠️ The Pathology (Why This Exists)

Linear textbook grinders operate on predictable proportional control: study $x$ hours, absorb $\Delta L$ facts, pass exam.

This repository models the exact opposite architecture—a **high-activation-energy, first-principles synthesis engine**:

1. Sees a basic operation (e.g., standard multiplication).
2. Gets annoyed by its inefficiency during exam week.
3. Ignores the actual test to derive non-Newtonian operator calculus from bare axioms.
4. Fails the multiple-choice rubric because 85% of accumulated knowledge $L$ now resides in ungraded frontier territory ($I - I_0$).

---

## 📐 Governing Equations

### 1. Dimensional State Space

Let $L(t)$ be retained knowledge, $I(t)$ be active problem scope, $V(t)$ be ingestion bandwidth, and $n(t) \in [0, 1]$ be single-pass retention efficiency, bounded by the irreducible axiomatic core $M$ and total domain ceiling $D$:

$$L_{\text{gain}} = \begin{cases} 0, & V < M \quad \text{(Ignition Barrier)} \\ \min(V, I) \cdot n \left(1 - \frac{L}{I}\right), & V \ge M \quad \text{(Coupon-Collector Percolation)} \end{cases}$$

$$L_{k+1} = (L_k + L_{\text{gain}}) e^{-\sigma \tau}$$

Meta-learning updates hardware capacity ($V$) and compiler efficiency ($n$) in log-space while scope creep ($\lambda$) actively tries to derail the entire simulation:

$$n_{k+1} = \min\!\left(1,\; n_k \exp\!\left(k \frac{L_{k+1}}{I_k} \tau\right)\right)$$

$$V_{k+1} = \min\!\left(D,\; V_k \exp\!\left(p \frac{L_{k+1}}{I_k D} \tau\right)\right)$$

$$I_{k+1} = \max\!\Big(M,\; \min\!\big(D,\; I_k e^{\lambda \tau}\big)\Big)$$

*(Note: For the exponent to remain dimensionless, $p$ carries units of $[\text{Info Volume}] \cdot [\text{Time}]^{-1}$. Yes, we checked.)*

### 2. Dimensionless Reduction ($[0, 1]^4$ Phase Space)

Normalizing against the domain ceiling $D$ via $u = L/D$, $v = V/D$, $\xi = I/D$, $\alpha = M/D$, and $\tilde{p} = p/D$ collapses the system onto the unit hypercube with zero floating-point drift ($\max \vert{}L/D - u\vert{} = 0.0$):

$$\tilde{p} \, u \, v \;\le\; \frac{dv}{dt} \;\le\; \frac{\tilde{p}}{\alpha} \, u \, v$$

Clamping scope to the axiomatic floor ($\xi \to \alpha$) yields a $1/\alpha$ compression amplifier—proving mathematically that **deriving everything from bare axioms is strictly faster than reading the textbook**.

---

## 🧪 Phase Regimes & Failure Modes

| Regime | Condition | Clinical Diagnosis |
| :--- | :--- | :--- |
| **Brick Mode** | $v < \alpha$ | Bandwidth is narrower than the minimum axiomatic core. $u_{\text{gain}} \equiv 0$. You stare at the page; the page stares back. |
| **Speedrun** | $1 \le v$ | Entire domain fits inside one ingestion window. You learn an entire programming language in an afternoon while randomly forgetting three basic syntax rules. |
| **The Rabbit-Hole Trap** | $\lambda \gg k$ | Scope explodes faster than meta-learning can keep up. You accidentally reconstruct cutting-edge literature from scratch and still feel like you know nothing ($u/\xi \to 0$). |
| **Steady-State Equilibrium** | $\xi \to 1, \; t \to \infty$ | Mastery locks onto $u^* = \frac{\min(v, 1)n}{e^{\sigma \tau} - 1 + \min(v, 1)n}$. You now own the domain and are legally obligated to start an unrelated side project. |

---

## 🚀 Quickstart

```python
import numpy as np
from solver import simulate_non_normalized, simulate_normalized

# Run both solvers with matched dimensionless scaling (p_tilde = p / D)
raw = simulate_non_normalized(steps=50, D=100.0, p=20.0, lam=0.08)
dimless = simulate_normalized(steps=50, p_tilde=0.20, lam=0.08)

# Verify exact scale invariance
assert np.max(np.abs(raw["L"] / 100.0 - dimless["u"])) == 0.0

print(f"Final Domain Mastery : {dimless['u'][-1]:.2%}")
print(f"Final Bandwidth Cap  : {dimless['v'][-1]:.2%}")
