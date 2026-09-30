"""
03_type_s_m_errors.py
=====================
Reproduce Figure 2 of Gelman, Hill, Yajima (2012) and illustrate Type S (sign)
and Type M (magnitude) errors -- the two ways a frequentist analysis can mislead
that classical "Type 1 error" thinking misses.

Type S error: claiming the wrong SIGN of an effect
Type M error: a published estimate is much LARGER in magnitude than the truth
              (because we only publish "significant" results)

Both are amplified by low-powered studies (large SE).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 2 of paper:  two sampling distributions, true effect = 0
# ---------------------------------------------------------------------------
x = np.linspace(-12, 12, 1000)
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(x, stats.norm.pdf(x, 0, 3), lw=2, color="C0",
        label="sampling distribution\n with large SD (SD = 3)")
ax.plot(x, stats.norm.pdf(x, 0, 1), lw=2, color="C3", ls="--",
        label="sampling distribution\n with small SD (SD = 1)")
ax.axvline(0, color="0.4", lw=0.8)
# shade the "significant" tails (|x| > 1.96 SD) for the underpowered study
sig_lo, sig_hi = -1.96*3, 1.96*3
x_neg = np.linspace(-12, sig_lo, 200)
x_pos = np.linspace(sig_hi, 12, 200)
ax.fill_between(x_neg, stats.norm.pdf(x_neg, 0, 3), color="C0", alpha=0.3,
                label="|estimate| > 1.96·SE\n(would be called 'significant')")
ax.fill_between(x_pos, stats.norm.pdf(x_pos, 0, 3), color="C0", alpha=0.3)
ax.set_xlabel("treatment effect estimate")
ax.set_ylabel("likelihood")
ax.set_title("Why low power yields LARGER estimates (Type M error)\n"
             "and may flip the sign (Type S error)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_type_sm_intuition.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_type_sm_intuition.png'}")


# ---------------------------------------------------------------------------
# Monte Carlo:  small true effect, varying SE
#   For each SE in a grid, simulate 50,000 studies, see how often a
#   "significant" result has wrong sign (Type S) and how much it inflates
#   the magnitude (Type M = |estimate|_significant / |true|)
# ---------------------------------------------------------------------------
def type_sm_simulation(true_effect=2.0, se_grid=None, n_sims=50000, seed=1):
    rng = np.random.default_rng(seed)
    if se_grid is None:
        se_grid = np.linspace(0.5, 8.0, 25)
    z_crit = 1.96
    type_s = np.zeros_like(se_grid)
    type_m = np.zeros_like(se_grid)
    power  = np.zeros_like(se_grid)
    for i, se in enumerate(se_grid):
        est = rng.normal(true_effect, se, size=n_sims)
        sig = np.abs(est) > z_crit * se
        if sig.sum() == 0:
            continue
        est_sig = est[sig]
        power[i] = sig.mean()
        # Type S: among significant, fraction with wrong sign
        type_s[i] = (np.sign(est_sig) != np.sign(true_effect)).mean()
        # Type M: average |estimate|/|true|  among significant results
        type_m[i] = (np.abs(est_sig) / abs(true_effect)).mean()
    return se_grid, type_s, type_m, power

se_grid, type_s, type_m, power = type_sm_simulation(true_effect=2.0)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
ax.plot(se_grid, power, color="C2", lw=2)
ax.set_xlabel("standard error of estimate")
ax.set_ylabel("Power = P(significant | $H_1$)")
ax.set_title("Power vs. SE\n(true effect = 2)")
ax.axhline(0.05, color="0.5", lw=0.7, ls=":")

ax = axes[1]
ax.plot(se_grid, type_s*100, color="C3", lw=2)
ax.set_xlabel("standard error of estimate")
ax.set_ylabel("Type S error (%)")
ax.set_title("Type S = wrong-sign rate\namong 'significant' results")

ax = axes[2]
ax.plot(se_grid, type_m, color="C0", lw=2)
ax.axhline(1.0, color="0.5", lw=0.7, ls=":")
ax.set_xlabel("standard error of estimate")
ax.set_ylabel("Type M = |est|/|true|\n  (among 'significant')")
ax.set_title("Type M = magnitude inflation\namong 'significant' results")

fig.suptitle("Gelman & Tuerlinckx (2000) -- the dangers of low-powered studies\n"
             "(true effect = 2; significance threshold = 1.96·SE)",
             y=1.05, fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_type_sm_simulation.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_type_sm_simulation.png'}")

# Print key numbers for the lecture
print("\n=== Key numbers (true effect = 2) ===")
print(f"{'SE':>5} {'Power':>8} {'Type S (%)':>12} {'Type M':>8}")
for i in [0, 5, 10, 15, 20, 24]:
    print(f"{se_grid[i]:>5.2f} {power[i]:>8.3f} "
          f"{type_s[i]*100:>12.2f} {type_m[i]:>8.2f}")

print("\n=> Lesson: when SE is large (low power),")
print("   - power approaches the size of the test (5%)")
print("   - but any 'significant' result is highly likely to be WRONG-SIGN")
print("     and to GREATLY OVERSTATE the true magnitude.")
print("   - This is what Gelman calls the 'statistical significance filter'.")
print("\nClassical multiple-comparisons corrections (Bonferroni, FDR) try to")
print("control Type 1 error but actively make Type S and Type M WORSE by")
print("reducing power further.  The Bayesian multilevel model attacks the")
print("real problem -- low information per group -- via partial pooling.")
print("\nDone.")
