"""
01_eight_schools_stan.py     [PRIMARY MCMC sampler for this course]
================================================================
Eight Schools (Rubin 1981; Gelman, Hill, Yajima 2012, Table 1)
fit with Stan via cmdstanpy.

Requirements:
    pip install cmdstanpy numpy pandas matplotlib scipy
    python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"

The Stan model is in eight_schools_nc.stan (non-centered parameterization).
"""

from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from cmdstanpy import CmdStanModel

# ---------------------------------------------------------------------------
# Eight Schools data (Rubin 1981; Table 1 of Gelman, Hill, Yajima 2012)
# ---------------------------------------------------------------------------
SCHOOLS = list("ABCDEFGH")
y     = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
J = len(y)

HERE      = Path(__file__).resolve().parent
STAN_FILE = HERE / "eight_schools_nc.stan"
FIG_DIR   = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# (A) Per-school 95% CI:  y_j +/- 1.96 sigma_j
# ---------------------------------------------------------------------------
z = stats.norm.ppf(0.975)
ci_lo_A = y - z * sigma
ci_hi_A = y + z * sigma

# ---------------------------------------------------------------------------
# (B) Bonferroni correction
# ---------------------------------------------------------------------------
alpha_bonf = 0.05 / J
z_bonf = stats.norm.ppf(1 - alpha_bonf / 2)
ci_lo_B = y - z_bonf * sigma
ci_hi_B = y + z_bonf * sigma
print(f"Bonferroni critical z for J={J}: {z_bonf:.3f}   (vs 1.96 uncorrected)")

# ---------------------------------------------------------------------------
# (C) Bayesian hierarchical model with Stan (cmdstanpy NUTS)
# ---------------------------------------------------------------------------
print("\nCompiling Stan model and running NUTS...")
model = CmdStanModel(stan_file=str(STAN_FILE))
fit = model.sample(
    data=dict(J=J, y=y.tolist(), sigma=sigma.tolist()),
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    seed=20260527,
    show_progress=False,
    refresh=0,
)

print("\n=== Stan summary ===")
print(fit.summary().loc[["mu", "tau", *[f"theta[{j+1}]" for j in range(J)]]])

theta_draws = fit.stan_variable("theta")
mu_draws    = fit.stan_variable("mu")
tau_draws   = fit.stan_variable("tau")

post_mean = theta_draws.mean(axis=0)
post_sd   = theta_draws.std(axis=0)
ci_lo_C   = np.quantile(theta_draws, 0.025, axis=0)
ci_hi_C   = np.quantile(theta_draws, 0.975, axis=0)

print("\n=== Reproduction of Table 1 (Gelman, Hill, Yajima 2012) ===")
table = pd.DataFrame({
    "School":   SCHOOLS,
    "y_j":      y,
    "sigma_j":  sigma,
    "Bayes M":  np.round(post_mean, 1),
    "Bayes SD": np.round(post_sd, 1),
})
print(table.to_string(index=False))

print(f"\nPosterior of tau (between-school SD):")
print(f"  median = {np.median(tau_draws):.2f}, "
      f"95% CrI = [{np.quantile(tau_draws,0.025):.2f}, "
      f"{np.quantile(tau_draws,0.975):.2f}]")
print(f"Posterior of mu:  mean = {mu_draws.mean():.2f}, SD = {mu_draws.std():.2f}")

# ---------------------------------------------------------------------------
# Three-panel comparison figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
x = np.arange(1, J + 1)
panels = [
    ("(A) Classical (no pooling)",        ci_lo_A, ci_hi_A, y),
    ("(B) Classical + Bonferroni",        ci_lo_B, ci_hi_B, y),
    ("(C) Bayesian multilevel (Stan)",    ci_lo_C, ci_hi_C, post_mean),
]
for ax, (title, lo, hi, pt) in zip(axes, panels):
    for j in range(J):
        ax.plot([x[j], x[j]], [lo[j], hi[j]], color="black", lw=1.6)
    ax.scatter(x, pt, color="white", edgecolor="black", s=48, zorder=3, lw=1.4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhline(np.mean(y), color="0.6", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(SCHOOLS)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("School")
    ax.set_ylim(-50, 75)
axes[0].set_ylabel("Treatment effect (SAT-V points)")
fig.suptitle("Eight Schools (Rubin 1981) -- Stan NUTS",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_eight_schools_compare.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG_DIR / 'fig_eight_schools_compare.png'}")

# ---------------------------------------------------------------------------
# Shrinkage diagram and tau posterior
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(np.zeros(J), y, color="C0", s=70, zorder=3, label="raw $y_j$")
ax.scatter(np.ones(J),  post_mean, color="C3", s=70, zorder=3,
           label=r"posterior mean $\hat\theta_j$")
for j in range(J):
    ax.annotate("", xy=(0.97, post_mean[j]), xytext=(0.03, y[j]),
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))
    ax.text(-0.06, y[j], SCHOOLS[j], ha="right", va="center", fontsize=10)
ax.axhline(mu_draws.mean(), color="black", lw=1, ls="--",
           label=fr"posterior mean of $\mu$ = {mu_draws.mean():.1f}")
ax.set_xticks([0, 1])
ax.set_xticklabels(["raw\n(no pooling)", "Bayesian MLM\n(partial pooling)"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel("Treatment effect")
ax.set_title("Shrinkage in the Eight Schools model (Stan)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_eight_schools_shrinkage.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_eight_schools_shrinkage.png'}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(tau_draws, bins=60, density=True, color="C0", alpha=0.75, edgecolor="white")
ax.axvline(np.median(tau_draws), color="black", ls="--",
           label=fr"posterior median $\tau$ = {np.median(tau_draws):.1f}")
ax.set_xlabel(r"$\tau$  (between-school SD of TRUE effects)")
ax.set_ylabel("Posterior density")
ax.set_title(r"Posterior of $\tau$ (Stan):  data prefer little between-school variation")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_eight_schools_tau_posterior.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_eight_schools_tau_posterior.png'}")

# Pairwise comparisons
print("\n=== Pairwise comparisons (28 pairs) ===")
n_sig_cls = n_sig_bonf = n_sig_bayes = 0
for j in range(J):
    for k in range(j + 1, J):
        diff = y[j] - y[k]
        se = np.sqrt(sigma[j]**2 + sigma[k]**2)
        z_jk = diff / se
        if abs(z_jk) > z: n_sig_cls += 1
        if abs(z_jk) > z_bonf: n_sig_bonf += 1
        diff_draws = theta_draws[:, j] - theta_draws[:, k]
        q025, q975 = np.quantile(diff_draws, [0.025, 0.975])
        if q025 > 0 or q975 < 0: n_sig_bayes += 1

print(f"  Classical (uncorrected)    : {n_sig_cls}/28 significant")
print(f"  Classical + Bonferroni     : {n_sig_bonf}/28 significant")
print(f"  Bayesian (95% CrI excl. 0) : {n_sig_bayes}/28 significant")
print()
print("=> The Bayesian multilevel model is the most conservative here, not via")
print("   ad-hoc width inflation but because the data prefer tau near 0.")
print("   This is exactly Gelman, Hill, Yajima (2012)'s main point.")
