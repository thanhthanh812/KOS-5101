"""
01_eight_schools_numpyro.py
============================
Same Eight Schools (Rubin 1981) analysis as 01_eight_schools_stan.py but
written with NumPyro (JAX-based probabilistic programming, identical NUTS).

NumPyro is useful when you want:
  - very fast NUTS via JAX JIT compilation
  - no C++ toolchain dependency (works inside a vanilla Python environment)
  - easy GPU / TPU support

Requirements:
    pip install numpyro jax numpy pandas matplotlib scipy
"""

from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)

# ---------------------------------------------------------------------------
# Data (Rubin 1981; Table 1 of Gelman, Hill, Yajima 2012)
# ---------------------------------------------------------------------------
SCHOOLS = list("ABCDEFGH")
y     = jnp.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
sigma = jnp.array([15., 10., 16., 11.,  9., 11., 10., 18.])
J = len(y)

HERE    = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# NumPyro model -- non-centered parameterization
# ---------------------------------------------------------------------------
def eight_schools_model(sigma, y=None):
    mu  = numpyro.sample("mu",  dist.Normal(0., 10.))
    tau = numpyro.sample("tau", dist.HalfNormal(10.))
    with numpyro.plate("school", J):
        eta   = numpyro.sample("eta", dist.Normal(0., 1.))
        theta = numpyro.deterministic("theta", mu + tau * eta)
        numpyro.sample("y", dist.Normal(theta, sigma), obs=y)


# ---------------------------------------------------------------------------
# Classical analyses (same as in the Stan script)
# ---------------------------------------------------------------------------
y_np     = np.asarray(y)
sigma_np = np.asarray(sigma)

z = stats.norm.ppf(0.975)
ci_lo_A = y_np - z * sigma_np
ci_hi_A = y_np + z * sigma_np

alpha_bonf = 0.05 / J
z_bonf = stats.norm.ppf(1 - alpha_bonf / 2)
ci_lo_B = y_np - z_bonf * sigma_np
ci_hi_B = y_np + z_bonf * sigma_np

# ---------------------------------------------------------------------------
# Run NumPyro NUTS
# ---------------------------------------------------------------------------
print("Running NumPyro NUTS...")
rng_key = random.PRNGKey(20260527)
kernel  = NUTS(eight_schools_model, target_accept_prob=0.95)
mcmc    = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=4,
               chain_method="sequential", progress_bar=False)
mcmc.run(rng_key, sigma, y=y)

print("\n=== NumPyro summary ===")
mcmc.print_summary()

samples = mcmc.get_samples()
theta_draws = np.asarray(samples["theta"])      # (n_draws, J)
mu_draws    = np.asarray(samples["mu"])
tau_draws   = np.asarray(samples["tau"])

post_mean = theta_draws.mean(axis=0)
post_sd   = theta_draws.std(axis=0)
ci_lo_C   = np.quantile(theta_draws, 0.025, axis=0)
ci_hi_C   = np.quantile(theta_draws, 0.975, axis=0)

print("\n=== Reproduction of Table 1 ===")
print(pd.DataFrame({
    "School":   SCHOOLS,
    "y_j":      y_np,
    "sigma_j":  sigma_np,
    "Bayes M":  np.round(post_mean, 1),
    "Bayes SD": np.round(post_sd, 1),
}).to_string(index=False))
print(f"\nPosterior of tau:  median = {np.median(tau_draws):.2f}, "
      f"95% CrI = [{np.quantile(tau_draws,0.025):.2f}, "
      f"{np.quantile(tau_draws,0.975):.2f}]")

# Three-panel comparison
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
x = np.arange(1, J + 1)
panels = [
    ("(A) Classical (no pooling)",     ci_lo_A, ci_hi_A, y_np),
    ("(B) Classical + Bonferroni",     ci_lo_B, ci_hi_B, y_np),
    ("(C) Bayesian MLM (NumPyro)",     ci_lo_C, ci_hi_C, post_mean),
]
for ax, (title, lo, hi, pt) in zip(axes, panels):
    for j in range(J):
        ax.plot([x[j], x[j]], [lo[j], hi[j]], color="black", lw=1.6)
    ax.scatter(x, pt, color="white", edgecolor="black", s=48, zorder=3, lw=1.4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhline(float(np.mean(y_np)), color="0.6", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(SCHOOLS)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("School")
    ax.set_ylim(-50, 75)
axes[0].set_ylabel("Treatment effect (SAT-V points)")
fig.suptitle("Eight Schools -- NumPyro NUTS",
             y=1.02, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_eight_schools_compare_numpyro.png",
            dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {FIG_DIR / 'fig_eight_schools_compare_numpyro.png'}")
print("\nDone.")
