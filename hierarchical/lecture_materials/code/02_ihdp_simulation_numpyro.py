"""
02_ihdp_simulation_numpyro.py
==============================
Same IHDP-like multilevel analysis as 02_ihdp_simulation_stan.py, but with
NumPyro (JAX-based NUTS).  Useful when you want a fast pure-Python workflow
that does not require a separate C++ Stan toolchain.

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


HERE    = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def simulate_ihdp(J=8, n_per_site=80, mu_gamma=80.0, sigma_gamma=5.0,
                  mu_delta=8.0, sigma_delta=1.2, sigma_y=22.0, seed=20260527):
    rng = np.random.default_rng(seed)
    gamma_true = rng.normal(mu_gamma, sigma_gamma, size=J)
    delta_true = rng.normal(mu_delta, sigma_delta, size=J)
    rows = []
    for j in range(J):
        for _ in range(n_per_site):
            P = int(rng.integers(0, 2))
            y = gamma_true[j] + delta_true[j] * P + rng.normal(0, sigma_y)
            rows.append((j + 1, P, y))
    return (pd.DataFrame(rows, columns=["site", "P", "y"]),
            gamma_true, delta_true)


def classical_per_site(df, J):
    est, se = np.zeros(J), np.zeros(J)
    for j in range(1, J + 1):
        s = df[df["site"] == j]
        yt = s.loc[s["P"] == 1, "y"].values
        yc = s.loc[s["P"] == 0, "y"].values
        est[j-1] = yt.mean() - yc.mean()
        se[j-1]  = np.sqrt(yt.var(ddof=1)/len(yt) + yc.var(ddof=1)/len(yc))
    z = stats.norm.ppf(0.975)
    return est, se, est - z*se, est + z*se


def ihdp_model(site, P, y=None, J=8):
    mu_gamma     = numpyro.sample("mu_gamma",     dist.Normal(0., 100.))
    mu_delta     = numpyro.sample("mu_delta",     dist.Normal(0., 100.))
    sigma_gamma  = numpyro.sample("sigma_gamma",  dist.HalfNormal(50.))
    sigma_delta  = numpyro.sample("sigma_delta",  dist.HalfNormal(50.))
    sigma_y      = numpyro.sample("sigma_y",      dist.HalfNormal(50.))

    with numpyro.plate("site", J):
        gamma_raw = numpyro.sample("gamma_raw", dist.Normal(0., 1.))
        delta_raw = numpyro.sample("delta_raw", dist.Normal(0., 1.))
    gamma = numpyro.deterministic("gamma", mu_gamma + sigma_gamma * gamma_raw)
    delta = numpyro.deterministic("delta", mu_delta + sigma_delta * delta_raw)

    mu_i = gamma[site] + delta[site] * P
    numpyro.sample("y", dist.Normal(mu_i, sigma_y), obs=y)


if __name__ == "__main__":
    J = 8
    print("Simulating IHDP-like data...")
    df, gamma_true, delta_true = simulate_ihdp(J=J)

    est_cls, se_cls, lo_cls, hi_cls = classical_per_site(df, J)
    zb = stats.norm.ppf(1 - 0.05/(2*J))
    lo_b, hi_b = est_cls - zb*se_cls, est_cls + zb*se_cls

    site_idx = jnp.asarray(df["site"].values - 1)
    P_arr    = jnp.asarray(df["P"].values, dtype=jnp.float32)
    y_arr    = jnp.asarray(df["y"].values, dtype=jnp.float32)

    print("Running NumPyro NUTS...")
    rng_key = random.PRNGKey(42)
    kernel  = NUTS(ihdp_model, target_accept_prob=0.95)
    mcmc    = MCMC(kernel, num_warmup=1000, num_samples=1500, num_chains=2,
                   chain_method="sequential", progress_bar=False)
    mcmc.run(rng_key, site_idx, P_arr, y=y_arr, J=J)
    samples = mcmc.get_samples()
    delta_draws = np.asarray(samples["delta"])
    sd_d        = np.asarray(samples["sigma_delta"])
    mu_d        = np.asarray(samples["mu_delta"])

    pm     = delta_draws.mean(axis=0)
    lo_mlm = np.quantile(delta_draws, 0.025, axis=0)
    hi_mlm = np.quantile(delta_draws, 0.975, axis=0)

    print(f"\nPosterior of sigma_delta: median = {np.median(sd_d):.2f}, "
          f"95% CrI = [{np.quantile(sd_d,0.025):.2f}, "
          f"{np.quantile(sd_d,0.975):.2f}]")
    print(f"Posterior of mu_delta:    mean   = {mu_d.mean():.2f}, "
          f"SD = {mu_d.std():.2f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    x = np.arange(1, J + 1)
    panels = [
        ("(A) Classical OLS",            est_cls, lo_cls, hi_cls),
        ("(B) Classical + Bonferroni",   est_cls, lo_b,   hi_b),
        ("(C) Bayesian MLM (NumPyro)",   pm,      lo_mlm, hi_mlm),
    ]
    ymin = min(lo_b.min(), lo_cls.min(), lo_mlm.min()) - 2
    ymax = max(hi_b.max(), hi_cls.max(), hi_mlm.max()) + 2
    for ax, (title, pt, lo, hi) in zip(axes, panels):
        for j in range(J):
            ax.plot([x[j], x[j]], [lo[j], hi[j]], color="black", lw=1.5)
        ax.scatter(x, pt, color="white", edgecolor="black", s=46, zorder=3, lw=1.3)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axhline(est_cls.mean(), color="0.6", lw=0.6, ls="--")
        ax.set_xticks(x); ax.set_title(title); ax.set_xlabel("site")
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("treatment effect estimate")
    fig.suptitle("IHDP-like simulation -- NumPyro NUTS",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ihdp_three_panels_numpyro.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG_DIR / 'fig_ihdp_three_panels_numpyro.png'}")
    print("\nDone.")
