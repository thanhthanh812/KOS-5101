"""
02_ihdp_simulation_stan.py     [PRIMARY MCMC sampler for this course]
======================================================================
IHDP-like 8-site simulation and Bayesian multilevel analysis via Stan
(cmdstanpy NUTS).  Reproduces Figure 1 of Gelman, Hill, Yajima (2012).

Requirements:
    pip install cmdstanpy numpy pandas matplotlib scipy
    python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"

The Stan model is in ihdp_multilevel.stan.
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

HERE      = Path(__file__).resolve().parent
STAN_FILE = HERE / "ihdp_multilevel.stan"
FIG_DIR   = HERE.parent / "figures"
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


def bonferroni(est, se, J):
    zb = stats.norm.ppf(1 - 0.05/(2*J))
    return est - zb*se, est + zb*se


def fit_stan(df, J, model):
    fit = model.sample(
        data=dict(
            N=len(df),
            J=J,
            site=df["site"].astype(int).tolist(),
            P=df["P"].astype(float).tolist(),
            y=df["y"].astype(float).tolist(),
        ),
        chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
        seed=42,
        show_progress=False,
        refresh=0,
    )
    delta_draws = fit.stan_variable("delta")            # (n_draws, J)
    sd_draws    = fit.stan_variable("sigma_delta")
    mu_draws    = fit.stan_variable("mu_delta")
    return (
        delta_draws.mean(axis=0),
        np.quantile(delta_draws, 0.025, axis=0),
        np.quantile(delta_draws, 0.975, axis=0),
        sd_draws, mu_draws, delta_draws,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    J = 8
    print("Simulating IHDP-like data...")
    df, gamma_true, delta_true = simulate_ihdp(J=J)

    print("\nTrue site-level treatment effects:")
    for j, d in enumerate(delta_true, 1):
        print(f"  site {j}: delta_true = {d:6.2f}")

    est_cls, se_cls, lo_cls, hi_cls = classical_per_site(df, J)
    lo_b, hi_b = bonferroni(est_cls, se_cls, J)

    print("\nCompiling and fitting Stan multilevel model...")
    model = CmdStanModel(stan_file=str(STAN_FILE))
    pm, lo_mlm, hi_mlm, sd_d, mu_d, dd = fit_stan(df, J, model)

    print(f"\nPosterior of sigma_delta: median = {np.median(sd_d):.2f}, "
          f"95% CrI = [{np.quantile(sd_d,0.025):.2f}, "
          f"{np.quantile(sd_d,0.975):.2f}]")
    print(f"Posterior of mu_delta:    mean   = {mu_d.mean():.2f}, "
          f"SD = {mu_d.std():.2f}")

    # Three-panel figure (Figure 1 reproduction)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    x = np.arange(1, J + 1)
    panels = [
        ("(A) Classical linear regression",  est_cls, lo_cls, hi_cls),
        ("(B) Classical + Bonferroni",        est_cls, lo_b,   hi_b),
        ("(C) Bayesian MLM (Stan)",           pm,      lo_mlm, hi_mlm),
    ]
    ymin = min(lo_b.min(), lo_cls.min(), lo_mlm.min()) - 2
    ymax = max(hi_b.max(), hi_cls.max(), hi_mlm.max()) + 2
    for ax, (title, pt, lo, hi) in zip(axes, panels):
        for j in range(J):
            ax.plot([x[j], x[j]], [lo[j], hi[j]], color="black", lw=1.5)
        ax.scatter(x, pt, color="white", edgecolor="black", s=46, zorder=3, lw=1.3)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axhline(est_cls.mean(), color="0.6", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_title(title)
        ax.set_xlabel("site")
        ax.set_ylim(ymin, ymax)
    axes[0].set_ylabel("treatment effect estimate")
    fig.suptitle("IHDP-like simulation -- Stan NUTS\n"
                 "(reproduction of Gelman, Hill, Yajima 2012, Figure 1)",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ihdp_three_panels.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG_DIR / 'fig_ihdp_three_panels.png'}")

    # tau posterior
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(sd_d, bins=60, density=True, color="C2", alpha=0.75, edgecolor="white")
    ax.axvline(np.median(sd_d), color="black", ls="--",
               label=fr"posterior median = {np.median(sd_d):.2f}")
    ax.axvline(delta_true.std(ddof=1), color="red", ls=":",
               label=fr"true SD = {delta_true.std(ddof=1):.2f}")
    ax.set_xlabel(r"$\sigma_\delta$ (between-site SD of true effects)")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Posterior of $\sigma_\delta$  (Stan, IHDP simulation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ihdp_tau_posterior.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_ihdp_tau_posterior.png'}")

    # shrinkage diagram
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(np.zeros(J), est_cls, color="C0", s=70, zorder=3, label="OLS estimate")
    ax.scatter(np.ones(J),  pm,     color="C3", s=70, zorder=3, label="posterior mean")
    for j in range(J):
        ax.annotate("", xy=(0.97, pm[j]), xytext=(0.03, est_cls[j]),
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))
        ax.text(-0.06, est_cls[j], f"site {j+1}", ha="right", va="center", fontsize=9)
    ax.axhline(mu_d.mean(), color="black", lw=1, ls="--",
               label=fr"posterior mean of $\mu_\delta$ = {mu_d.mean():.2f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["classical OLS\n(no pooling)", "Bayesian MLM (Stan)\n(partial pooling)"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("treatment effect")
    ax.set_title("Shrinkage in the IHDP multilevel model (Stan)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ihdp_shrinkage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_ihdp_shrinkage.png'}")

    # Save results
    np.savez(
        FIG_DIR / "ihdp_results.npz",
        est_cls=est_cls, se_cls=se_cls,
        lo_cls=lo_cls, hi_cls=hi_cls,
        lo_b=lo_b, hi_b=hi_b,
        pm_mean=pm, lo_mlm=lo_mlm, hi_mlm=hi_mlm,
        delta_true=delta_true,
        sd_draws=sd_d, mu_draws=mu_d,
    )
    print(f"Saved {FIG_DIR / 'ihdp_results.npz'}")
    print("\nDone.")
