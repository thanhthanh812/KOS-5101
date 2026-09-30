"""
04_shrinkage_zscore.py
=======================
Reproduce Figure 4 of Gelman, Hill, Yajima (2012): the shrinkage of the
multilevel-model z-score relative to the classical z-score as a function of
the variance ratio sigma_theta^2 / sigma_y^2.

For a two-group comparison theta_j - theta_k in a hierarchical normal model:

    z_classical  =  (ybar_j - ybar_k) / (sqrt(2) * sigma_y)

    z_multilevel = z_classical  *  1 / sqrt(1 + sigma_y^2 / sigma_theta^2)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  Always <= 1.  Approaches 1 as sigma_theta -> infinity
                                  (no shrinkage).  Approaches 0 as sigma_theta -> 0
                                  (full shrinkage).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

vr = np.linspace(0, 3, 500)          # variance ratio  sigma_theta^2 / sigma_y^2
shrinkage = 1.0 / np.sqrt(1 + 1/np.maximum(vr, 1e-9))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(vr, shrinkage, color="C0", lw=2.5,
        label=r"$\dfrac{z\ \mathrm{(multilevel)}}{z\ \mathrm{(classical)}}"
              r"=\dfrac{1}{\sqrt{1+\sigma_y^2/\sigma_\theta^2}}$")
ax.axhline(1.0, color="0.4", lw=0.8, ls="--",
           label="no shrinkage (classical z)")
ax.axhline(0.0, color="0.4", lw=0.8)
ax.fill_between(vr, 0, shrinkage, color="C0", alpha=0.15)
ax.set_xlabel(r"variance ratio  $\sigma_\theta^2 / \sigma_y^2$")
ax.set_ylabel(r"shrinkage of the z-score for $\theta_j-\theta_k$")
ax.set_ylim(-0.05, 1.1)
ax.set_xlim(0, 3)
ax.set_title("Shrinkage of the z-score for a pairwise comparison\n"
             "(reproduction of Fig. 4 in Gelman, Hill, Yajima 2012)")
ax.text(0.05, 0.15, "complete\nshrinkage", fontsize=9, color="0.3")
ax.text(2.6,  1.02, "no\nshrinkage", fontsize=9, color="0.3", ha="right")
ax.legend(loc="lower right", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_shrinkage_zscore.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_shrinkage_zscore.png'}")
