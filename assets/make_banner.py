"""Render assets/banner.png: the cohort-retention matrix as a glowing mosaic (python assets/make_banner.py)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analytics import cohort_matrix, load  # noqa: E402
from banner import make_banner  # noqa: E402

_, orders, _ = load()
m = cohort_matrix(orders).drop(columns=["cohort_size", 0]).iloc[:, :12].to_numpy(float)


def art(ax, acc):
    rows, cols = m.shape
    ax.set_xlim(-cols * 0.45, cols + 0.3)
    ax.set_ylim(rows + 0.5, -0.5)
    for i in range(rows):
        for j in range(cols):
            v = m[i, j]
            if np.isnan(v):
                continue
            a = min(v / 22, 1)
            color = acc[0] if a > 0.7 else acc[3] if a > 0.45 else acc[1]
            for grow, alpha in ((0.32, 0.05), (0.2, 0.1)):
                ax.add_patch(FancyBboxPatch((j - 0.4 - grow / 2, i - 0.4 - grow / 2), 0.8 + grow, 0.8 + grow,
                                            boxstyle="round,pad=0,rounding_size=0.2", color=color, alpha=alpha * a, lw=0))
            ax.add_patch(FancyBboxPatch((j - 0.38, i - 0.38), 0.76, 0.76, boxstyle="round,pad=0,rounding_size=0.18",
                                        color=color, alpha=0.25 + 0.7 * a, lw=0))
    ax.text(-cols * 0.05, rows + 0.3, "each tile = a monthly cohort × months since first order, brighter = more customers returning",
            color="white", alpha=0.55, fontsize=9, va="top")


make_banner(Path(__file__).parent / "banner.png", "Customer Segmentation & CLV",
            "RFM · cohorts · COD/RTO · LTV:CAC · predictive CLV",
            ["Python", "scikit-learn", "RFM + K-means", "Jupyter"], art, theme="neon", emoji_free_kicker="D2C e-commerce · 60K customers")
