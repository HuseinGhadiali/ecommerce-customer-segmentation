"""Charts for the README and notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from analytics import ROOT
from viz_style import BLUES, MUTED, SERIES, TEXT, TEXT_2, apply_style, finish

IMG = ROOT / "images"
apply_style()


def cohort_heatmap(m, path=IMG / "cohort_heatmap.png"):
    mat = m.drop(columns=["cohort_size", 0]).iloc[:, :12]
    fig, ax = plt.subplots(figsize=(10, 7.5))
    cmap = LinearSegmentedColormap.from_list("b", BLUES)
    cmap.set_bad("#f1f0ec")
    ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), cmap=cmap, aspect="auto", vmin=0, vmax=25)
    ax.set_xticks(range(mat.shape[1]), [f"M{c}" for c in mat.columns])
    ax.set_yticks(range(len(mat)), [f"{p.strftime('%b %Y')}  (n={n:,})" for p, n in zip(mat.index, m.cohort_size)], fontsize=8)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(mat.to_numpy(float)):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 15 else TEXT)
    finish(fig, ax, "Only ~1 in 5 customers orders again the next month",
           "% of each monthly cohort placing an order N months after their first", path)


def sale_vs_fullprice(ret, path=IMG / "sale_vs_fullprice_retention.png"):
    months = [1, 3, 6, 12]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (flag, label) in enumerate([(0, "Acquired at full price"), (1, "Acquired in a sale")]):
        y = ret.loc[flag, [f"repeat_{m}m_pct" for m in months]].to_numpy() / 100
        ax.plot(months, y, marker="o", markersize=7, color=SERIES[i], label=label)
        ax.annotate(f"{label}: {y[-1]:.0%}", (12, y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT)
    ax.set_xticks(months, ["1 month", "3 months", "6 months", "12 months"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0.5, 17)
    ax.set_ylabel("Ordered again within…")
    ax.legend(loc="lower right")
    finish(fig, ax, "Sale-acquired customers are 30% less likely to come back",
           "Repeat-purchase rate by how the customer was acquired", path)


def rto_chart(t, path=IMG / "rto_by_tier.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(t))
    ax.hlines(y, t.rto_rate_pct_Prepaid / 100, t.rto_rate_pct_COD / 100, color=MUTED, lw=2)
    ax.scatter(t.rto_rate_pct_Prepaid / 100, y, s=90, color=SERIES[0], label="Prepaid (UPI / card)", zorder=3)
    ax.scatter(t.rto_rate_pct_COD / 100, y, s=90, color=SERIES[1], label="Cash on Delivery", zorder=3)
    for yi, (c, s) in enumerate(zip(t.rto_rate_pct_COD, t.cod_share_pct)):
        ax.text(c / 100 + 0.012, yi, f"{c:.0f}% refused  ·  COD is {s:.0f}% of orders", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 0.62)
    ax.legend(loc="upper right")
    ax.set_xlabel("Return-to-origin (parcel refused at the door) rate")
    finish(fig, ax, "1 in 3 COD parcels to Tier-3 towns comes back unopened",
           "RTO rate by city tier and payment mode", path)


def segment_chart(s, path=IMG / "rfm_segments.png"):
    s = s.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(s))
    ax.barh(y + 0.2, s.customer_pct / 100, height=0.38, color=MUTED, label="% of customers")
    ax.barh(y - 0.2, s.revenue_pct / 100, height=0.38, color=SERIES[0], label="% of revenue")
    ax.set_yticks(y, s.index)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    finish(fig, ax, "Champions are 6.5% of customers but 19% of revenue",
           "RFM segments: share of customers vs share of lifetime revenue", path)


def ltv_cac_chart(t, path=IMG / "ltv_cac.png"):
    t = t.sort_values("ltv_to_cac")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = [SERIES[1] if v < 1 else SERIES[0] for v in t.ltv_to_cac]
    ax.barh(t.index, t.ltv_to_cac, color=colors, height=0.55)
    ax.axvline(1, color=TEXT_2, lw=1)
    for yi, (v, ltv, cac) in enumerate(zip(t.ltv_to_cac, t.ltv_12m_contribution, t.cac_inr)):
        ax.text(v + 0.1, yi, f"{v:.1f}x   (Rs {ltv:,.0f} vs CAC Rs {cac:,.0f})", va="center", fontsize=9, color=TEXT)
    ax.set_xlim(0, t.ltv_to_cac.max() * 1.6)
    ax.set_xlabel("12-month contribution ÷ acquisition cost   (vertical line = break-even, 1.0x)")
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "Influencer customers never pay back their acquisition cost",
           "LTV:CAC by acquisition channel (12-month contribution margin)", path)


def clv_lift_chart(lift, metrics, path=IMG / "clv_model_lift.png"):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = [SERIES[0] if d <= 2 else MUTED for d in lift.index.astype(int)]
    ax.bar(lift.index.astype(int), lift / 100, color=colors, width=0.6)
    ax.axhline(0.10, color=TEXT_2, lw=1)
    ax.text(10.45, 0.105, "random", fontsize=8.5, color=TEXT_2, ha="right")
    ax.set_xticks(range(1, 11), [f"D{d}" for d in range(1, 11)])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Customers ranked by predicted CLV (decile 1 = highest)")
    ax.set_ylabel("Share of actual next-180-day revenue")
    finish(fig, ax, f"Top 20% by predicted CLV deliver {metrics['top20_capture_pct']:.0f}% of future revenue",
           f"Gradient-boosting CLV model on a 30% holdout (naive 'past spend' baseline: {metrics['top20_capture_naive_pct']:.0f}%)", path)
