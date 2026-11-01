"""Synthetic order history for an Indian D2C beauty & personal-care brand (2024-2025).

Fully synthetic. Patterns built in so the analysis has something real to find:
* Cash-on-Delivery (COD) is far more common in Tier-2/3 towns, and COD parcels are often refused at the door
  (RTO, return-to-origin), which costs two-way shipping and usually loses the customer
* Customers acquired during big sale events (Diwali, End-of-Season) churn faster than full-price customers
* Influencer-acquired customers spend big on the first order but rarely come back; search/organic customers stay
* Skincare buyers replenish every ~6-8 weeks; makeup and fragrance buyers are far less regular

Usage: python src/generate_data.py --out data/raw --seed 21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")

SALES = [  # (name, start, end, signup multiplier, discount)
    ("Republic Day Sale", "2024-01-24", "2024-01-28", 2.0, 0.25), ("Anniversary Sale", "2024-03-08", "2024-03-12", 2.2, 0.30),
    ("End of Season Sale", "2024-06-20", "2024-06-30", 2.0, 0.35), ("Diwali Sale", "2024-10-20", "2024-11-03", 3.2, 0.40),
    ("End of Season Sale", "2024-12-20", "2024-12-31", 2.0, 0.35), ("Republic Day Sale", "2025-01-23", "2025-01-27", 2.0, 0.25),
    ("Anniversary Sale", "2025-03-08", "2025-03-12", 2.2, 0.30), ("End of Season Sale", "2025-06-20", "2025-06-30", 2.0, 0.35),
    ("Diwali Sale", "2025-10-10", "2025-10-24", 3.2, 0.40), ("End of Season Sale", "2025-12-20", "2025-12-31", 2.0, 0.35),
]
TIERS = {  # tier: (customer share, COD propensity, AOV multiplier, RTO multiplier)
    "Metro": (0.38, 0.26, 1.15, 0.8), "Tier 1": (0.27, 0.38, 1.0, 1.0),
    "Tier 2": (0.22, 0.55, 0.88, 1.2), "Tier 3": (0.13, 0.66, 0.80, 1.45),
}
CITIES = {
    "Metro": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"],
    "Tier 1": ["Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Indore", "Surat"],
    "Tier 2": ["Nagpur", "Bhopal", "Coimbatore", "Vadodara", "Visakhapatnam", "Patna", "Ludhiana", "Nashik", "Raipur"],
    "Tier 3": ["Siliguri", "Bhilai", "Jalandhar", "Udaipur", "Kota", "Hubballi", "Gorakhpur", "Bareilly", "Karnal"],
}
CHANNELS = {  # channel: (share, lifetime days, first AOV mult, repeat AOV mult)
    "Organic / Direct": (0.20, 420, 1.0, 1.0), "Google Search": (0.18, 360, 1.0, 1.0),
    "Instagram Ads": (0.26, 170, 0.95, 0.95), "Influencer": (0.14, 120, 1.35, 0.9),
    "Referral": (0.08, 480, 1.0, 1.05), "Marketplace Insert": (0.14, 220, 0.9, 0.9),
}
CATEGORIES = {  # category: (share, mean days between orders, base AOV Rs)
    "Skincare": (0.36, 48, 780), "Haircare": (0.20, 60, 620), "Makeup": (0.18, 95, 850),
    "Bath & Body": (0.12, 70, 520), "Fragrance": (0.08, 140, 1200), "Gift Sets": (0.06, 180, 1450),
}


def in_sale(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    d = pd.to_datetime(dates).to_numpy()
    disc = np.zeros(len(d))
    name = np.full(len(d), None, dtype=object)
    for n, a, b, _, ds in SALES:
        m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
        disc[m] = ds
        name[m] = n
    return disc, name


def generate(seed: int = 21, n_customers: int = 60000) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")
    w = np.linspace(0.7, 1.3, len(days))
    for _, a, b, mult, _ in SALES:
        w[(days >= a) & (days <= b)] *= mult
    signup = days[rng.choice(len(days), n_customers, p=w / w.sum())]

    tier = rng.choice(list(TIERS), n_customers, p=[v[0] for v in TIERS.values()])
    city = np.array([rng.choice(CITIES[t]) for t in tier])
    channel = rng.choice(list(CHANNELS), n_customers, p=[v[0] for v in CHANNELS.values()])
    category = rng.choice(list(CATEGORIES), n_customers, p=[v[0] for v in CATEGORIES.values()])
    _, sale_name = in_sale(pd.Series(signup))
    sale_acq = sale_name != None  # noqa: E711

    life = np.array([rng.exponential(CHANNELS[c][1]) for c in channel])
    life *= 0.45 * np.where(sale_acq, 0.55, 1.0)                # deal-seekers churn faster
    gap_mean = np.array([CATEGORIES[c][1] for c in category]) * rng.lognormal(0, 0.25, n_customers)
    base_aov = np.array([CATEGORIES[c][2] for c in category]) * np.array([TIERS[t][2] for t in tier]) \
        * rng.lognormal(0, 0.3, n_customers)
    cod_prop = np.clip(np.array([TIERS[t][1] for t in tier]) + rng.normal(0, 0.12, n_customers), 0.02, 0.95)

    customers = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n_customers + 1)], "signup_date": signup.date,
        "city": city, "city_tier": tier, "acquisition_channel": channel, "first_category": category,
        "acquired_in_sale": sale_acq.astype(int),
    })

    # ----- order timeline: first order at signup, then gamma-distributed gaps until churn or END -----
    K = 40
    gaps = rng.gamma(2.2, gap_mean[:, None] / 2.2, (n_customers, K))
    t = np.concatenate([np.zeros((n_customers, 1)), np.cumsum(gaps, 1)], 1)
    horizon = np.minimum(life, (END - signup).days.to_numpy())
    valid = t <= horizon[:, None]
    valid[:, 0] = True
    ci, k = np.nonzero(valid)
    orders = pd.DataFrame({"ci": ci, "seq": k + 1,
                           "order_date": signup.to_numpy()[ci] + (t[ci, k] * 86400e9).astype("timedelta64[ns]")})
    orders["order_date"] = orders.order_date.dt.normalize()
    n = len(orders)
    first = orders.seq.to_numpy() == 1
    ch = channel[ci]
    aov_mult = np.where(first, [CHANNELS[c][2] for c in ch], [CHANNELS[c][3] for c in ch])
    sale_disc, sale_nm = in_sale(orders.order_date)
    gross = base_aov[ci] * aov_mult * rng.lognormal(0, 0.28, n) * np.where(sale_disc > 0, 1.25, 1.0)  # stock up in sales
    coupon_disc = np.where(first & (sale_disc == 0), 0.20, np.where(rng.random(n) < 0.15, 0.10, 0.0))
    disc_rate = np.maximum(sale_disc, coupon_disc)
    loyal_prepaid = np.where(orders.seq.to_numpy() >= 3, 0.15, 0.0)
    cod = rng.random(n) < np.clip(cod_prop[ci] - loyal_prepaid, 0.02, 0.95)
    rto_p = np.where(cod, 0.21, 0.02) * np.array([TIERS[x][3] for x in tier[ci]]) * np.where(first, 1.25, 0.8)
    u = rng.random(n)
    status = np.where(u < rto_p, "RTO", np.where(u < rto_p + np.where(cod, 0.04, 0.02), "Cancelled",
                      np.where(u < rto_p + np.where(cod, 0.07, 0.06), "Returned", "Delivered")))
    orders["status"] = status
    # a refused / returned FIRST order usually ends the relationship
    bad_first = first & np.isin(status, ["RTO", "Returned"])
    lost = np.zeros(n_customers, dtype=bool)
    lost[ci[bad_first]] = rng.random(bad_first.sum()) < 0.7
    orders = orders[~(lost[ci] & ~first)].reset_index(drop=True)
    keep = ~(lost[ci] & ~first)
    ci, cod, gross, disc_rate, sale_nm, status = ci[keep], cod[keep], gross[keep], disc_rate[keep], sale_nm[keep], status[keep]
    n = len(orders)

    # category of each order: mostly the customer's main category, sometimes cross-sell
    cats = np.array(list(CATEGORIES))
    cross = rng.random(n) < 0.3
    order_cat = np.where(cross, rng.choice(cats, n, p=[v[0] for v in CATEGORIES.values()]), category[ci])
    items = 1 + rng.poisson(np.where(sale_nm != None, 1.4, 0.8))  # noqa: E711
    gross = np.round(gross / 10) * 10
    discount = np.round(gross * disc_rate)
    net = gross - discount
    shipping_fee_charged = np.where(net < 499, 49, 0)
    orders = orders.assign(
        order_id=[f"O{i:07d}" for i in range(1, n + 1)],
        customer_id=customers.customer_id.to_numpy()[ci],
        order_seq=orders.seq,
        category=order_cat, items=items, sale_event=sale_nm,
        payment_mode=np.where(cod, "COD", rng.choice(["UPI", "Card", "Wallet"], n, p=[0.72, 0.2, 0.08])),
        gross_value=gross, discount=discount, net_value=net, shipping_fee_charged=shipping_fee_charged,
    )
    orders = orders.sort_values(["order_date", "order_id"]).reset_index(drop=True)
    orders["order_id"] = [f"O{i:07d}" for i in range(1, len(orders) + 1)]
    cols = ["order_id", "customer_id", "order_date", "order_seq", "category", "items", "sale_event", "payment_mode",
            "status", "gross_value", "discount", "net_value", "shipping_fee_charged"]
    cac = pd.DataFrame({"acquisition_channel": list(CHANNELS),
                        "cac_inr": [90, 260, 420, 650, 180, 150]})        # planning assumptions, documented in README
    return {"customers": customers, "orders": orders[cols], "channel_cac": cac}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--customers", type=int, default=60000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, v in generate(a.seed, a.customers).items():
        v.to_csv(out / f"{k}.csv", index=False)
        print(f"{k:<12}{len(v):>9,} rows")


if __name__ == "__main__":
    main()
