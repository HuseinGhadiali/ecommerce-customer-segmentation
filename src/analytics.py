"""Customer analytics for a D2C brand: RFM, cohorts, clustering, COD/RTO economics, LTV:CAC and predictive CLV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = pd.Timestamp("2025-12-31")

# Unit-economics assumptions (stated so a reviewer can challenge them)
GROSS_MARGIN = 0.68           # product gross margin on net revenue, typical for D2C beauty
SHIP_COST = 75                # forward shipping per parcel (Rs)
COD_FEE = 35                  # courier COD handling fee (Rs)
RTO_COST = 75 + 60            # forward + reverse shipping on a refused parcel; stock is recovered (Rs)
PAYMENT_GATEWAY = 0.02        # prepaid gateway fee on net value


def load(raw: Path = ROOT / "data" / "raw"):
    customers = pd.read_csv(raw / "customers.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(raw / "orders.csv", parse_dates=["order_date"])
    cac = pd.read_csv(raw / "channel_cac.csv")
    return customers, orders, cac


def add_order_economics(orders: pd.DataFrame) -> pd.DataFrame:
    """Contribution margin per order after product cost, shipping, COD fees and RTO losses."""
    o = orders.copy()
    delivered = o.status == "Delivered"
    rto = o.status == "RTO"
    returned = o.status == "Returned"
    cod = o.payment_mode == "COD"
    o["revenue"] = np.where(delivered, o.net_value + o.shipping_fee_charged, 0.0)
    o["contribution"] = (
        np.where(delivered, o.net_value * GROSS_MARGIN + o.shipping_fee_charged - SHIP_COST
                 - np.where(cod, COD_FEE, o.net_value * PAYMENT_GATEWAY), 0.0)
        - np.where(rto, RTO_COST, 0.0)
        - np.where(returned, SHIP_COST * 2, 0.0)
    )
    return o


# ------------------------------------------------------------------ KPIs
def kpi_summary(customers, orders) -> pd.DataFrame:
    o = add_order_economics(orders)
    d = o[o.status == "Delivered"]
    by_cust = o.groupby("customer_id").size()
    rows = []
    for yr in (2024, 2025):
        oy, dy = o[o.order_date.dt.year == yr], d[d.order_date.dt.year == yr]
        rows.append({"year": yr, "orders": len(oy), "delivered_revenue_cr": dy.revenue.sum() / 1e7,
                     "aov": dy.net_value.mean(), "active_customers": oy.customer_id.nunique(),
                     "new_customers": (customers.signup_date.dt.year == yr).sum(),
                     "rto_rate_pct": 100 * (oy.status == "RTO").mean(), "cod_share_pct": 100 * (oy.payment_mode == "COD").mean(),
                     "contribution_cr": oy.contribution.sum() / 1e7})
    out = pd.DataFrame(rows).set_index("year").round(2)
    out.attrs["repeat_rate_pct"] = round(100 * (by_cust > 1).mean(), 1)
    return out


# ------------------------------------------------------------------ cohorts
def cohort_matrix(orders, max_months: int = 12) -> pd.DataFrame:
    o = orders[orders.status != "Cancelled"].copy()
    o["order_month"] = o.order_date.dt.to_period("M")
    o["cohort"] = o.groupby("customer_id").order_month.transform("min")
    o["age"] = (o.order_month - o.cohort).apply(lambda x: x.n)
    sizes = o[o.age == 0].groupby("cohort").customer_id.nunique()
    m = o[o.age <= max_months].groupby(["cohort", "age"]).customer_id.nunique().unstack()
    m = m.div(sizes, axis=0) * 100
    last = o.order_month.max()
    for c in m.index:                       # hide months that have not happened yet
        m.loc[c, [a for a in m.columns if c + a > last]] = np.nan
    m.insert(0, "cohort_size", sizes)
    return m.round(1)


def retention_by_group(orders, customers, group: str, months=(1, 3, 6, 12)) -> pd.DataFrame:
    """% of customers who ordered again within N months of their first order."""
    o = orders[orders.status != "Cancelled"].merge(customers[["customer_id", group]], on="customer_id")
    first = o.groupby("customer_id").order_date.min().rename("first")
    o = o.join(first, on="customer_id")
    o["days"] = (o.order_date - o["first"]).dt.days
    res = {}
    for m in months:
        eligible = first[first <= SNAPSHOT - pd.DateOffset(months=m)].index
        sub = o[o.customer_id.isin(eligible)]
        rep = sub[(sub.days > 0) & (sub.days <= m * 30.4)].customer_id.unique()
        g = customers.set_index("customer_id").loc[eligible, group]
        res[f"repeat_{m}m_pct"] = pd.Series(g.index.isin(rep).astype(float), index=g.index).groupby(g.to_numpy()).mean() * 100
    return pd.DataFrame(res).round(1)


# ------------------------------------------------------------------ RFM
SEGMENT_PLAYBOOK = {
    "Champions": "Early access to launches, referral programme, no discounts needed",
    "Loyal": "Bundles and subscriptions (skincare replenishment), loyalty points",
    "Potential Loyalist": "Second-purchase nudge at day 30-45 with a cross-sell from their category",
    "New": "Onboarding series: how-to content, review request, prepaid incentive",
    "At Risk": "Win-back: personalised offer on their last category, before day 120",
    "Can't Lose Them": "High-value lapsed: call / WhatsApp from CX, strongest offer",
    "Hibernating": "Low-cost channels only (email, WhatsApp broadcast) during sale events",
}


def rfm(orders, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    d = orders[orders.status == "Delivered"]
    r = d.groupby("customer_id").agg(last=("order_date", "max"), frequency=("order_id", "count"),
                                     monetary=("net_value", "sum"))
    r["recency_days"] = (snapshot - r["last"]).dt.days
    r["R"] = pd.qcut(r.recency_days.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # frequency is lumpy (most customers have 1 order), so score it on fixed thresholds
    r["F"] = pd.cut(r.frequency, [0, 1, 2, 3, 5, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    r["M"] = pd.qcut(r.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    conds = [
        (r.R >= 4) & (r.F >= 4),
        (r.R >= 3) & (r.F >= 3),
        (r.R >= 4) & (r.F == 2),
        (r.R >= 4) & (r.F == 1),
        (r.R <= 2) & (r.F >= 3) & (r.M >= 4),
        (r.R <= 3) & (r.F >= 2),
    ]
    names = ["Champions", "Loyal", "Potential Loyalist", "New", "Can't Lose Them", "At Risk"]
    r["segment"] = np.select(conds, names, "Hibernating")
    return r.drop(columns="last")


def segment_summary(r: pd.DataFrame) -> pd.DataFrame:
    s = r.groupby("segment").agg(customers=("R", "size"), avg_recency_days=("recency_days", "mean"),
                                 avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean"),
                                 revenue=("monetary", "sum"))
    s["customer_pct"] = 100 * s.customers / s.customers.sum()
    s["revenue_pct"] = 100 * s.revenue / s.revenue.sum()
    s["action"] = s.index.map(SEGMENT_PLAYBOOK)
    order = list(SEGMENT_PLAYBOOK)
    return s.loc[[x for x in order if x in s.index]].drop(columns="revenue").round(1)


def kmeans_segments(r: pd.DataFrame, k_range=range(3, 8), seed: int = 0):
    X = StandardScaler().fit_transform(np.log1p(r[["recency_days", "frequency", "monetary"]]))
    sample = np.random.default_rng(seed).choice(len(X), min(8000, len(X)), replace=False)
    scores = {k: silhouette_score(X[sample], KMeans(k, n_init=10, random_state=seed).fit_predict(X[sample])) for k in k_range}
    best = max(scores, key=scores.get)
    labels = KMeans(best, n_init=10, random_state=seed).fit_predict(X)
    prof = r.assign(cluster=labels).groupby("cluster").agg(
        customers=("R", "size"), recency_days=("recency_days", "median"), orders=("frequency", "mean"),
        spend=("monetary", "mean")).round(1)
    prof = prof.sort_values("spend", ascending=False)
    return pd.Series(scores, name="silhouette").round(3), prof, labels


# ------------------------------------------------------------------ COD / RTO
def rto_analysis(orders, customers) -> pd.DataFrame:
    o = orders.merge(customers[["customer_id", "city_tier"]], on="customer_id")
    o["pay"] = np.where(o.payment_mode == "COD", "COD", "Prepaid")
    t = o.groupby(["city_tier", "pay"]).agg(orders=("order_id", "size"), rto_rate_pct=("status", lambda s: 100 * (s == "RTO").mean()))
    t = t.unstack("pay")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["cod_share_pct"] = 100 * t.orders_COD / (t.orders_COD + t.orders_Prepaid)
    return t[["cod_share_pct", "rto_rate_pct_COD", "rto_rate_pct_Prepaid"]].round(1)


def prepaid_nudge_scenario(orders, shift: float = 0.30, incentive: float = 40.0) -> dict:
    """What if a Rs `incentive` prepaid discount moved `shift` of COD orders to prepaid?"""
    o = orders[orders.order_date.dt.year == 2025]
    cod = o[o.payment_mode == "COD"]
    pre = o[o.payment_mode != "COD"]
    rto_cod, rto_pre = (cod.status == "RTO").mean(), (pre.status == "RTO").mean()
    moved = len(cod) * shift
    rto_saved = moved * (rto_cod - rto_pre)
    delivered_moved = moved * (1 - rto_pre)
    saving = rto_saved * RTO_COST + delivered_moved * (COD_FEE - cod.net_value.mean() * PAYMENT_GATEWAY)
    # parcels that would have been refused now get delivered: extra contribution
    extra = rto_saved * (cod.net_value.mean() * GROSS_MARGIN - SHIP_COST)
    cost = delivered_moved * incentive
    return {"cod_orders_2025": len(cod), "orders_shifted": round(moved), "rto_avoided": round(rto_saved),
            "rto_rate_cod_pct": round(float(100 * rto_cod), 1), "rto_rate_prepaid_pct": round(float(100 * rto_pre), 1),
            "logistics_saving_lakh": round(float(saving / 1e5), 1), "recovered_margin_lakh": round(float(extra / 1e5), 1),
            "incentive_cost_lakh": round(float(cost / 1e5), 1), "net_benefit_lakh": round(float((saving + extra - cost) / 1e5), 1)}


# ------------------------------------------------------------------ LTV : CAC
def ltv_by_channel(orders, customers, cac, months: int = 12) -> pd.DataFrame:
    o = add_order_economics(orders).merge(customers[["customer_id", "signup_date", "acquisition_channel"]], on="customer_id")
    eligible = customers[customers.signup_date <= SNAPSHOT - pd.DateOffset(months=months)]
    o = o[o.customer_id.isin(eligible.customer_id) & (o.order_date <= o.signup_date + pd.DateOffset(months=months))]
    per_cust = o.groupby("customer_id").contribution.sum().reindex(eligible.customer_id, fill_value=0)
    t = per_cust.groupby(eligible.set_index("customer_id").acquisition_channel).agg(["mean", "size"])
    t.columns = [f"ltv_{months}m_contribution", "customers"]
    t = t.join(cac.set_index("acquisition_channel"))
    t["ltv_to_cac"] = t[f"ltv_{months}m_contribution"] / t.cac_inr
    t["payback"] = np.where(t.ltv_to_cac >= 1, "within 12 months", "not paid back")
    return t.sort_values("ltv_to_cac", ascending=False).round(2)


# ------------------------------------------------------------------ predictive CLV
def clv_features(orders, customers, obs_days: int = 90, pred_days: int = 180):
    """Features from each customer's first `obs_days`; target = delivered spend in the following `pred_days`."""
    o = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    o["day"] = (o.order_date - o.signup_date).dt.days
    cutoff = SNAPSHOT - pd.Timedelta(days=obs_days + pred_days)
    elig = customers[customers.signup_date <= cutoff].set_index("customer_id")
    o = o[o.customer_id.isin(elig.index)]
    obs = o[o.day < obs_days]
    fut = o[(o.day >= obs_days) & (o.day < obs_days + pred_days) & (o.status == "Delivered")]
    f = obs.groupby("customer_id").agg(
        orders_90d=("order_id", "size"), spend_90d=("net_value", "sum"), avg_discount_pct=("discount", "sum"),
        cod_share=("payment_mode", lambda s: (s == "COD").mean()), any_rto=("status", lambda s: (s == "RTO").any()),
        first_aov=("net_value", "first"), n_categories=("category", "nunique"),
    )
    f["avg_discount_pct"] = 100 * f.avg_discount_pct / (f.spend_90d + f.avg_discount_pct).replace(0, np.nan)
    f = f.join(elig[["city_tier", "acquisition_channel", "first_category", "acquired_in_sale"]])
    for c in ["city_tier", "acquisition_channel", "first_category"]:
        f[c] = f[c].astype("category")
    f["any_rto"] = f.any_rto.astype(int)
    y = fut.groupby("customer_id").net_value.sum().reindex(f.index, fill_value=0)
    return f, y


def train_clv_model(X, y, seed: int = 0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          categorical_features="from_dtype", random_state=seed)
    model.fit(Xtr, ytr)
    pred = np.clip(model.predict(Xte), 0, None)
    # business metric: how much of next-180-day revenue do the top-scored customers deliver?
    dec = pd.DataFrame({"pred": pred, "actual": yte.to_numpy()})
    dec["decile"] = pd.qcut(dec.pred.rank(method="first", ascending=False), 10, labels=range(1, 11))
    lift = dec.groupby("decile", observed=True).actual.sum() / dec.actual.sum() * 100
    baseline = np.clip(Xte.spend_90d.to_numpy() * (ytr.sum() / Xtr.spend_90d.sum()), 0, None)  # naive: future ∝ past
    base_dec = pd.qcut(pd.Series(baseline).rank(method="first", ascending=False), 10, labels=range(1, 11))
    base_top2 = yte.to_numpy()[np.isin(base_dec, [1, 2])].sum() / yte.sum() * 100
    metrics = {"r2": round(float(r2_score(yte, pred)), 3), "r2_naive_baseline": round(float(r2_score(yte, baseline)), 3),
               "top20_capture_pct": round(float(lift.loc[[1, 2]].sum()), 1), "top20_capture_naive_pct": round(float(base_top2), 1),
               "test_customers": len(yte)}
    return model, metrics, lift.round(1)


def inter_purchase_days(orders) -> pd.DataFrame:
    o = orders[orders.status == "Delivered"].sort_values(["customer_id", "order_date"])
    o["gap"] = o.groupby("customer_id").order_date.diff().dt.days
    first_cat = o.groupby("customer_id").category.transform("first")
    g = o[o.order_seq >= 2].assign(first_category=first_cat).dropna(subset=["gap"])
    return g.groupby("first_category").gap.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].round(0) \
        .rename(columns={"25%": "p25_days", "50%": "median_days", "75%": "p75_days"}).sort_values("median_days")
