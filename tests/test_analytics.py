import numpy as np
import pandas as pd

import analytics as A


def test_generated_data_integrity(data):
    c, o, _ = data
    assert o.order_id.is_unique and c.customer_id.is_unique
    assert set(o.customer_id) <= set(c.customer_id)
    assert (o.net_value == o.gross_value - o.discount).all()
    first = o.groupby("customer_id").order_date.min()
    assert (first >= c.set_index("customer_id").signup_date.loc[first.index]).all()
    assert set(o.status) <= {"Delivered", "RTO", "Returned", "Cancelled"}


def test_order_economics():
    o = pd.DataFrame({"status": ["Delivered", "RTO", "Returned", "Cancelled"], "payment_mode": ["UPI", "COD", "UPI", "COD"],
                      "net_value": [1000.0, 800.0, 500.0, 300.0], "shipping_fee_charged": [0, 0, 0, 0]})
    e = A.add_order_economics(o)
    assert e.revenue.tolist() == [1000.0, 0, 0, 0]
    assert np.isclose(e.contribution[0], 1000 * A.GROSS_MARGIN - A.SHIP_COST - 1000 * A.PAYMENT_GATEWAY)
    assert e.contribution[1] == -A.RTO_COST and e.contribution[3] == 0


def test_rfm_covers_every_buyer_once(data):
    _, o, _ = data
    r = A.rfm(o)
    assert len(r) == o[o.status == "Delivered"].customer_id.nunique()
    assert r[["R", "F", "M"]].isin([1, 2, 3, 4, 5]).all().all()
    assert set(r.segment) <= set(A.SEGMENT_PLAYBOOK)
    s = A.segment_summary(r)
    assert abs(s.customer_pct.sum() - 100) < 0.5 and s.action.notna().all()


def test_cohort_matrix_shape_and_future_masking(data):
    _, o, _ = data
    m = A.cohort_matrix(o)
    assert (m[0] == 100).all()
    last = m.index.max()
    assert m.loc[last, [c for c in m.columns if c not in ("cohort_size", 0)]].isna().all()
    assert (m.drop(columns="cohort_size").fillna(0) <= 100).all().all()


def test_ltv_cac_and_rto(data):
    c, o, cac = data
    t = A.ltv_by_channel(o, c, cac)
    assert set(t.index) == set(cac.acquisition_channel)
    rto = A.rto_analysis(o, c)
    assert (rto.rto_rate_pct_COD > rto.rto_rate_pct_Prepaid).all()


def test_clv_model_beats_random(data):
    c, o, _ = data
    X, y = A.clv_features(o, c)
    _, metrics, lift = A.train_clv_model(X, y)
    assert metrics["top20_capture_pct"] > 20          # random targeting would capture ~20%
    assert abs(lift.sum() - 100) < 0.5


def test_no_leakage_in_clv_features(data):
    c, o, _ = data
    X, _ = A.clv_features(o, c)
    o2 = o.merge(c[["customer_id", "signup_date"]], on="customer_id")
    early = o2[(o2.order_date - o2.signup_date).dt.days < 90].groupby("customer_id").size()
    assert (X.orders_90d == early.reindex(X.index)).all()
