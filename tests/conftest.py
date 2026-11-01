import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generate_data import generate  # noqa: E402


@pytest.fixture(scope="session")
def data():
    d = generate(seed=5, n_customers=8000)
    c, o = d["customers"].copy(), d["orders"].copy()
    c["signup_date"] = pd.to_datetime(c.signup_date)
    o["order_date"] = pd.to_datetime(o.order_date)
    return c, o, d["channel_cac"]
