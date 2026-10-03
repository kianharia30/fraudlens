from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import Config
from fraudlens.features.pipeline import UNSEEN_CODE, FeaturePipeline


def _train() -> pd.DataFrame:
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": range(n),
            "isFraud": rng.integers(0, 2, n),
            "TransactionAmt": rng.uniform(1, 100, n).round(2),
            "card1": rng.choice([100.0, 200.0, 300.0], n),
            "card2": 1.0,
            "card3": 1.0,
            "card5": rng.choice([1.0, 2.0], n),
            "addr1": 5.0,
            "ProductCD": pd.Categorical(rng.choice(["W", "C"], n)),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "R_emaildomain": None,
            "almost_empty": np.where(np.arange(n) < 2, 1.0, np.nan),
            "constant": 7.0,
        }
    )


def test_fit_drops_sparse_constant_and_excluded_columns(tmp_config: Config) -> None:
    p = FeaturePipeline.from_config(tmp_config).fit(_train())
    assert "almost_empty" in p.dropped_sparse
    assert "constant" in p.dropped_constant
    for col in ("TransactionID", "TransactionDT", "isFraud", "card_uid"):
        assert col not in p.feature_names
    assert "card_uid_freq" in p.feature_names and "card1_freq" in p.feature_names


def test_encoders_use_train_statistics_only(tmp_config: Config) -> None:
    train = _train()
    p = FeaturePipeline.from_config(tmp_config).fit(train)
    expected = float((train["card1"] == 100.0).mean())
    # A later block made only of card1=100 must still get the *train* frequency.
    later = train.iloc[:5].copy()
    later["card1"] = 100.0
    assert p.transform(later)["card1_freq"].iloc[0] == pytest.approx(expected)


def test_unseen_and_missing_categories(tmp_config: Config) -> None:
    p = FeaturePipeline.from_config(tmp_config).fit(_train())
    X = p.transform_records(
        [
            {"TransactionAmt": 10.0, "ProductCD": "H", "card1": 999.0},  # unseen product + card
            {"TransactionAmt": 10.0, "ProductCD": None},
        ]
    )
    assert X["ProductCD"].iloc[0] == UNSEEN_CODE
    assert np.isnan(X["ProductCD"].iloc[1])
    assert X["card1_freq"].iloc[0] == 0.0


def test_transform_is_stable_for_records_and_frames(tmp_config: Config) -> None:
    train = _train()
    p = FeaturePipeline.from_config(tmp_config).fit(train)
    frame = p.transform(train.iloc[:3])
    # Records arrive from JSON: ints instead of floats, extra unknown keys, missing columns.
    records = [
        {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in r.items()}
        | {"junk": 1}
        for r in train.iloc[:3].astype(object).to_dict("records")
    ]
    pd.testing.assert_frame_equal(frame, p.transform_records(records))
    assert list(frame.columns) == p.feature_names
    assert (frame.dtypes == np.float32).all()


def test_stateless_features(tmp_config: Config) -> None:
    p = FeaturePipeline.from_config(tmp_config).fit(_train())
    X = p.transform_records(
        [
            {"TransactionAmt": 12.34, "P_emaildomain": "a.com", "R_emaildomain": "b.com"},
            {"TransactionAmt": 5.0, "P_emaildomain": "a.com", "R_emaildomain": "a.com"},
        ]
    )
    assert X["amt_cents"].tolist() == [34.0, 0.0]
    assert X["email_domain_mismatch"].tolist() == [1.0, 0.0]


def test_transform_before_fit_raises(tmp_config: Config) -> None:
    with pytest.raises(RuntimeError):
        FeaturePipeline.from_config(tmp_config).transform(_train())
