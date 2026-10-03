"""Tiny synthetic IEEE-CIS look-alike for tests, CI and a no-data smoke run.

The rows mimic the *schema* (column names, dtypes, missingness, a card-level structure)
and contain a learnable fraud signal. They say nothing about real fraud behaviour; any
metric computed on them is meaningless and is labelled ``"dataset": "synthetic"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = np.array(["W", "C", "R", "H", "S"])
CARD4 = np.array(["visa", "mastercard", "american express", "discover"])
CARD6 = np.array(["debit", "credit"])
EMAILS = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"])
RISKY_EMAILS = np.array(["protonmail.com", "mail.com"])
DEVICES = np.array(["Windows", "iOS Device", "MacOS", "SM-G960U", "Trident/7.0"])


def make_synthetic(
    n_rows: int = 6000, n_cards: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(transactions, identity)`` frames shaped like the IEEE-CIS training files."""
    rng = np.random.default_rng(seed)
    card = rng.integers(0, n_cards, n_rows)
    card_base_amt = rng.lognormal(3.5, 0.8, n_cards)
    card_attrs = {
        "card1": rng.integers(1000, 18000, n_cards).astype(float),
        "card2": rng.choice([111.0, 321.0, 555.0, np.nan], n_cards, p=[0.4, 0.3, 0.28, 0.02]),
        "card3": rng.choice([150.0, 185.0], n_cards, p=[0.9, 0.1]),
        "card5": rng.choice([226.0, 224.0, 166.0], n_cards),
        "addr1": rng.choice([299.0, 325.0, 204.0, np.nan], n_cards, p=[0.3, 0.3, 0.3, 0.1]),
        "card4": rng.choice(CARD4, n_cards),
        "card6": rng.choice(CARD6, n_cards, p=[0.75, 0.25]),
        "email": rng.choice(EMAILS, n_cards),
        "device": rng.choice(DEVICES, n_cards),
    }

    # Fraud arrives in bursts on compromised cards: higher amounts, new emails/devices.
    is_fraud = rng.random(n_rows) < 0.05
    times = np.sort(rng.integers(86_400, 86_400 * 60, n_rows))
    amount = card_base_amt[card] * rng.lognormal(0, 0.3, n_rows)
    # Only some frauds look unusual; some genuine rows look odd too (so the model errs).
    odd_fraud = is_fraud & (rng.random(n_rows) < 0.5)
    odd_genuine = ~is_fraud & (rng.random(n_rows) < 0.03)
    amount = np.where(odd_fraud | odd_genuine, amount * rng.uniform(1.5, 6, n_rows), amount)

    email = card_attrs["email"][card].astype(object)
    email[(is_fraud & (rng.random(n_rows) < 0.35)) | (rng.random(n_rows) < 0.01)] = rng.choice(
        RISKY_EMAILS
    )
    device = card_attrs["device"][card].astype(object)
    device[(is_fraud & (rng.random(n_rows) < 0.3)) | (rng.random(n_rows) < 0.02)] = "Linux"

    tx = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n_rows),
            "isFraud": is_fraud.astype(int),
            "TransactionDT": times,
            "TransactionAmt": np.round(amount, 2),
            "ProductCD": np.where(
                is_fraud & (rng.random(n_rows) < 0.5), "C", rng.choice(PRODUCTS, n_rows)
            ),
            **{
                k: card_attrs[k][card]
                for k in ("card1", "card2", "card3", "card4", "card5", "card6", "addr1")
            },
            "addr2": 87.0,
            "dist1": np.where(rng.random(n_rows) < 0.6, np.nan, rng.exponential(30, n_rows)),
            "P_emaildomain": email,
            "R_emaildomain": np.where(
                rng.random(n_rows) < 0.7, np.array(None, dtype=object), email
            ),
            "C1": rng.poisson(np.where(is_fraud, 2.5, 1.5)).astype(float),
            "C13": rng.poisson(10, n_rows).astype(float),
            "D1": np.where(
                is_fraud & (rng.random(n_rows) < 0.4), 0.0, rng.integers(0, 600, n_rows)
            ).astype(float),
            "D15": rng.integers(0, 600, n_rows).astype(float),
            "M4": rng.choice(np.array(["M0", "M1", "M2", None], dtype=object), n_rows),
            "M6": rng.choice(np.array(["T", "F", None], dtype=object), n_rows),
            "V1": rng.choice([1.0, np.nan], n_rows),
            "V258": np.where(is_fraud, rng.normal(1.8, 1, n_rows), rng.normal(1, 1, n_rows)),
            "V999_sparse": np.where(rng.random(n_rows) < 0.99, np.nan, 1.0),
        }
    )

    has_id = rng.random(n_rows) < np.where(is_fraud, 0.6, 0.25)
    ident = pd.DataFrame(
        {
            "TransactionID": tx["TransactionID"][has_id].to_numpy(),
            "id_01": rng.choice([0.0, -5.0, -10.0], int(has_id.sum())),
            "id_31": rng.choice(
                ["chrome 63.0", "mobile safari 11.0", "firefox"], int(has_id.sum())
            ),
            "DeviceType": rng.choice(["desktop", "mobile"], int(has_id.sum())),
            "DeviceInfo": device[has_id],
        }
    )
    return tx, ident
