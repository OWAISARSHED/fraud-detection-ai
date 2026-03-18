"""
Synthetic Financial Transaction Data Generator
Generates realistic transaction data with embedded fraud patterns for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_transaction_dataset(n_samples: int = 10000, fraud_ratio: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic financial transaction dataset with realistic fraud patterns.
    
    Args:
        n_samples: Total number of transactions
        fraud_ratio: Fraction of fraudulent transactions
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with transaction features and fraud label
    """
    np.random.seed(seed)
    random.seed(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ─── Legitimate Transactions ──────────────────────────────────────────────
    # Compute exact split sizes that sum to n_legit
    l1 = int(n_legit * 0.7); l2 = int(n_legit * 0.2); l3 = n_legit - l1 - l2
    legit_amounts = np.concatenate([
        np.random.lognormal(mean=3.5, sigma=1.2, size=l1),  # Small routine
        np.random.lognormal(mean=5.0, sigma=1.0, size=l2),  # Medium
        np.random.lognormal(mean=7.0, sigma=0.8, size=l3),  # Large but legit
    ])
    legit_amounts = np.clip(legit_amounts, 1, 50000)

    h1 = int(n_legit * 0.5); h2 = int(n_legit * 0.3); h3 = n_legit - h1 - h2
    legit_hours = np.concatenate([
        np.random.normal(loc=10, scale=3, size=h1),   # Morning
        np.random.normal(loc=15, scale=2, size=h2),   # Afternoon
        np.random.normal(loc=20, scale=2, size=h3),   # Evening
    ])
    legit_hours = np.clip(legit_hours % 24, 0, 23).astype(int)

    legit_data = {
        'amount':                  legit_amounts,
        'hour_of_day':             legit_hours,
        'day_of_week':             np.random.randint(0, 7, n_legit),
        'merchant_category':       np.random.choice([0, 1, 2, 3, 4, 5], n_legit, p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10]),
        'transaction_frequency':   np.random.poisson(lam=3, size=n_legit),
        'avg_transaction_amount':  legit_amounts * np.random.uniform(0.8, 1.2, n_legit),
        'distance_from_home_km':   np.random.exponential(scale=15, size=n_legit),
        'is_foreign_transaction':  np.random.binomial(1, 0.05, n_legit),
        'num_transactions_1h':     np.random.poisson(lam=1, size=n_legit),
        'num_transactions_24h':    np.random.poisson(lam=5, size=n_legit),
        'account_age_days':        np.random.randint(30, 3650, n_legit),
        'credit_limit':            np.random.choice([1000, 2500, 5000, 10000, 25000, 50000], n_legit),
        'card_present':            np.random.binomial(1, 0.80, n_legit),
        'pin_used':                np.random.binomial(1, 0.70, n_legit),
        'online_transaction':      np.random.binomial(1, 0.30, n_legit),
        'is_fraud':                np.zeros(n_legit, dtype=int),
    }

    # ─── Fraudulent Transactions ──────────────────────────────────────────────
    # Fraud patterns: unusual amounts, late-night hours, foreign txn, high velocity
    # Compute exact split sizes that sum to n_fraud
    fa1 = int(n_fraud * 0.2); fa2 = int(n_fraud * 0.5); fa3 = n_fraud - fa1 - fa2
    fraud_amounts = np.concatenate([
        np.random.uniform(0.01, 1.0, fa1),               # Testing small amounts
        np.random.lognormal(mean=8.0, sigma=0.5, size=fa2),  # Large
        np.random.uniform(90, 100, fa3),                  # Round numbers
    ])
    fraud_amounts = np.clip(fraud_amounts, 0.01, 100000)

    fh1 = int(n_fraud * 0.4); fh2 = int(n_fraud * 0.3); fh3 = n_fraud - fh1 - fh2
    fraud_hours = np.concatenate([
        np.random.randint(0, 5, fh1),    # Early morning (suspicious)
        np.random.randint(22, 24, fh2),  # Late night
        np.random.randint(6, 23, fh3),   # Mixed
    ])

    fraud_data = {
        'amount':                  fraud_amounts,
        'hour_of_day':             fraud_hours.astype(int),
        'day_of_week':             np.random.randint(0, 7, n_fraud),
        'merchant_category':       np.random.choice([0, 1, 2, 3, 4, 5], n_fraud, p=[0.10, 0.10, 0.15, 0.20, 0.25, 0.20]),
        'transaction_frequency':   np.random.poisson(lam=8, size=n_fraud),
        'avg_transaction_amount':  fraud_amounts * np.random.uniform(0.1, 0.4, n_fraud),
        'distance_from_home_km':   np.random.exponential(scale=150, size=n_fraud),
        'is_foreign_transaction':  np.random.binomial(1, 0.45, n_fraud),
        'num_transactions_1h':     np.random.poisson(lam=5, size=n_fraud),
        'num_transactions_24h':    np.random.poisson(lam=15, size=n_fraud),
        'account_age_days':        np.random.randint(1, 365, n_fraud),
        'credit_limit':            np.random.choice([1000, 2500, 5000, 10000, 25000, 50000], n_fraud),
        'card_present':            np.random.binomial(1, 0.25, n_fraud),
        'pin_used':                np.random.binomial(1, 0.15, n_fraud),
        'online_transaction':      np.random.binomial(1, 0.80, n_fraud),
        'is_fraud':                np.ones(n_fraud, dtype=int),
    }

    legit_df  = pd.DataFrame(legit_data)
    fraud_df  = pd.DataFrame(fraud_data)
    df = pd.concat([legit_df, fraud_df], ignore_index=True).sample(frac=1, random_state=seed)

    # ─── Derived Features ─────────────────────────────────────────────────────
    df['amount_to_avg_ratio']    = df['amount'] / (df['avg_transaction_amount'] + 1e-9)
    df['credit_utilization']     = df['amount'] / (df['credit_limit'] + 1)
    df['is_night_transaction']   = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['high_velocity']          = (df['num_transactions_1h'] > 3).astype(int)
    df['is_weekend']             = (df['day_of_week'] >= 5).astype(int)

    # Simulated timestamps
    base_time = datetime(2024, 1, 1)
    df['timestamp'] = [
        (base_time + timedelta(
            days=random.randint(0, 365),
            hours=int(h),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )).strftime('%Y-%m-%d %H:%M:%S')
        for h in df['hour_of_day']
    ]

    df['transaction_id'] = [f"TXN{str(i).zfill(8)}" for i in range(len(df))]
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLUMNS = [
    'amount', 'hour_of_day', 'day_of_week', 'merchant_category',
    'transaction_frequency', 'avg_transaction_amount', 'distance_from_home_km',
    'is_foreign_transaction', 'num_transactions_1h', 'num_transactions_24h',
    'account_age_days', 'credit_limit', 'card_present', 'pin_used',
    'online_transaction', 'amount_to_avg_ratio', 'credit_utilization',
    'is_night_transaction', 'high_velocity', 'is_weekend'
]
