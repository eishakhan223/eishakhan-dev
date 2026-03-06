"""
Sales Demand Forecasting System - Synthetic Dataset Generator
Simulates FMCG retail sales data (Walmart/Superstore style)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

PRODUCTS = [
    "Detergent_500g", "Shampoo_200ml", "Toothpaste_100g",
    "Body_Lotion_300ml", "Face_Wash_150ml", "Soap_75g",
    "Conditioner_200ml", "Moisturizer_50g"
]

STORES = ["Store_North", "Store_South", "Store_East", "Store_West"]

def generate_sales_data():
    start_date = datetime(2020, 1, 1)
    end_date   = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq="W-MON")

    records = []
    for store in STORES:
        for product in PRODUCTS:
            base_demand   = np.random.randint(50, 300)
            trend_slope   = np.random.uniform(-0.05, 0.15)
            seasonality_amp = np.random.uniform(0.1, 0.4)

            for i, date in enumerate(date_range):
                week_of_year = date.isocalendar()[1]
                month        = date.month

                # Trend component
                trend = base_demand + trend_slope * i

                # Seasonal component (winter/summer peaks)
                seasonal = seasonality_amp * base_demand * np.sin(
                    2 * np.pi * week_of_year / 52
                )

                # Holiday bumps: Dec, Jan, Eid months (3, 4)
                holiday_bump = 0
                if month in [12, 1]:
                    holiday_bump = base_demand * 0.25
                elif month in [3, 4]:
                    holiday_bump = base_demand * 0.15

                # Promotion flag (random ~15% weeks)
                promotion = int(np.random.random() < 0.15)
                promo_lift = promotion * base_demand * 0.20

                # Noise
                noise = np.random.normal(0, base_demand * 0.08)

                sales_qty = max(0, int(
                    trend + seasonal + holiday_bump + promo_lift + noise
                ))

                records.append({
                    "date":       date,
                    "store":      store,
                    "product":    product,
                    "sales_qty":  sales_qty,
                    "promotion":  promotion,
                    "price":      round(np.random.uniform(50, 500), 2)
                })

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/fmcg_sales_raw.csv", index=False)
    print(f"Dataset generated: {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    print(df.head())
