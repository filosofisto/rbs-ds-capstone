import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Dimension	Business Meaning
# ------------------------------------------------------------
# Growth	    Is the customer expanding or shrinking?
# Stability	    Are purchases consistent or volatile?
# Seasonality	Do they buy in patterns (peaks/troughs)?
# Churn Risk	Are they slowing down or going silent?
# Momentum	    What is happening recently?
# -------------------------------------------------------------

def growth_slope(row):
    y = row.values
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]

def prepare_monthly_data(df: pd.DataFrame):
    """
    Prepare a dataframe with the month columns.
    m-35 → m-0  (past → present)

    :param df: Dataframe
    :return: Monthly data
    """
    month_cols = [col for col in df.columns if col.startswith("m-")]

    # Ensure correct order (oldest → newest)
    month_cols = sorted(month_cols, key=lambda x: int(x.split('-')[1]), reverse=True)

    monthly_data = df[month_cols]

    return monthly_data


def describe_fast_growing_customers():
    print("""
Detect Fast-Growing Customers
We use trend slope (linear regression per row)

Interpretation:
• Positive large slope → accelerating customer
• Negative slope → shrinking customer
    """)

def fast_growing_customers(df: pd.DataFrame, monthly_data: pd.DataFrame):
    """
    Detect Fast-Growing Customers
    We use trend slope (linear regression per row)

    Interpretation:
    • Positive large slope → accelerating customer
    • Negative slope → shrinking customer

    :param df: Dataframe
    :param monthly_data: Dataframe with monthly growth data
    :return:
    """
    df['growth_trend'] = monthly_data.apply(growth_slope, axis=1)

    # Flag top growers
    df['fast_growing'] = df['growth_trend'] > df['growth_trend'].quantile(0.90)

    return df[['growth_trend', 'fast_growing']].describe()

def stable_vs_seasonal_buyers(df: pd.DataFrame, monthly_data: pd.DataFrame):
    """
    Stable vs Seasonal Buyers
    We measure volatility and coefficient of variation.

    | Pattern         | Meaning                |
    | --------------- | ---------------------- |
    | Low volatility  | Reliable steady buyer  |
    | High volatility | Seasonal/project-based |

    :param df: Dataframe
    :param monthly_data:
    :return:
    """
    df['monthly_mean'] = monthly_data.mean(axis=1)
    df['monthly_std'] = monthly_data.std(axis=1)

    df['stability_index'] = df['monthly_std'] / (df['monthly_mean'] + 1)

    # Lower = more stable
    df['stable_customer'] = df['stability_index'] < df['stability_index'].quantile(0.25)
    df['highly_variable'] = df['stability_index'] > df['stability_index'].quantile(0.75)

def churn_risk_detection(df: pd.DataFrame):
    """
    We look at recent activity drop.
    :param df: Dataframe
    :return:

    | Ratio | Meaning          |
    | ----- | ---------------- |
    | ~1    | stable           |
    | <0.5  | serious slowdown |
    | ~0    | likely churned   |
    """
    recent_3 = df[['m-0', 'm-1', 'm-2']].mean(axis=1)
    past_6 = df[['m-3', 'm-4', 'm-5', 'm-6', 'm-7', 'm-8']].mean(axis=1)

    df['recent_drop_ratio'] = recent_3 / (past_6 + 1)

    df['churn_risk'] = df['recent_drop_ratio'] < 0.4

def customer_behavior_segments(df: pd.DataFrame):
    """
    Business-Level Insights You Can Now Extract:
    - Which sectors grow fastest
    - Where churn concentrates
    - Which customer types drive revenue
    :param df:
    :return:
    """
    df['behavior_segment'] = np.select(
        [
            df['fast_growing'],
            df['churn_risk'],
            df['stable_customer'],
            df['highly_variable']
        ],
        [
            'FAST_GROWING',
            'CHURN_RISK',
            'STABLE',
            'SEASONAL'
        ],
        default='NORMAL'
    )

    segment_distribution = df['behavior_segment'].value_counts(normalize=True) * 100
    print(segment_distribution)

