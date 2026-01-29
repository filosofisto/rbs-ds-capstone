import pandas as pd
import matplotlib.pyplot as plt

def overview_dataset(df: pd.DataFrame):
    print(df.info())
    print(df.describe())
    print(df.head())

def number_companies_per_sector(df: pd.DataFrame):
    """
    Number of companies per sector
    :param df: Dataframe
    :return:
    """
    sector_counts = (
        df
        .groupby(['ateco_code'])
        .size()
        .reset_index(name='company_count')
        .sort_values('company_count', ascending=False)
        .reset_index(drop=True)
    )

    sector_counts['share_%'] = (
            sector_counts['company_count'] / len(df) * 100
    ).round(2)

    return sector_counts

def sector_revenue_concentration(df: pd.DataFrame):
    """
    Goal: see which sectors actually generate money (not just company counts)
    :param df: Dataframe
    :return:
    """
    sector_revenue = (
        df
        .groupby('ateco_code')['revenue']
        .sum()
        .reset_index()
        .sort_values('revenue', ascending=False)
    )

    sector_revenue['revenue_share_%'] = (
        sector_revenue['revenue'] / sector_revenue['revenue'].sum() * 100
    ).round(2)

    return sector_revenue

def plot_sector_revenue_pie(df, top_n=10):

    sector_revenue = (
        df.groupby('ateco_code')['revenue']
        .sum()
        .sort_values(ascending=False)
    )

    top_sectors = sector_revenue.head(top_n)
    others = sector_revenue.iloc[top_n:].sum()

    pie_data = pd.concat([
        top_sectors,
        pd.Series({'Others': others})
    ])

    base_colors = plt.cm.tab20.colors[:len(top_sectors)]

    colors = list(base_colors) + ["#d9d9d9"]  # soft gray for Others

    plt.figure(figsize=(9, 9))
    plt.pie(
        pie_data,
        labels=pie_data.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'white'}
    )

    plt.title(f"Sector Revenue Concentration (Top {top_n} + Others)")
    plt.tight_layout()
    plt.show()

def sector_pareto(df: pd.DataFrame):
    """
    Pareto cumulative curve (80/20 visualization)
    :param df: Dataframe
    :return:
    """
    sector_rev = (
        df.groupby("ateco_code")["revenue"]
          .sum()
          .sort_values(ascending=False)
          .reset_index()
    )

    sector_rev["cum_revenue"] = sector_rev["revenue"].cumsum()
    total_revenue = sector_rev["revenue"].sum()

    sector_rev["cum_share"] = sector_rev["cum_revenue"] / total_revenue

    return sector_rev

def plot_pareto(sector_rev: pd.DataFrame):
    """
    Plot Pareto cumulative curve
    :param sector_rev: Dataframe result from sector_pareto()
    :return:
    """
    plt.figure(figsize=(10,6))

    plt.plot(sector_rev["cum_share"].values)
    plt.axhline(0.8, linestyle="--")   # 80% revenue line

    plt.xlabel("Sectors (sorted by revenue)")
    plt.ylabel("Cumulative revenue share")
    plt.title("Pareto Curve — Sector Revenue Concentration")

    plt.ylim(0, 1.05)
    plt.show()

def sector_pareto_num(sector_rev: pd.DataFrame):
    """
    How many sectors generate 80% of revenue (Pareto-style)
    """
    cutoff = (sector_rev["cum_share"] <= 0.8).sum()
    total_sectors = len(sector_rev)
    percent = cutoff / total_sectors

    print(f"{cutoff} sectors generate ~80% of total revenue")
    print(f"Total sectors: {total_sectors}")
    # Pick whichever style you prefer:
    print(f"Percent:    {percent:.1%}")           # cleanest
    # print(f"Percent:    {percent * 100:.1f}%")  # very explicit
    # print(f"Percent:    {percent:.2%}")         # more decimals

def dominant_sectors(sector_rev: pd.DataFrame, threshold=0.80):
    """
    Extract dominant sectors (80% revenue contributors)
    :param sector_rev:
    :param threshold:
    :return:
    """
    dominant = sector_rev[sector_rev["cum_share"] <= threshold].copy()

    dominant["sector_share_%"] = (
        dominant["revenue"] / sector_rev["revenue"].sum() * 100
    ).round(2)

    dominant["cum_share_%"] = (dominant["cum_share"] * 100).round(2)

    return dominant

def revenue_statistics_per_sector(df: pd.DataFrame):
    """
    Revenue statistics per sector
    :param df: Dataframe
    :return:
    """
    sector_revenue_stats = (
        df
        .groupby(['ateco_code', 'ateco_desc'])['revenue']
        .agg(
            companies='count',
            total_revenue='sum',
            avg_revenue='mean',
            median_revenue='median',
            std_revenue='std'
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )

    print(sector_revenue_stats.head(10))

def top_provinces(df: pd.DataFrame):
    """
    Top provinces
    :param df: Dataframe
    :return:
    """
    province_summary = (
        df['province']
        .value_counts()
        .reset_index()
    )

    province_summary.columns = ['province', 'company_count']

    province_summary['share_%'] = (
            province_summary['company_count'] / len(df) * 100
    ).round(2)

    print(province_summary.head(10))

def revenue_performance_by_province(df: pd.DataFrame):
    province_revenue = (
        df
        .groupby('province')['revenue']
        .agg(
            companies='count',
            total_revenue='sum',
            avg_revenue='mean'
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )

    print(province_revenue.head(10))



