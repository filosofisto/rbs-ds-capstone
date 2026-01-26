import pandas as pd

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
        .groupby(['ateco_code', 'ateco_desc'])
        .size()
        .reset_index(name='company_count')
        .sort_values('company_count', ascending=False)
    )

    sector_counts['share_%'] = (
            sector_counts['company_count'] / len(df) * 100
    ).round(2)

    print(sector_counts.head(10))

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



