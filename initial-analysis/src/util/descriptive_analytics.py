import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.ticker as mtick

from util.general_utilities import info

def show_dataset(title: str, df: pd.DataFrame, head_count=10):
    info(title)
    print(df.head(head_count))

def number_companies_per_sector(df: pd.DataFrame):
    companies_by_sector = (
        df.groupby('sector')['company_id']
        .nunique()
        .reset_index(name='n_companies')
        .sort_values('n_companies', ascending=False)
    )

    return companies_by_sector

def amount_purchase_per_sector(df: pd.DataFrame):
    purchase_by_sector = (
        df.groupby('sector')['Purchase']
        .sum()
        .reset_index(name='total_purchase')
        .sort_values('total_purchase', ascending=False)
    )

    return purchase_by_sector

def number_companies_per_province(df: pd.DataFrame):
    companies_by_sector = (
        df.groupby('province')['company_id']
        .nunique()
        .reset_index(name='n_companies')
        .sort_values('n_companies', ascending=False)
    )

    return companies_by_sector

def amount_purchase_per_province(df: pd.DataFrame):
    purchase_by_sector = (
        df.groupby('province')['Purchase']
        .sum()
        .reset_index(name='total_purchase')
        .sort_values('total_purchase', ascending=False)
    )

    return purchase_by_sector

def number_companies_per_legal_nature(df: pd.DataFrame):
    companies_by_sector = (
        df.groupby('legal_nature')['company_id']
        .nunique()
        .reset_index(name='n_companies')
        .sort_values('n_companies', ascending=False)
    )

    return companies_by_sector

def amount_purchase_per_legal_nature(df: pd.DataFrame):
    purchase_by_sector = (
        df.groupby('legal_nature')['Purchase']
        .sum()
        .reset_index(name='total_purchase')
        .sort_values('total_purchase', ascending=False)
    )

    return purchase_by_sector


def global_graph_setup():
    # Optional: nicer global style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11

# ──────────────────────────────────────────────────────────────
# 1 + 2. Sector – companies & purchase
# ──────────────────────────────────────────────────────────────

def plot_sector_analysis(comp_sector, purch_sector: pd.DataFrame, top=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax2.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f'{x / 1_000_000:.1f} M €')
    )

    sns.barplot(data=comp_sector.head(top), x='n_companies', y='sector', palette='Blues_d', ax=ax1)
    ax1.set_title(f"Number of Companies by Sector (Top {top})")
    ax1.set_xlabel('Number of Companies')

    sns.barplot(data=purch_sector.head(10), x='total_purchase', y='sector', palette='Greens_d', ax=ax2)
    ax2.set_title(f"Total Purchase by Sector (Top {top})")

    ax2.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f} M €')
    )
    ax2.set_xlabel('Total Purchase (millions €)')

    plt.tight_layout()
    plt.show()


def plot_companies_vs_purchase_by_sector(comp_sector, purch_sector: pd.DataFrame, top_n=10):
    comp = comp_sector.head(top_n)
    purch = purch_sector.head(top_n)

    # Merge so we have matching order
    df_plot = comp.merge(purch, on='sector').sort_values('n_companies', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Companies — left axis (bars)
    color1 = 'tab:blue'
    ax1.set_xlabel('Sector')
    ax1.set_ylabel('Number of Companies', color=color1)
    ax1.bar(df_plot['sector'], df_plot['n_companies'], color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Purchase — right axis (line or second bars)
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Total Purchase (millions €)', color=color2)
    ax2.plot(df_plot['sector'], df_plot['total_purchase'] / 1_000_000,
             color=color2, marker='o', linewidth=2, markersize=8)
    # or use bars: ax2.bar(df_plot['sector'], df_plot['total_purchase'] / 1_000_000,
    #                      color=color2, alpha=0.4, width=0.4)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Rotate x-labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Sectors: Companies vs Total Purchase')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 3 + 4. Province – companies & purchase
# ──────────────────────────────────────────────────────────────

def plot_province_analysis(comp_prov, purch_prov: pd.DataFrame, top_n=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

    ax2.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f'{x / 1_000_000:.1f} M €')
    )

    sns.barplot(
        data=comp_prov.head(top_n),
        x='n_companies', y='province',
        palette='Purples_d', ax=ax1
    )
    ax1.set_title(f'Number of Companies by Province (Top {top_n})')

    sns.barplot(
        data=purch_prov.head(top_n),
        x='total_purchase', y='province',
        palette='Oranges_d', ax=ax2
    )
    ax2.set_title(f'Total Purchase by Province (Top {top_n})')

    plt.tight_layout()
    plt.show()

def plot_companies_vs_purchase_by_province(comp_prov, purch_prov: pd.DataFrame, top_n=10):
    comp = comp_prov.head(top_n)
    purch = purch_prov.head(top_n)

    # Merge so we have matching order
    df_plot = comp.merge(purch, on='province').sort_values('n_companies', ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Companies — left axis (bars)
    color1 = 'tab:blue'
    ax1.set_xlabel('Province')
    ax1.set_ylabel('Number of Companies', color=color1)
    ax1.bar(df_plot['province'], df_plot['n_companies'], color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Purchase — right axis (line or second bars)
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Total Purchase (millions €)', color=color2)
    ax2.plot(df_plot['province'], df_plot['total_purchase'] / 1_000_000,
             color=color2, marker='o', linewidth=2, markersize=8)
    # or use bars: ax2.bar(df_plot['sector'], df_plot['total_purchase'] / 1_000_000,
    #                      color=color2, alpha=0.4, width=0.4)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Rotate x-labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Province: Companies vs Total Purchase')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 5 + 6. Legal nature – companies & purchase
# ──────────────────────────────────────────────────────────────

def plot_legal_nature_analysis(comp_legal, purch_legal: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: number of companies (bar)
    sns.barplot(
        data=comp_legal,
        x='n_companies', y='legal_nature',
        palette='coolwarm', ax=ax1
    )
    ax1.set_title('Number of Companies by Legal Nature')
    ax1.set_xlabel('Number of Companies')

    # Right: pie chart with legend
    wedges, texts, autotexts = ax2.pie(
        purch_legal['total_purchase'],
        labels=None,  # ← remove labels from pie
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,  # move % closer to center
        colors=sns.color_palette("Set2"),
        textprops={'fontsize': 11}
    )

    # Add legend on the right side
    ax2.legend(
        wedges,
        purch_legal['legal_nature'],
        title="Legal Nature",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),  # place legend outside to the right
        fontsize=10
    )

    ax2.set_title('Share of Total Purchase by Legal Nature')

    plt.tight_layout()
    plt.show()


def plot_total_purchases_by_month(df: pd.DataFrame):
    # Aggregate total purchase per month
    monthly_total = (
        df.groupby('Month')['Purchase']
        .sum()
        .reset_index()
        .sort_values('Month')  # from -35 to 0
    )

    # Convert Month to more readable label (optional)
    monthly_total['Month_label'] = monthly_total['Month'].apply(lambda x: f"m{x}" if x < 0 else "m0")

    plt.figure(figsize=(12, 6))

    # Line + markers
    sns.lineplot(
        data=monthly_total,
        x='Month',
        y='Purchase',
        marker='o',
        linewidth=2,
        markersize=8,
        color='darkblue'
    )

    # Optional: add bar for better volume perception
    plt.bar(monthly_total['Month'], monthly_total['Purchase'], alpha=0.3, color='lightblue')

    plt.title('Total Purchases by Month (All Companies)', fontsize=14, pad=15)
    plt.xlabel('Month (negative = months before reference point)', fontsize=12)
    plt.ylabel('Total Purchase (€)', fontsize=12)

    # Format y-axis in millions if needed
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f} M')
    )

    # Add grid and improve readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(monthly_total['Month'], rotation=45)

    plt.tight_layout()
    plt.show()

    # Also print the table
    print("Total purchases by month:")
    print(monthly_total[['Month', 'Purchase']].to_string(index=False))


def plot_purchases_by_sector_and_month_lines(
        df,
        top_n=10,
        rank_by='purchase'  # 'purchase' or 'companies'
):
    if rank_by == 'purchase':
        top_sectors = df.groupby('sector')['Purchase'].sum().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Sectors by Total Purchase"
        sort_metric = df.groupby('sector')['Purchase'].sum()
    else:
        top_sectors = df.groupby('sector')['company_id'].nunique().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Sectors by Number of Companies"
        sort_metric = df.groupby('sector')['company_id'].nunique()

    df_top = df[df['sector'].isin(top_sectors)].copy()

    monthly_sector = (
        df_top.groupby(['Month', 'sector'])['Purchase']
        .sum()
        .reset_index()
        .sort_values(['sector', 'Month'])
    )

    sorted_sectors = sort_metric.loc[top_sectors].sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(14, 8))

    # Plot with explicit order in hue (controls legend order)
    sns.lineplot(
        data=monthly_sector,
        x='Month',
        y='Purchase',
        hue='sector',
        hue_order=sorted_sectors,  # ← this sorts the legend
        marker='o',
        linewidth=1.8,
        markersize=6,
        palette='tab10'
    )

    plt.title(f'Total Purchases by Month – {title_suffix}', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Purchase (€)', fontsize=12)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f} M €')
    )

    # Legend will now follow the sorted order
    plt.legend(
        title='Sector',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=10
    )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_purchases_by_sector_and_month_heatmap(df: pd.DataFrame):
    # Pivot table: rows = sectors, columns = months
    pivot = (
        df.pivot_table(
            values='Purchase',
            index='sector',
            columns='Month',
            aggfunc='sum',
            fill_value=0
        )
    )

    # Optional: normalize per sector (show relative pattern)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        cbar_kws={'label': 'Total Purchase (€)'}
    )

    plt.title('Purchases Heatmap: Sector × Month', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sector', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_purchases_by_province_and_month_lines(
        df,
        top_n=10,
        rank_by='purchase'  # 'purchase' or 'companies'
):
    if rank_by == 'purchase':
        top_province = df.groupby('province')['Purchase'].sum().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Province by Total Purchase"
        sort_metric = df.groupby('province')['Purchase'].sum()
    else:
        top_province = df.groupby('province')['company_id'].nunique().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Provinces by Number of Companies"
        sort_metric = df.groupby('province')['company_id'].nunique()

    df_top = df[df['province'].isin(top_province)].copy()

    monthly_province = (
        df_top.groupby(['Month', 'province'])['Purchase']
        .sum()
        .reset_index()
        .sort_values(['province', 'Month'])
    )

    sorted_province = sort_metric.loc[top_province].sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(14, 8))

    # Plot with explicit order in hue (controls legend order)
    sns.lineplot(
        data=monthly_province,
        x='Month',
        y='Purchase',
        hue='province',
        hue_order=sorted_province,  # ← this sorts the legend
        marker='o',
        linewidth=1.8,
        markersize=6,
        palette='tab10'
    )

    plt.title(f'Total Purchases by Month – {title_suffix}', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Purchase (€)', fontsize=12)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f} M €')
    )

    # Legend will now follow the sorted order
    plt.legend(
        title='Province',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=10
    )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_purchases_by_province_and_month_heatmap(df: pd.DataFrame):
    # Pivot table: rows = sectors, columns = months
    pivot = (
        df.pivot_table(
            values='Purchase',
            index='province',
            columns='Month',
            aggfunc='sum',
            fill_value=0
        )
    )

    # Optional: normalize per sector (show relative pattern)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        cbar_kws={'label': 'Total Purchase (€)'}
    )

    plt.title('Purchases Heatmap: Province × Month', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Province', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_purchases_by_legal_nature_and_month_lines(
        df,
        top_n=10,
        rank_by='purchase'  # 'purchase' or 'companies'
):
    if rank_by == 'purchase':
        top_province = df.groupby('legal_nature')['Purchase'].sum().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Legal Nature by Total Purchase"
        sort_metric = df.groupby('legal_nature')['Purchase'].sum()
    else:
        top_province = df.groupby('legal_nature')['company_id'].nunique().nlargest(top_n).index.tolist()
        title_suffix = f"Top {top_n} Legal Nature by Number of Companies"
        sort_metric = df.groupby('legal_nature')['company_id'].nunique()

    df_top = df[df['legal_nature'].isin(top_province)].copy()

    monthly_province = (
        df_top.groupby(['Month', 'legal_nature'])['Purchase']
        .sum()
        .reset_index()
        .sort_values(['legal_nature', 'Month'])
    )

    sorted_province = sort_metric.loc[top_province].sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(14, 8))

    # Plot with explicit order in hue (controls legend order)
    sns.lineplot(
        data=monthly_province,
        x='Month',
        y='Purchase',
        hue='legal_nature',
        hue_order=sorted_province,  # ← this sorts the legend
        marker='o',
        linewidth=1.8,
        markersize=6,
        palette='tab10'
    )

    plt.title(f'Total Purchases by Month – {title_suffix}', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Purchase (€)', fontsize=12)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x / 1_000_000:.1f} M €')
    )

    # Legend will now follow the sorted order
    plt.legend(
        title='Legal Nature',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=10
    )

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_purchases_by_legal_nature_and_month_heatmap(df: pd.DataFrame):
    # Pivot table: rows = sectors, columns = months
    pivot = (
        df.pivot_table(
            values='Purchase',
            index='legal_nature',
            columns='Month',
            aggfunc='sum',
            fill_value=0
        )
    )

    # Optional: normalize per sector (show relative pattern)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        cbar_kws={'label': 'Total Purchase (€)'}
    )

    plt.title('Purchases Heatmap: Province × Month', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Legal Nature', fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_avg_monthly_purchase_per_sector_company(df: pd.DataFrame):
    # First: total purchase per company per sector
    company_sector_total = (
        df.groupby(['sector', 'company_id'])['Purchase']
        .sum()
        .reset_index()
    )

    # Then: average per company per sector
    avg_per_company = (
        company_sector_total
        .groupby('sector')['Purchase']
        .mean()
        .reset_index(name='avg_monthly_purchase')
        .sort_values('avg_monthly_purchase', ascending=False)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=avg_per_company,
        x='avg_monthly_purchase',
        y='sector',
        palette='viridis'
    )

    plt.title('Average Monthly Purchase per Company by Sector', fontsize=14)
    plt.xlabel('Average Purchase per Company per Month (€)', fontsize=12)
    plt.ylabel('Sector', fontsize=12)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:,.0f} €')
    )

    # Add value labels on bars
    for i, v in enumerate(avg_per_company['avg_monthly_purchase']):
        plt.text(v + 0.02 * max(avg_per_company['avg_monthly_purchase']), i, f'{v:,.0f}', va='center')

    plt.tight_layout()
    plt.show()

    print("Average monthly purchase per company by sector:")
    print(avg_per_company.round(0).to_string(index=False))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_avg_monthly_purchase_per_province_company(df: pd.DataFrame, top_n=10):
    # Total purchase per company per province
    company_prov_total = (
        df.groupby(['province', 'company_id'])['Purchase']
        .sum()
        .reset_index()
    )

    # Average purchase per company per province
    avg_per_company = (
        company_prov_total
        .groupby('province')['Purchase']
        .mean()
        .reset_index(name='avg_monthly_purchase')
        .sort_values('avg_monthly_purchase', ascending=False)
    )

    # Take only top N
    top_avg = avg_per_company.head(top_n)

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=top_avg,
        x='avg_monthly_purchase',
        y='province',
        palette='viridis',
        order=top_avg['province']  # keeps the sorted order
    )

    plt.title(f'Average Monthly Purchase per Company – Top {top_n} Provinces', fontsize=14)
    plt.xlabel('Average Purchase per Company per Month (€)', fontsize=12)
    plt.ylabel('Province', fontsize=12)

    # Format x-axis with commas and € symbol
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:,.0f} €')
    )

    # Add value labels on the right of each bar
    max_val = top_avg['avg_monthly_purchase'].max()
    for i, v in enumerate(top_avg['avg_monthly_purchase']):
        plt.text(
            v + 0.02 * max_val,
            i,
            f'{v:,.0f} €',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.show()

    print(f"\nAverage monthly purchase per company – Top {top_n} provinces:")
    print(top_avg.round(0).to_string(index=False))

def plot_avg_monthly_purchase_per_legal_nature_company(df: pd.DataFrame):
    # First: total purchase per company per sector
    company_sector_total = (
        df.groupby(['legal_nature', 'company_id'])['Purchase']
        .sum()
        .reset_index()
    )

    # Then: average per company per sector
    avg_per_company = (
        company_sector_total
        .groupby('legal_nature')['Purchase']
        .mean()
        .reset_index(name='avg_monthly_purchase')
        .sort_values('avg_monthly_purchase', ascending=False)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=avg_per_company,
        x='avg_monthly_purchase',
        y='legal_nature',
        palette='viridis'
    )

    plt.title('Average Monthly Purchase per Company by Legal Nature', fontsize=14)
    plt.xlabel('Average Purchase per Company per Month (€)', fontsize=12)
    plt.ylabel('Province', fontsize=12)

    # Format x-axis
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:,.0f} €')
    )

    # Add value labels on bars
    for i, v in enumerate(avg_per_company['avg_monthly_purchase']):
        plt.text(v + 0.02 * max(avg_per_company['avg_monthly_purchase']), i, f'{v:,.0f}', va='center')

    plt.tight_layout()
    plt.show()

    print("Average monthly purchase per company by legal nature:")
    print(avg_per_company.round(0).to_string(index=False))

# def number_companies_per_sector(df: pd.DataFrame):
#     """
#     Number of companies per sector
#     :param df: Dataframe
#     :return:
#     """
#     sector_counts = (
#         df
#         .groupby(['ateco_code'])
#         .size()
#         .reset_index(name='company_count')
#         .sort_values('company_count', ascending=False)
#         .reset_index(drop=True)
#     )
#
#     sector_counts['share_%'] = (
#             sector_counts['company_count'] / len(df) * 100
#     ).round(2)
#
#     return sector_counts

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



