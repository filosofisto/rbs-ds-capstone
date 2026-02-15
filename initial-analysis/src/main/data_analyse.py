from pathlib import Path

from util.behavior_intelligence import prepare_monthly_data, fast_growing_customers, describe_fast_growing_customers
from util.descriptive_analytics import number_companies_per_sector, revenue_statistics_per_sector, top_provinces, \
    revenue_performance_by_province, sector_revenue_concentration, plot_sector_revenue_pie, sector_pareto, plot_pareto, \
    sector_pareto_num, dominant_sectors, show_dataset, amount_purchase_per_sector, number_companies_per_province, \
    amount_purchase_per_province, number_companies_per_legal_nature, amount_purchase_per_legal_nature, \
    plot_sector_analysis, global_graph_setup, plot_companies_vs_purchase_by_sector, plot_province_analysis, \
    plot_companies_vs_purchase_by_province, plot_legal_nature_analysis, plot_total_purchases_by_month, \
    plot_purchases_by_sector_and_month_lines, plot_purchases_by_sector_and_month_heatmap, \
    plot_avg_monthly_purchase_per_sector_company, plot_purchases_by_province_and_month_heatmap, \
    plot_purchases_by_province_and_month_lines, plot_purchases_by_legal_nature_and_month_lines, \
    plot_purchases_by_legal_nature_and_month_heatmap, plot_avg_monthly_purchase_per_province_company, \
    plot_avg_monthly_purchase_per_legal_nature_company
from util.general_utilities import title, info
from util.io_utils import load_data

def main():
    # -----------------------------------------------------------
    # Data Analyse
    # -----------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root, "dataset-split-purchase.csv")

    title("Descriptive Analysis")
    comp_sector = number_companies_per_sector(df)
    show_dataset("Number of Companies by Sector", comp_sector)
    purch_sector = amount_purchase_per_sector(df)
    show_dataset("Amount Purchase by Sector", purch_sector)
    comp_prov = number_companies_per_province(df)
    show_dataset("Number of Companies by Province", number_companies_per_province(df))
    purch_prov = amount_purchase_per_province(df)
    show_dataset("Amount Purchase by Province", purch_prov)
    comp_legal = number_companies_per_legal_nature(df)
    show_dataset("Number of Companies by Legal Nature", comp_legal)
    purch_legal = amount_purchase_per_legal_nature(df)
    show_dataset("Amount Purchase by Legal Nature", purch_legal)

    global_graph_setup()
    # plot_sector_analysis(comp_sector, purch_sector)
    # plot_companies_vs_purchase_by_sector(comp_sector, purch_sector)
    # plot_province_analysis(comp_prov, purch_prov)
    # plot_companies_vs_purchase_by_province(comp_prov, purch_prov)
    # plot_legal_nature_analysis(comp_legal, purch_legal)
    # plot_total_purchases_by_month(df)
    # plot_purchases_by_sector_and_month_lines(df)
    # plot_purchases_by_sector_and_month_heatmap(df)
    # plot_avg_monthly_purchase_per_company(df)
    # plot_purchases_by_province_and_month_lines(df)
    # plot_purchases_by_province_and_month_heatmap(df)
    # plot_purchases_by_legal_nature_and_month_lines(df)
    # plot_purchases_by_legal_nature_and_month_heatmap(df)
    plot_avg_monthly_purchase_per_sector_company(df)
    plot_avg_monthly_purchase_per_province_company(df)
    plot_avg_monthly_purchase_per_legal_nature_company(df)

    # title("Sector Revenue Concentration")
    # print(sector_revenue_concentration(df).head(10))

    # title("Revenue statistics per sector (* Which sectors are attractive *)")
    # revenue_statistics_per_sector(df)
    # plot_sector_revenue_pie(df, 10)

    # title("Pareto Performance by Sector")
    # sector_rev = sector_pareto(df)
    # sector_pareto_num(sector_rev)
    # dominant_sec = dominant_sectors(sector_rev)
    # print(dominant_sec.head(50))
    # plot_pareto(sector_rev)

    return

    title("Top Provinces (* Where revenue concentrates geographically *)")
    top_provinces(df)

    title("Revenue Performance by Province")
    revenue_performance_by_province(df)

    # ---------------------------------------------------------
    # Core Behavioral Dimensions We Will Build
    #
    # From your monthly columns m-35 … m-0 we will derive:
    #
    # Dimension	Business Meaning
    # Growth	    Is the customer expanding or shrinking?
    # Stability	    Are purchases consistent or volatile?
    # Seasonality	Do they buy in patterns (peaks/troughs)?
    # Churn Risk	Are they slowing down or going silent?
    # Momentum	    What is happening recently?
    # ---------------------------------------------------------
    month_data = prepare_monthly_data(df)
    stat = fast_growing_customers(df, month_data)
    describe_fast_growing_customers()
    print(stat)

if __name__ == "__main__":
    main()
