from pathlib import Path

from util.behavior_intelligence import prepare_monthly_data, fast_growing_customers, describe_fast_growing_customers
from util.descriptive_analytics import number_companies_per_sector, revenue_statistics_per_sector, top_provinces, \
    revenue_performance_by_province, sector_revenue_concentration, plot_sector_revenue_pie, sector_pareto, plot_pareto, \
    sector_pareto_num, dominant_sectors
from util.general_utilities import title
from util.io_utils import load_data

def main():
    # -----------------------------------------------------------
    # Data Analyse
    # -----------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root, "dataset-en-cleaned.csv")

    title("Number of companies per sector")
    print(number_companies_per_sector(df).head(10))

    title("Sector Revenue Concentration")
    print(sector_revenue_concentration(df).head(10))

    # title("Revenue statistics per sector (* Which sectors are attractive *)")
    # revenue_statistics_per_sector(df)
    # plot_sector_revenue_pie(df, 10)

    title("Pareto Performance by Sector")
    sector_rev = sector_pareto(df)
    sector_pareto_num(sector_rev)
    dominant_sec = dominant_sectors(sector_rev)
    print(dominant_sec.head(50))
    plot_pareto(sector_rev)

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
