from pathlib import Path

from util.descriptive_analytics import number_companies_per_sector, revenue_statistics_per_sector, top_provinces, \
    revenue_performance_by_province
from util.general_utilities import title
from util.io_utils import load_data

def main():
    # -----------------------------------------------------------
    # Data Analyse
    # -----------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root, "dataset-en-cleaned.csv")

    title("Number of companies per sector")
    number_companies_per_sector(df)

    title("Revenue statistics per sector (* Which sectors are attractive *)")
    revenue_statistics_per_sector(df)

    title("Top Provinces (* Where revenue concentrates geographically *)")
    top_provinces(df)

    title("Revenue Performance by Province")
    revenue_performance_by_province(df)

if __name__ == "__main__":
    main()
