from pathlib import Path

from util.io_utils import load_data
from util.segmentation_and_profiling import segmentation_and_profiling, avg_purchase_by_cluster_and_season


def main():
    # -----------------------------------------------------------
    # Data Analyse
    # -----------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    df = load_data(project_root, "dataset-split-purchase-seasoned.csv")

    company_df = segmentation_and_profiling(df)
    avg_purchase_by_cluster_and_season(df, company_df)

if __name__ == "__main__":
    main()