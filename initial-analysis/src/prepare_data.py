from pathlib import Path

from util.general_utilities import overview_dataset
from util.io_utils import create_en_translated_file_dataset, create_sanitized_dataset, check_ateco_columns_nullability, \
    check_ateco_code_duplication, group_data, check_revenue_vs_months, split_month_purchase, \
    create_ateco_dotless_column, create_sector_columns, persist_dataset, load_data


def main():
    # 1. Load Original Dataset
    project_root = Path(__file__).resolve().parents[1]

    # 2. Translate Column Heads from Italian to English
    df = create_en_translated_file_dataset(project_root, "dataset.csv", "dataset-en.csv")

    # 3. Check ateco columns nullability
    check_ateco_columns_nullability(df)

    # 4. Sanitize data
    df = create_sanitized_dataset(project_root, "dataset-en.csv", "dataset-en-clean.csv")

    # 5. Overview Sanitized Data
    overview_dataset(df)

    # 6. ateco_dotless column
    create_ateco_dotless_column(df)

    # 7. Create sector and sub_sector columns
    create_sector_columns(df)

    # 8. Split purchase per month
    split_month_purchase(project_root, df, "dataset-split-purchase.csv")

    # 9. Overview Final Dataset
    overview_dataset(load_data(project_root, "dataset-split-purchase.csv"))


if __name__ == "__main__":
    main()
