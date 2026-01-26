from pathlib import Path

from util.descriptive_analytics import overview_dataset
from util.io_utils import create_en_translated_file_dataset, create_sanitized_dataset

def main():
    # 1. Load Original Dataset
    project_root = Path(__file__).resolve().parents[1]

    # 2. Translate Column Heads from Italian to English
    df = create_en_translated_file_dataset(project_root, "dataset.csv", "dataset-en.csv")

    # 3. Dataset Overview
    overview_dataset(df)

    # 4. Sanitize data
    df = create_sanitized_dataset(project_root, "dataset-en.csv", "dataset-en-cleaned.csv")

    # 5. Sanitized Data Overview
    overview_dataset(df)


if __name__ == "__main__":
    main()
