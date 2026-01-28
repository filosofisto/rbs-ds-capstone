import pandas as pd

def load_data(project_root, file: str) -> pd.DataFrame:
    """
    Load data from file
    :param project_root: Project root folder
    :param file: File name
    :return:
    """
    data_from_path = project_root / "data" / file

    df = pd.read_csv(
        data_from_path,
        sep=";",
        encoding="cp1252",
        decimal=",",
    )

    return df

def create_en_translated_file_dataset(project_root, file_from: str, file_to: str) -> pd.DataFrame:
    """
    Create a new input file translating columns head from Italian to English
    :param project_root: Root project folder
    :param file_from: Input file path
    :param file_to: Output file path
    :return: Panda dataframe
    """
    df = load_data(project_root, file_from)

    column_mapping = {
        "ATECO": "ateco_code",
        "ATECO_DESC": "ateco_description",
        "Natura": "legal_nature",
        "Fatturato": "revenue",
        "Dipendenti": "employees",
        "Provincia": "province",
    }

    df = df.rename(columns=column_mapping)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    monthly_cols = [c for c in df.columns if c.startswith("m-")]
    df[monthly_cols] = df[monthly_cols].apply(pd.to_numeric, errors="coerce")

    data_to_path = project_root / "data" / file_to
    df.to_csv(data_to_path, index=False, sep=";", encoding="utf-8")

    return df

def create_sanitized_dataset(project_root, path_from: str, path_to: str) -> pd.DataFrame:
    """
    Create a version of file without empty ateco_code and ateco_desc and additionally make province to UNKNOWN for empty ones
    :param project_root: Root project folder
    :param path_from: Input file path
    :param path_to: Target file path
    :return: Panda dataframe
    """
    df = load_data(project_root, path_from)

    # Remove rows where important info is missing
    df = df.dropna(subset=['ateco_code', 'ateco_desc', 'province'])

    # Fill missing provinces with "UNKNOWN"
    # df['province'] = df['province'].fillna("OTHERS")

    # Check results
    print("Cleaned rows:", len(df))
    print("Missing province after cleaning:", df['province'].isnull().sum())

    # Optionally, save to CSV
    data_path_to = project_root / "data" / path_to
    df.to_csv(data_path_to, index=False, sep=";", encoding="utf-8")

    return df

