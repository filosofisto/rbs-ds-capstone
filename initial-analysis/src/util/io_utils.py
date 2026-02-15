import pandas as pd

from util.general_utilities import info, debug, error


def load_data(project_root, file: str) -> pd.DataFrame:
    """
    Load data from file
    :param project_root: Project root folder
    :param file: File name
    :return:
    """
    data_from_path = project_root / ".." / "data" / file

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
    info("Translating columns to English Language...")
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

    persist_dataset(project_root, df, file_to)

    return df

def check_ateco_columns_nullability(df: pd.DataFrame):
    """
    Check if ateco_code and ateco_desc are nullable compatible, that means each row where ateco_code is null,
    ateco_desc is null as well
    :param df:
    :return:
    """
    info("Checking wheter rows with ateco_code null and ateco_desc null are the same rows")
    # Total nulls in each (for reference)
    null_ateco_code = df['ateco_code'].isnull().sum()
    null_ateco_desc = df['ateco_desc'].isnull().sum()
    print(f"Nulls in ateco_code: {null_ateco_code}")
    print(f"Nulls in ateco_desc: {null_ateco_desc}")

    # Rows where BOTH are null
    both_null = df[df['ateco_code'].isnull() & df['ateco_desc'].isnull()].shape[0]
    print(f"Rows where BOTH are null: {both_null}")

    # Rows where ateco_code is null but ateco_desc is NOT
    code_null_desc_not = df[df['ateco_code'].isnull() & df['ateco_desc'].notnull()].shape[0]
    print(f"Rows where ateco_code null but ateco_desc NOT: {code_null_desc_not}")

    # Rows where ateco_desc is null but ateco_code is NOT
    desc_null_code_not = df[df['ateco_desc'].isnull() & df['ateco_code'].notnull()].shape[0]
    print(f"Rows where ateco_desc null but ateco_code NOT: {desc_null_code_not}")

    # Quick summary
    if code_null_desc_not == 0 and desc_null_code_not == 0:
        print("All nulls are in the exact same rows—no mismatches.")
    else:
        print("There are mismatches in null patterns.")


def check_ateco_code_duplication(df: pd.DataFrame):
    """
    Verify duplications of ateco_code.
    :param df: Dataframe
    :return:
    """
    duplicates_count = df['ateco_code'].duplicated().sum()

    print(f"Number of duplicate rows (based on ateco_code): {duplicates_count}")

    if duplicates_count == 0:
        print("→ No repetitions — ateco_code is unique")
    else:
        print(f"→ Found {duplicates_count} repetitions")


import pandas as pd


def check_revenue_vs_months(df: pd.DataFrame, tolerance_eur: float = 10.0) -> None:
    """
    Compare 'revenue' column with the sum of monthly columns (m-0, m-1, ..., m-35).

    Prints:
    - Summary statistics of revenue, monthly sum, differences
    - Percentage of exact matches, near matches, and meaningful differences
    - Top rows with largest absolute and relative differences

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'revenue' and columns like 'm-0', 'm-1', ..., 'm-35'
    tolerance_eur : float, default 10.0
        Threshold in euros to consider a difference as "near match"
    """
    info("Compare 'revenue' column with the sum of monthly columns (m-0, m-1, ..., m-35)...")

    # 1. Find all monthly columns
    monthly_cols = [col for col in df.columns if col.startswith('m-') and col[2:].isdigit()]

    if not monthly_cols:
        error("No monthly columns (m-*) found in the DataFrame.")
        return

    if 'revenue' not in df.columns:
        error("Column 'revenue' not found in the DataFrame.")
        return

    info(f"Found {len(monthly_cols)} monthly columns: {monthly_cols[:5]}{' ...' if len(monthly_cols) > 5 else ''}")

    # 2. Calculate sum of monthly values
    df['monthly_sum'] = df[monthly_cols].sum(axis=1, skipna=True)

    # Count how many monthly values were non-null per row
    df['n_months_nonnull'] = df[monthly_cols].notna().sum(axis=1)

    # 3. Calculate differences
    df['diff'] = df['revenue'] - df['monthly_sum']

    # Avoid division by zero in percentage
    df['diff_pct'] = (
            (df['revenue'] - df['monthly_sum']) /
            df['revenue'].replace(0, pd.NA).astype('Float64') * 100
    ).round(2)

    # 4. Summary statistics
    cols_to_show = ['revenue', 'monthly_sum', 'diff', 'diff_pct', 'n_months_nonnull']
    stats = df[cols_to_show].describe().round(2)

    print("\n" + "=" * 60)
    print("Revenue vs Monthly Sum – Summary Statistics")
    print("=" * 60)
    print(stats)

    # 5. Match quality counts
    total_rows = len(df)

    exact_matches = (df['diff'] == 0).sum()
    near_matches = (df['diff'].abs() <= tolerance_eur).sum()
    meaningful_diff = total_rows - near_matches

    print("\n" + "-" * 60)
    print("Match Quality:")
    print("-" * 60)
    print(f"  Exact matches (diff = 0)         : {exact_matches:>8,} rows  ({exact_matches / total_rows:>6.1%})")
    print(
        f"  Near matches (±{tolerance_eur:.0f} €)            : {near_matches:>8,} rows  ({near_matches / total_rows:>6.1%})")
    print(
        f"  Meaningful differences (> ±{tolerance_eur:.0f} €) : {meaningful_diff:>8,} rows  ({meaningful_diff / total_rows:>6.1%})")

    # 6. Show some problematic rows (if any)
    if meaningful_diff > 0:
        print("\n" + "=" * 60)
        print(f"Top 10 rows with largest absolute difference (|diff| > {tolerance_eur} €)")
        print("=" * 60)
        mismatches_abs = df[df['diff'].abs() > tolerance_eur].copy()
        if not mismatches_abs.empty:
            print(
                mismatches_abs
                .sort_values('diff', ascending=False)
                .head(10)[['revenue', 'monthly_sum', 'diff', 'diff_pct', 'n_months_nonnull']]
                .round(2)
                .to_string()
            )

        print("\n" + "=" * 60)
        print("Top 10 rows with largest relative difference (|diff_pct|)")
        print("=" * 60)
        mismatches_rel = df[df['diff_pct'].abs() > 1].copy()  # only show >1% difference
        if not mismatches_rel.empty:
            print(
                mismatches_rel
                .sort_values('diff_pct', key=abs, ascending=False)
                .head(10)[['revenue', 'monthly_sum', 'diff', 'diff_pct', 'n_months_nonnull']]
                .round(2)
                .to_string()
            )
    else:
        print("\nPerfect or near-perfect match across all rows ✓")


def create_sanitized_dataset(project_root, path_from: str, path_to: str) -> pd.DataFrame:
    """
    Create a version of file without empty ateco_code and ateco_desc and additionally make province to UNKNOWN for empty ones
    :param project_root: Root project folder
    :param path_from: Input file path
    :param path_to: Target file path
    :return: Panda dataframe
    """
    info("Sanitizing Data...")
    df = load_data(project_root, path_from)

    # Remove rows where important info is missing
    info("Removing rows with ateco_code and ateco_desc null")
    df = df.dropna(subset=['ateco_code', 'ateco_desc'])

    # Fill missing provinces with "OTHERS"
    info("Fixing null provinces to OTHERS")
    df['province'] = df['province'].fillna("OTHERS")

    # Check results
    info(f"Cleaned rows: {len(df)}")
    info(f"Missing province after cleaning: {df['province'].isnull().sum()}")

    # Optionally, save to CSV
    data_path_to = persist_dataset(project_root, df, path_to)

    info(f"File generated: {data_path_to}")

    return df

def group_data(project_root, df: pd.DataFrame, path_to: str):
    """
    Group (summing up numeric columns) by ateco_code, ateco_desc, legal_nature and province.
    :param project_root: Project root folder
    :param df: Dataframe
    :param path_to: Target file path
    :return:
    """
    info("Grouping data by 'ateco_code', 'ateco_desc', 'legal_nature', 'province', 'revenue' and 'employees'...")
    # Define the grouping columns
    group_cols = ['ateco_code', 'ateco_desc', 'legal_nature', 'province', 'revenue', 'employees']

    # Group and sum all numeric columns automatically
    grouped = (
        df
        .groupby(group_cols, dropna=False)  # keep rows where some keys are NaN
        .sum(numeric_only=True)  # sum only numeric columns
        .reset_index()  # bring grouping columns back as normal columns
    )

    data_path_to = project_root / "data" / path_to
    grouped.to_csv(data_path_to, index=False, sep=";", encoding="utf-8")

    info(f"File generated: {data_path_to}")

    return grouped

def split_month_purchase(df: pd.DataFrame):
    """
    Split the rows creating a row per month.
    :param project_root: Project root folder
    :param df: Dataframe
    :param path_to: Target file path
    :return:
    """
    info("Splitting rows per month...")

    # 1. Find all monthly columns (assuming lowercase 'm-')
    monthly_cols = [col for col in df.columns if col.startswith('m-') and col[2:].isdigit()]

    # Sort them for consistency (m-35 to m-0)
    monthly_cols = sorted(monthly_cols, key=lambda x: int(x.split('-')[1]))

    info(f"Found {len(monthly_cols)} monthly columns: {monthly_cols[:5]} ... {monthly_cols[-5:]}")

    # 2. Add company_id (1 to N, where N = len(df))
    df['company_id'] = range(1, len(df) + 1)

    # 3. Melt to long format
    # id_vars = all non-monthly columns
    id_vars = [col for col in df.columns if col not in monthly_cols]

    expanded_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=monthly_cols,
        var_name='Month_str',  # temporary
        value_name='Purchase'
    )

    # 4. Process Month column: extract number and make negative int (e.g., -35 for m-35)
    expanded_df['Month'] = expanded_df['Month_str'].str.replace('m-', '').astype(int) * -1
    expanded_df = expanded_df.drop(columns=['Month_str'])  # clean up

    # 5. Optional: sort by company_id and Month (oldest to newest)
    expanded_df = expanded_df.sort_values(['company_id', 'Month'])

    # Quick check
    info(f"Original shape: {df.shape}")
    info(f"Expanded shape: {expanded_df.shape} (should be ~36x original rows)")
    print(expanded_df.head(10))

    return expanded_df

def clean_ateco_dotless(code):
    if pd.isna(code):
        return None
    # Remove dots
    clean = code.replace('.', '')
    # Remove trailing '00' if present
    if clean.endswith('00'):
        clean = clean[:-2]
    return clean


def create_ateco_dotless_column(df: pd.DataFrame):
    # Apply transformations
    df['ateco_dotless'] = df['ateco_code'].apply(clean_ateco_dotless)


def create_sector_columns(df: pd.DataFrame):
    # Sector = first character (usually a letter)
    df['sector'] = df['ateco_code'].str[0]

    # Sub-sector = first character + first two digits after the dot
    df['sub_sector'] = (
        df['ateco_code']
        .str[0]                                 # G
        + df['ateco_code'].str[2:4]            # 47
    )

    # Optional: make sure we don't create invalid sub_sectors when format is broken
    df['sub_sector'] = df['sub_sector'].where(
        df['ateco_code'].str.len() >= 4,
        pd.NA
    )

def get_season(date):
    """
    Returns the season name in English based on the given date.
    Uses approximate astronomical/meteorological seasons for the Northern Hemisphere.
    """
    month = date.month
    day = date.day

    if (month == 12 and day >= 21) or month in [1, 2] or (month == 3 and day < 21):
        return 'Winter'
    elif (month == 3 and day >= 21) or month in [4, 5] or (month == 6 and day < 21):
        return 'Spring'
    elif (month == 6 and day >= 21) or month in [7, 8] or (month == 9 and day < 21):
        return 'Summer'
    else:  # Sep 21 – Dec 20
        return 'Autumn'

def create_season_columns(project_root, df: pd.DataFrame, path_to: str):
    # Assuming reference_date is already defined as before
    reference_date = pd.Timestamp('2025-12-31')

    # Create date column
    df['date'] = df['Month'].apply(
        lambda m: reference_date + pd.offsets.MonthEnd(m)
    )
    df['date'] = pd.to_datetime(df['date'])

    # Add English season column
    df['season'] = df['date'].apply(get_season)

    persist_dataset(project_root, df, path_to)

def persist_dataset(project_root, df: pd.DataFrame, path_to: str):
    data_path_to = project_root / ".." / "data" / path_to
    df.to_csv(data_path_to, index=False, sep=";", encoding="utf-8")

    return data_path_to