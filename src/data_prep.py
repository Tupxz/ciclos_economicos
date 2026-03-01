"""
Data preparation module for macroeconomic dataset ingestion and standardization.

This module reads quarterly economic data from FRED Excel files and builds
a standardized dataset for time series analysis.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def read_fred_xlsx(path: Path) -> pd.Series:
    """
    Read a FRED Excel file and extract the data series.
    
    FRED Excel files often have multiple sheets with metadata in early sheets.
    This function robustly detects which sheet contains actual time series data
    by sampling each sheet, counting valid dates in the first column, and selecting
    the sheet with the most dates (must meet minimum threshold of 30).
    
    Returns a clean pandas Series with DatetimeIndex.
    
    Args:
        path (Path): Full path to the FRED Excel file (.xlsx).
        
    Returns:
        pd.Series: Indexed by date (DatetimeIndex), values are numeric observations.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no sheet with sufficient valid dates is found.
    """
    if not path.exists():
        raise FileNotFoundError(f"FRED file not found: {path}")
    
    # Get all sheet names
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    
    # Evaluate each sheet: count valid dates and find first valid date index
    sheet_scores = {}  # {sheet_name: (valid_date_count, first_valid_idx)}
    
    for sheet_name in sheet_names:
        # Read sample (first 400 rows, first 2 columns) to evaluate sheet
        df_sample = pd.read_excel(
            path, sheet_name=sheet_name, header=None, nrows=400, usecols=[0, 1]
        )
        
        if len(df_sample) < 1:
            continue
        
        # Try to parse first column as dates (handles strings, Timestamps, Excel numbers)
        dates_parsed = pd.to_datetime(
            df_sample.iloc[:, 0], errors="coerce"
        )
        
        # Count valid (non-NaT) dates
        valid_date_mask = dates_parsed.notna()
        valid_date_count = valid_date_mask.sum()
        
        # Find first index with valid date
        first_valid_indices = valid_date_mask[valid_date_mask].index
        first_valid_idx = first_valid_indices[0] if len(first_valid_indices) > 0 else None
        
        if first_valid_idx is not None:
            sheet_scores[sheet_name] = (valid_date_count, first_valid_idx)
    
    # Minimum threshold for valid dates
    MIN_DATE_THRESHOLD = 30
    
    if not sheet_scores:
        raise ValueError(
            f"No valid data sheet found in {path.name}. "
            f"Available sheets: {sheet_names}. "
            f"No sheet contains at least {MIN_DATE_THRESHOLD} valid dates."
        )
    
    # Select sheet with the most valid dates
    best_sheet = max(sheet_scores.items(), key=lambda x: x[1][0])
    best_sheet_name = best_sheet[0]
    best_valid_count = best_sheet[1][0]
    first_valid_idx = best_sheet[1][1]
    
    # Check if best sheet meets minimum threshold
    if best_valid_count < MIN_DATE_THRESHOLD:
        detail = " | ".join(
            [f"{s}: {c} dates" for s, (c, _) in sheet_scores.items()]
        )
        raise ValueError(
            f"No sheet in {path.name} has {MIN_DATE_THRESHOLD}+ valid dates. "
            f"Sheets evaluated: {detail}"
        )
    
    # Read full data from best sheet
    df = pd.read_excel(path, sheet_name=best_sheet_name, header=None)
    
    # Extract data starting from first valid date, take only first two columns
    data = df.iloc[first_valid_idx:, :2].copy()
    data.columns = ["date", "value"]
    
    # Convert to proper types
    data["date"] = pd.to_datetime(data["date"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    
    # Remove NaN rows and sort by date
    data = data.dropna().sort_values("date")
    
    if len(data) == 0:
        raise ValueError(
            f"No valid data rows found in sheet '{best_sheet_name}' of {path.name} "
            f"(after parsing dates and values)"
        )
    
    # Return as Series with date as index
    series = pd.Series(data["value"].values, index=data["date"], name=path.stem)
    
    return series


def to_quarterly_mean(s: pd.Series) -> pd.Series:
    """
    Aggregate a monthly or daily series to quarterly mean.
    
    Groups observations by quarter and computes the mean. Normalizes
    the resulting index to quarter-end timestamps for consistency.
    
    Args:
        s (pd.Series): Time series with DatetimeIndex (any frequency).
        
    Returns:
        pd.Series: Quarterly series with index normalized to quarter-end.
    """
    # Resample to quarterly mean (using 'QE' for quarter-end in pandas 2.2+)
    quarterly = s.resample("QE").mean()
    quarterly.index = quarterly.index.to_period("Q").to_timestamp("Q")
    
    return quarterly


def build_dataset(raw_dir: Path) -> pd.DataFrame:
    """
    Build a standardized macroeconomic dataset from FRED files.
    
    Reads the required Excel files from raw_dir. For quarterly series (GDP,
    consumption, investment, deflator), only normalizes index to quarter-end.
    For monthly series (unemployment, fedfunds), aggregates to quarterly mean.
    Computes derived variables (inflation, GDP growth), aligns all data by
    date, and validates before returning.
    
    File expectations:
    - GDPC1.xlsx: Real GDP (quarterly data)
    - PCECC96.xlsx: Real consumption expenditures (quarterly data)
    - GPDIC1.xlsx: Real private investment (quarterly data)
    - GDPDEF.xlsx: GDP price deflator (quarterly data)
    - UNRATE.xlsx: Unemployment rate (monthly data)
    - FEDFUNDS.xlsx: Federal funds rate (monthly data)
    
    Args:
        raw_dir (Path): Path to data/raw directory containing FRED Excel files.
        
    Returns:
        pd.DataFrame: Quarterly dataset with columns:
            ["date", "GDP", "GDP Var", "consumption", "investment", 
             "unemployment", "inflation", "fedfunds"]
        
    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If any file cannot be processed or validation fails.
    """
    # List of required files
    required_files = {
        "GDP": "GDPC1.xlsx",
        "consumption": "PCECC96.xlsx",
        "investment": "GPDIC1.xlsx",
        "deflator": "GDPDEF.xlsx",
        "unemployment": "UNRATE.xlsx",
        "fedfunds": "FEDFUNDS.xlsx",
    }
    
    # Check that all required files exist
    for key, filename in required_files.items():
        file_path = raw_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required FRED file '{filename}' not found in {raw_dir}"
            )
    
    # Read all series
    series_dict = {}
    for key, filename in required_files.items():
        file_path = raw_dir / filename
        series_dict[key] = read_fred_xlsx(file_path)
    
    # Apply appropriate aggregation/normalization:
    # - Quarterly series: only normalize index to quarter-end
    # - Monthly series: aggregate to quarterly mean (which also normalizes index)
    quarterly_series = ["GDP", "consumption", "investment", "deflator"]
    monthly_series = ["unemployment", "fedfunds"]
    
    for key in quarterly_series:
        # Normalize index to quarter-end
        s = series_dict[key]
        s.index = s.index.to_period("Q").to_timestamp("Q")
        series_dict[key] = s
    
    for key in monthly_series:
        # Aggregate monthly to quarterly mean
        series_dict[key] = to_quarterly_mean(series_dict[key])
    
    # Merge all series into a single DataFrame aligned by date
    df = pd.DataFrame(series_dict)
    df = df.sort_index()
    
    # Drop rows with any NaN to ensure complete records (align to common period)
    df = df.dropna()
    
    # Compute derived variables
    # GDP growth: 400 * log-difference (annualized percentage change)
    gdp_log_diff = np.log(df["GDP"]).diff()
    df["GDP Var"] = 400 * gdp_log_diff
    
    # Inflation: 400 * log-difference of GDP deflator (annualized)
    deflator_log_diff = np.log(df["deflator"]).diff()
    df["inflation"] = 400 * deflator_log_diff
    
    # Drop the deflator column (no longer needed)
    df = df.drop(columns=["deflator"])
    
    # Drop first row (contains NaN from diff() operation)
    df = df.iloc[1:].copy()
    
    # Reorder columns as specified
    column_order = [
        "GDP",
        "GDP Var",
        "consumption",
        "investment",
        "unemployment",
        "inflation",
        "fedfunds",
    ]
    df = df[column_order]
    
    # Add date as a column (reset index to make date a column, not index)
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})
    
    # Ensure date is at the front
    df = df[["date"] + column_order]
    
    # Validate dataset before returning
    validate_dataset(df)
    
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the macroeconomic dataset for completeness and consistency.
    
    Checks:
    1. All required columns are present with exact names.
    2. No missing values in any required column.
    3. All date values are unique.
    
    Args:
        df (pd.DataFrame): Dataset to validate (output of build_dataset).
        
    Raises:
        ValueError: If any validation check fails.
    """
    required_columns = [
        "date",
        "GDP",
        "GDP Var",
        "consumption",
        "investment",
        "unemployment",
        "inflation",
        "fedfunds",
    ]
    
    # Check 1: All required columns present
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Dataset has: {list(df.columns)}"
        )
    
    # Check 2: No missing values in required columns
    null_mask = df[required_columns].isnull().any()
    null_cols = null_mask[null_mask].index.tolist()
    if null_cols:
        raise ValueError(
            f"Found missing values in columns: {null_cols}. "
            f"All required columns must have complete data."
        )
    
    # Check 3: Unique dates
    if not df["date"].is_unique:
        raise ValueError(
            "Dataset contains duplicate dates. All dates must be unique."
        )


def save_dataset(df: pd.DataFrame, processed_dir: Path, stem: str = "dataset_taller") -> tuple:
    """
    Save the processed macroeconomic dataset to both XLSX and CSV formats.
    
    Creates both Excel and CSV versions of the dataset for portability and
    reproducibility. The Excel file preserves formatting and is suitable for
    manual inspection; the CSV is suitable for data pipelines and version control.
    The date column is exported without time component (date-only format).
    
    Args:
        df (pd.DataFrame): The dataset to save (output of build_dataset).
        processed_dir (Path): Output directory where files will be saved.
        stem (str): Base filename without extension (default: "dataset_taller").
        
    Returns:
        tuple: Two-element tuple containing (xlsx_path, csv_path) as Path objects.
        
    Raises:
        ValueError: If the DataFrame does not pass validation.
    """
    # Validate dataset before saving
    validate_dataset(df)
    
    # Create export copy and convert date to date-only format (no time component)
    df_export = df.copy()
    df_export["date"] = pd.to_datetime(df_export["date"]).dt.date
    
    # Ensure output directory exists
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    xlsx_path = processed_dir / f"{stem}.xlsx"
    csv_path = processed_dir / f"{stem}.csv"
    
    # Save to Excel (with index=False to avoid duplication)
    df_export.to_excel(xlsx_path, index=False, engine="openpyxl")
    
    # Save to CSV (with index=False for consistency)
    df_export.to_csv(csv_path, index=False)
    
    return (xlsx_path, csv_path)


if __name__ == "__main__":
    import sys
    
    # Default paths
    default_raw_dir = Path("data/raw")
    default_processed_dir = Path("data/processed")
    
    # Parse command line arguments if provided
    if len(sys.argv) >= 3:
        raw_dir = Path(sys.argv[1])
        processed_dir = Path(sys.argv[2])
    else:
        raw_dir = default_raw_dir
        processed_dir = default_processed_dir
    
    # Build dataset from FRED files
    try:
        df = build_dataset(raw_dir)
        print(f"✓ Dataset built: {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error building dataset: {e}")
        sys.exit(1)
    
    # Save dataset to both formats
    try:
        xlsx_path, csv_path = save_dataset(df, processed_dir)
        print(f"✓ Saved to {xlsx_path}")
        print(f"✓ Saved to {csv_path}")
    except Exception as e:
        print(f"✗ Error saving dataset: {e}")
        sys.exit(1)
