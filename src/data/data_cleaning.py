from src.utils.utils import re, np, pd


DROP_COLS = ["ID", "Customer_ID", "Month", "Name", "SSN"]

FILL_MISSING_STR = [
    "Credit_Mix",
    "Occupation",
    "Type_of_Loan",
    "Payment_Behaviour",
]

STRIP_NOISE_COLS = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Num_Credit_Inquiries",
    "Total_EMI_per_month",
    "Credit_Utilization_Ratio",
]

RANGE_RULES = [
    ("Age", 18, 100),
    ("Num_of_Loan", 0, 50),
    ("Num_Bank_Accounts", 0, 20),
    ("Num_Credit_Card", 0, 50),
    ("Interest_Rate", 0, 100),
    ("Credit_History_Age", 0.01, None),
]


def _convert_credit_history_age(x) -> float:
    """
    Convert a credit history string such as
    '22 Years and 3 Months' into decimal years.

    Returns NaN if the value is missing.
    """
    if pd.isna(x):
        return np.nan

    years = re.search(r"(\d+)\s+Years", str(x))
    months = re.search(r"(\d+)\s+Months", str(x))

    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0

    return round(y + m / 12, 4)


def _apply_range_rule(data: pd.DataFrame,
                      col: str,
                      min_val: float | None,
                      max_val: float | None) -> pd.DataFrame:
    """
    Apply logical bounds to a numeric column.

    Values outside the specified range are set to NaN
    and an '<col>_invalid' indicator column is created.
    """
    mask = pd.Series(False, index=data.index)

    if min_val is not None:
        mask |= data[col] < min_val
    if max_val is not None:
        mask |= data[col] > max_val

    if mask.any():
        data[f"{col}_invalid"] = mask.astype(int)
        data.loc[mask, col] = np.nan

    return data


def _impute_numeric(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Impute missing numeric values using the median.

    Creates an '<col>_missing' indicator column when missing values exist.
    """
    if data[col].isna().any():
        data[f"{col}_missing"] = data[col].isna().astype(int)
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

    return data


def data_preprocessing(data: pd.DataFrame,
                       save_path: str | None = None) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline for the Credit Score dataset.

    Steps performed:
        1. Drop identification and PII columns.
        2. Remove corrupted string patterns and noise.
        3. Convert selected columns to numeric.
        4. Transform Credit_History_Age into decimal years.
        5. Impute missing categorical values with 'Missing'.
        6. Apply logical range validation rules.
        7. Impute remaining numeric missing values with median.
        8. Normalize Payment_of_Min_Amount values.

    Parameters
    ----------
    data : pd.DataFrame
        Raw input dataset.
    save_path : str | None, optional
        If provided, the cleaned dataset is saved to this path.

    Returns
    -------
    pd.DataFrame
        Cleaned and processed dataset.
    """

    data = data.copy()

    # Drop identification columns
    data.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # Remove global string noise
    data = data.replace({r"_+": "", r"!@9#%8": ""}, regex=True)
    data = data.replace("", np.nan)

    # Convert noisy numeric columns
    for col in STRIP_NOISE_COLS:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # Convert Credit_History_Age to decimal years
    if "Credit_History_Age" in data.columns:
        data["Credit_History_Age"] = data["Credit_History_Age"].apply(
            _convert_credit_history_age
        )

    # Impute categorical columns
    for col in FILL_MISSING_STR:
        if col in data.columns:
            data[col] = data[col].fillna("Missing")

    # Apply logical range rules
    for col, lo, hi in RANGE_RULES:
        if col in data.columns:
            data = _apply_range_rule(data, col, lo, hi)

    # Impute remaining numeric columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if not col.endswith(("_invalid", "_missing")):
            data = _impute_numeric(data, col)

    # Normalize categorical inconsistencies
    if "Payment_of_Min_Amount" in data.columns:
        data["Payment_of_Min_Amount"] = (
            data["Payment_of_Min_Amount"]
            .astype(str)
            .str.replace("NM", "No", regex=False)
        )

    if save_path is not None:
        data.to_csv(save_path, index=False)

    return data

