from libraries import re, np, pd, logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    """Convert '22 Years and 3 Months' to decimal years."""
    if pd.isna(x):
        return np.nan

    years = re.search(r"(\d+)\s+Years", str(x))
    months = re.search(r"(\d+)\s+Months", str(x))

    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0

    return round(y + m / 12, 4)


def _apply_range_rule(data: pd.DataFrame, col: str,
                      min_val: float | None,
                      max_val: float | None) -> pd.DataFrame:
    """Apply logical bounds, create invalid flag, and set out-of-range to NaN."""
    mask = pd.Series(False, index=data.index)

    if min_val is not None:
        mask |= data[col] < min_val
    if max_val is not None:
        mask |= data[col] > max_val

    n_invalid = mask.sum()

    if n_invalid > 0:
        data[f"{col}_invalid"] = mask.astype(int)
        data.loc[mask, col] = np.nan

        logger.info(
            f"[{col}] {n_invalid} values outside "
            f"[{min_val}, {max_val}] → set to NaN"
        )

    return data


def _impute_numeric(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Impute numeric column with median and create missing flag."""
    n_missing = data[col].isna().sum()

    if n_missing > 0:
        data[f"{col}_missing"] = data[col].isna().astype(int)
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

        logger.info(
            f"[{col}] {n_missing} NaN imputed with median={median_val:.4f}"
        )

    return data


def data_preprocessing(data: pd.DataFrame,
                       save_path: str | None = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the Credit Score dataset.

    Steps:
        1. Drop identification columns.
        2. Clean corrupted string patterns.
        3. Convert noisy numeric columns.
        4. Convert Credit_History_Age to decimal years.
        5. Impute categorical missing values.
        6. Apply logical range validation.
        7. Impute remaining numeric missing values.
        8. Normalize categorical inconsistencies.

    Returns:
        Cleaned DataFrame.
    """

    data = data.copy()

    logger.info("=" * 60)
    logger.info("START DATA PREPROCESSING")
    logger.info(f"Input shape: {data.shape}")

    # 1. Drop IDs
    data.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    # 2. Clean global string noise
    data = data.replace({r"_+": "", r"!@9#%8": ""}, regex=True)
    data = data.replace("", np.nan)

    # 3. Convert numeric columns with noise
    for col in STRIP_NOISE_COLS:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # 4. Convert Credit_History_Age
    if "Credit_History_Age" in data.columns:
        data["Credit_History_Age"] = data["Credit_History_Age"].apply(
            _convert_credit_history_age
        )

    # 5. Impute categorical missing
    for col in FILL_MISSING_STR:
        if col in data.columns:
            n = data[col].isna().sum()
            if n > 0:
                data[col] = data[col].fillna("Missing")
                logger.info(f"[{col}] {n} missing → 'Missing'")

    # 6. Apply logical range rules
    for col, lo, hi in RANGE_RULES:
        if col in data.columns:
            data = _apply_range_rule(data, col, lo, hi)

    # 7. Impute remaining numeric missing
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if not col.endswith(("_invalid", "_missing")):
            data = _impute_numeric(data, col)

    # 8. Normalize Payment_of_Min_Amount
    if "Payment_of_Min_Amount" in data.columns:
        data["Payment_of_Min_Amount"] = (
            data["Payment_of_Min_Amount"]
            .astype(str)
            .str.replace("NM", "No", regex=False)
        )

    logger.info(f"Output shape: {data.shape}")
    logger.info("END DATA PREPROCESSING")
    logger.info("=" * 60)

    if save_path is not None:
        data.to_csv(save_path, index=False)
        logger.info(f"File saved to: {save_path}")

    return data