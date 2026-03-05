import pandas as pd

FINAL_FEATURES = [
    'Outstanding_Debt',
    'Interest_Rate',
    'Delay_from_due_date',
    'Num_Credit_Card',
    'Changed_Credit_Limit',
    'Total_EMI_per_month',
    'Credit_Mix_Standard',
    'Credit_Mix_Good',
    'Payment_of_Min_Amount_Yes'
]


def load_data(path):
    """
    Load and prepare the dataset for modeling.

    This function performs the following steps:
        1. Reads the dataset from the specified path.
        2. Maps the target variable `Credit_Score` into numeric labels.
        3. Separates features (X) and target (y).
        4. Converts categorical variables into dummy variables.
        5. Selects the final set of features used for modeling.

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing the selected variables.
    y : pd.Series
        Target variable encoded as:
            0 -> Poor
            1 -> Standard
            2 -> Good
    """

    df = pd.read_csv(path)

    mapping = {
        "Poor": 0,
        "Standard": 1,
        "Good": 2
    }

    # Encode target variable
    y = df["Credit_Score"].map(mapping)

    # Separate features
    X = df.drop("Credit_Score", axis=1)

    # Create dummy variables for categorical features
    X = pd.get_dummies(X, drop_first=True).astype(float)

    # Select final features used by the model
    X = X[FINAL_FEATURES]

    return X, y