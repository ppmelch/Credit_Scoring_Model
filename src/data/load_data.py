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

    df = pd.read_csv(path)

    mapping = {
        "Poor": 0,
        "Standard": 1,
        "Good": 2
    }

    y = df["Credit_Score"].map(mapping)

    X = df.drop("Credit_Score", axis=1)

    X = pd.get_dummies(X, drop_first=True).astype(float)

    X = X[FINAL_FEATURES]

    return X, y