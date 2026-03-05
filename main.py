import numpy as np
from sklearn.base import accuracy_score
from libraries import pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from experiment import Experiment
from model import CreditScoreModel
from viz import bucket_analysis, plot_accuracy_vs_threshold, plot_score_distribution


def main():

    df = pd.read_csv("data/clean_train.csv")

    X = df.drop("Credit_Score", axis=1)
    X = pd.get_dummies(X, drop_first=True).astype(float)

    mapping = {
        "Poor": 0,
        "Standard": 1,
        "Good": 2
    }

    y = df["Credit_Score"].map(mapping)


    final_features = [
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


    X_model = X[final_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_model,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    experiment = Experiment()

    coef, intercept = experiment.run(X_train, y_train)

    class_idx = 0 # Standard class

    coef_std = coef[class_idx]
    intercept_std = intercept[class_idx]

    X_test_scaled = experiment.transform(X_test)

    model = CreditScoreModel(
        coef=coef_std,
        intercept=intercept_std,
        features=final_features
    )
        
    # scores para test (evaluación)
    scores_test = model.credit_score(X_test_scaled)

    y_pred = np.array([model.classify(s) for s in scores_test])

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Scores:", scores_test[:10])


    # scores para train (análisis)
    X_train_scaled = experiment.transform(X_train)

    scores_train = model.credit_score(X_train_scaled)

    print(f"Thresholds used → t1: {model.t1}, t2: {model.t2}")

    plot_score_distribution(scores_train, y_train)

    plot_accuracy_vs_threshold(scores_train, y_train)


    summary = bucket_analysis(scores_train, y_train)

    print(summary)
    
if __name__ == "__main__":
    main()