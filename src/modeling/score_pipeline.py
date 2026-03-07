import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.modeling.trainer import Experiment
from src.modeling.score_model import CreditScoreModel


def train_score_model(experiment: Experiment, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray, X_test: pd.DataFrame, features: list[str]) -> tuple[CreditScoreModel, np.ndarray, np.ndarray]:
    """
    Train the credit scoring model and prepare transformed datasets.

    This function executes the experiment pipeline to train a logistic
    regression model and extracts coefficients to build a linear
    credit scoring model.

    The score is constructed as the difference between the coefficients
    of the "Good" and "Poor" classes:

        score = logit_good − logit_poor

    The trained preprocessing pipeline is then applied to both the
    training and test datasets.

    Parameters
    ----------
    experiment : Experiment
        Experiment object responsible for fitting the preprocessing
        pipeline and the logistic regression model.
    X_train : pd.DataFrame
        Training feature dataset.
    y_train : pd.Series or np.ndarray
        Training target labels.
    X_test : pd.DataFrame
        Test feature dataset.
    features : list
        List of feature names used by the model.

    Returns
    -------
    model : CreditScoreModel
        Trained credit scoring model.
    X_train_scaled : np.ndarray
        Transformed training features after preprocessing.
    X_test_scaled : np.ndarray
        Transformed test features after preprocessing.
    """

    coef, intercept = experiment.run(X_train, y_train)

    coef_good = coef[2]
    coef_poor = coef[0]

    coef_score = coef_good - coef_poor
    intercept_score = intercept[2] - intercept[0]

    X_train_scaled = experiment.transform(X_train)
    X_test_scaled = experiment.transform(X_test)

    model = CreditScoreModel(
        coef=coef_score,
        intercept=intercept_score,
        features=features
    )

    return model, X_train_scaled, X_test_scaled


def evaluate_model(model: CreditScoreModel, X_train: np.ndarray, X_test: np.ndarray, y_test: pd.Series | np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the credit scoring model's performance.

    This function computes the credit scores for both the training and
    test datasets, generates predictions, and calculates the classification
    accuracy on the test set.

    Parameters
    ----------
    model : CreditScoreModel
        Trained credit scoring model.
    X_train : np.ndarray
        Transformed training features.
    X_test : np.ndarray
        Transformed test features.
    y_test : pd.Series or np.ndarray
        True labels for the test dataset.

    Returns
    -------
    acc : float
        Classification accuracy on the test set.
    scores_train : np.ndarray
        Credit scores for the training dataset.
    scores_test : np.ndarray
        Credit scores for the test dataset.
    y_pred_train : np.ndarray
        Predicted class labels for the training dataset.
    y_pred_test : np.ndarray
        Predicted class labels for the test dataset.
    """

    scores_train = model.credit_score(X_train)
    scores_test = model.credit_score(X_test)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred_test)

    return acc, scores_train, scores_test, y_pred_train, y_pred_test


def score_dataset(data: pd.DataFrame, model: CreditScoreModel, experiment: Experiment, save_path: str = "predicted_scores.csv") -> pd.DataFrame:
    """
    Compute credit scores and predicted categories for a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to score.
    model : CreditScoreModel
        Trained credit score model.
    experiment : Experiment
        Experiment object containing the fitted preprocessing pipeline.
    save_path : str, optional
        Path where the scored dataset will be saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Score, Classification and Credit_Category.
    """

    scaler = experiment.pipeline.pipeline.named_steps["preprocessor"]

    X = data[model.features]

    X_scaled = scaler.transform(X)

    scores = model.credit_score(X_scaled)

    classification = model.predict(X_scaled)

    category_map = {
        0: "Poor",
        1: "Standard",
        2: "Good"
    }

    result = pd.DataFrame({
        "Score": scores,
        "Classification": classification,
        "Credit_Category": [category_map[c] for c in classification]
    })

    result.to_csv(save_path, index=False)

    return result
