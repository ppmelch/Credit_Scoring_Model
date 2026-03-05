from sklearn.metrics import accuracy_score
from src.modeling.score_model import CreditScoreModel


def train_score_model(experiment, X_train, y_train, X_test, class_idx, features):
    """
    Train the credit scoring model and prepare scaled datasets.

    This function runs the experiment pipeline to fit the model, extracts the
    coefficients corresponding to a selected class, and constructs a
    `CreditScoreModel`. It also applies the trained preprocessing pipeline
    to both the training and test feature sets.

    Parameters
    ----------
    experiment : Experiment
        Experiment object responsible for training the pipeline.
    X_train : pd.DataFrame
        Training feature dataset.
    y_train : pd.Series or np.ndarray
        Training target labels.
    X_test : pd.DataFrame
        Test feature dataset.
    class_idx : int
        Index of the class whose coefficients will be used to build the
        credit score model.
    features : list
        List of feature names used in the model.

    Returns
    -------
    tuple
        model : CreditScoreModel
            Trained credit score model.
        X_train_scaled : np.ndarray
            Transformed training features.
        X_test_scaled : np.ndarray
            Transformed test features.
    """

    coef, intercept = experiment.run(X_train, y_train)

    coef_std = coef[class_idx]
    intercept_std = intercept[class_idx]

    X_train_scaled = experiment.transform(X_train)
    X_test_scaled = experiment.transform(X_test)

    model = CreditScoreModel(
        coef=coef_std,
        intercept=intercept_std,
        features=features
    )

    return model, X_train_scaled, X_test_scaled


def evaluate_model(model, X_train, X_test, y_test):
    """
    Evaluate the credit scoring model.

    This function computes credit scores for the training and test datasets
    and evaluates prediction accuracy on the test set.

    Parameters
    ----------
    model : CreditScoreModel
        Trained credit scoring model.
    X_train : np.ndarray
        Transformed training feature matrix.
    X_test : np.ndarray
        Transformed test feature matrix.
    y_test : pd.Series or np.ndarray
        True labels for the test dataset.

    Returns
    -------
    tuple
        acc : float
            Classification accuracy on the test set.
        scores_train : np.ndarray
            Credit scores computed for the training dataset.
        scores_test : np.ndarray
            Credit scores computed for the test dataset.
    """

    scores_train = model.credit_score(X_train)
    scores_test = model.credit_score(X_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return acc, scores_train, scores_test
