import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.modeling.train_model import CreditScoringPipeline


class Experiment:
    """
    Manage a credit scoring training experiment.

    This class handles dataset splitting, model training through a
    preprocessing pipeline, and storage of learned model parameters.
    """

    def __init__(self, version: str = "v1") -> None:
        """
        Initialize experiment settings.

        Parameters
        ----------
        version : str, optional
            Version identifier for the experiment (default: "v1").
        """
        self.version = version

        self.model = LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            random_state=42
        )

        self.pipeline = None
        self.coef = None
        self.intercept = None

    def split_data(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.

        Stratified sampling is used to preserve the class distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or np.ndarray
            Target labels.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def run(self, X_train: pd.DataFrame, y_train: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Train the credit scoring pipeline.

        The method fits the preprocessing pipeline and logistic regression
        model, then extracts the learned coefficients.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series or np.ndarray
            Training target labels.

        Returns
        -------
        tuple
            coef : np.ndarray
                Model coefficients.
            intercept : np.ndarray
                Model intercepts.
        """
        pipeline = CreditScoringPipeline(
            self.model,
            scale_numeric=True
        )

        pipeline.fit(X_train, y_train)

        coef, intercept = pipeline.get_coefficients()

        self.pipeline = pipeline
        self.coef = coef
        self.intercept = intercept

        return coef, intercept

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted preprocessing pipeline.

        Parameters
        ----------
        X : pd.DataFrame or array-like
            Input features.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        return self.pipeline.pipeline.named_steps["preprocessor"].transform(X)
