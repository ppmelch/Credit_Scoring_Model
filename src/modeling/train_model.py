import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


class CreditScoringPipeline:
    """
    End-to-end pipeline for training and evaluating a credit scoring model.

    This class encapsulates preprocessing and model training using a
    scikit-learn Pipeline. It provides utilities for cross-validation,
    evaluation, prediction, and coefficient extraction from the trained model.
    """

    def __init__(self, model, scale_numeric: bool = True) -> None:
        """
        Initialize the pipeline.

        Parameters
        ----------
        model : sklearn estimator
            Machine learning model used for training (e.g., LogisticRegression).
        scale_numeric : bool, optional
            Whether to apply StandardScaler to numeric features.
        """
        self.model = model
        self.scale_numeric = scale_numeric
        self.pipeline = None

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create the preprocessing component of the pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset used to determine numeric columns.

        Returns
        -------
        ColumnTransformer
            Preprocessing transformer that scales numeric features.
        """
        numeric_cols = X.columns

        numeric_transformer = (
            StandardScaler() if self.scale_numeric else "passthrough"
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols)
            ]
        )

        return preprocessor

    def build_pipeline(self, X: pd.DataFrame) -> None:
        """
        Build the full training pipeline.

        The pipeline consists of:
            preprocessing → model training

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset used to configure preprocessing.
        """
        if self.pipeline is not None:
            return

        preprocessor = self._build_preprocessor(X)

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def cross_validate(self, X: pd.DataFrame, y: pd.Series | np.ndarray, cv: int = 5) -> tuple[float, float]:
        """
        Perform stratified cross-validation on the pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset.
        y : pd.Series or np.ndarray
            Target variable.
        cv : int, optional
            Number of cross-validation folds.

        Returns
        -------
        tuple
            Mean and standard deviation of the F1 macro score.
        """
        self.build_pipeline(X)

        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            self.pipeline,
            X,
            y,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1
        )

        return np.mean(scores), np.std(scores)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """
        Train the pipeline on the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset.
        y : pd.Series or np.ndarray
            Target variable.
        """
        self.build_pipeline(X)
        self.pipeline.fit(X, y)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series | np.ndarray) -> tuple[float, float, str]:
        """
        Evaluate model performance on the test dataset.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature dataset.
        y_test : pd.Series or np.ndarray
            True labels for the test dataset.

        Returns
        -------
        tuple
            accuracy : float
                Classification accuracy.
            auc : float
                Multi-class ROC AUC score.
            report : str
                Full classification report.
        """
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro"
        )

        report = classification_report(y_test, y_pred)

        return acc, auc, report

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        return self.pipeline.predict(X)

    def get_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract model coefficients and intercepts.

        Returns
        -------
        tuple
            coef : np.ndarray
                Model coefficients.
            intercept : np.ndarray
                Model intercept values.
        """
        model = self.pipeline.named_steps["model"]

        return model.coef_, model.intercept_

    def save(self, path: str) -> None:
        """
        Save the trained pipeline to disk.

        Parameters
        ----------
        path : str
            File path where the pipeline will be stored.
        """
        joblib.dump(self.pipeline, path)