import logging

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.utils.utils import pd
from src.modeling.train_model import CreditScoringPipeline
import mlflow
import mlflow.sklearn

logging.getLogger("mlflow").setLevel(logging.CRITICAL)


class Experiment:

    def __init__(self, version="v1"):

        self.version = version

        self.model = LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            random_state=42
        )

        self.pipeline = None
        self.coef = None
        self.intercept = None


    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        return X_train, X_test, y_train, y_test


    def run(self, X_train, y_train):

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
    

    def transform(self, X):

        scaler = self.pipeline.pipeline.named_steps["preprocessor"]

        X_scaled = scaler.transform(X)

        return X_scaled