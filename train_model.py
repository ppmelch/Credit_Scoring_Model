import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)


class CreditScoringPipeline:
    """
    End-to-end pipeline for training, tuning, and evaluating
    credit scoring classification models.
    """

    def __init__(self, model, scale_numeric=True):
        self.model = model
        self.scale_numeric = scale_numeric
        self.pipeline = None


    def _build_preprocessor(self, X):

        categorical_cols = X.select_dtypes(include="object").columns
        numeric_cols = X.select_dtypes(exclude="object").columns

        numeric_transformer = (
            StandardScaler() if self.scale_numeric else "passthrough"
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat",
                 OneHotEncoder(drop="first", handle_unknown="ignore"),
                 categorical_cols),
                ("num",
                 numeric_transformer,
                 numeric_cols)
            ]
        )

        return preprocessor


    def build_pipeline(self, X):

        preprocessor = self._build_preprocessor(X)

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

    def cross_validate(self, X, y, cv=5):

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


    def tune(self, X, y, param_dist, n_iter=20, cv=5):

        self.build_pipeline(X)

        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )

        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=skf,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        search.fit(X, y)

        self.pipeline = search.best_estimator_

        return search.best_params_, search.best_score_


    def fit(self, X, y):

        self.build_pipeline(X)
        self.pipeline.fit(X, y)

    def evaluate(self, X_test, y_test):

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


    def predict(self, X):

        return self.pipeline.predict(X)

    def save(self, path):

        joblib.dump(self.pipeline, path)