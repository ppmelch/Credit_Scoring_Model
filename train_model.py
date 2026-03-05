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

    def __init__(self, model, scale_numeric=True):

        self.model = model
        self.scale_numeric = scale_numeric
        self.pipeline = None


    def _build_preprocessor(self, X):

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


    def build_pipeline(self, X):

        if self.pipeline is not None:
            return

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


    def get_coefficients(self):

        model = self.pipeline.named_steps["model"]

        return model.coef_, model.intercept_


    def save(self, path):

        joblib.dump(self.pipeline, path)