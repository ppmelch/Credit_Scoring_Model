import logging

from libraries import pd , XGBClassifier
from train_model import CreditScoringPipeline
import mlflow
import mlflow.sklearn
logging.getLogger("mlflow").setLevel(logging.CRITICAL)


class Experiment:

    def __init__(self, version="v1"):

        self.version = version

        self.models = {
            "XGBoost": (
                XGBClassifier(
                    n_estimators=100,
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1
                ),
                False
            ),
        }
        self.results = {}
        self.best_model = None


    def run(self, X_train, X_test, y_train, y_test):

        mlflow.set_experiment(f"Credit_Scoring_{self.version}")

        for name, (model, scale_flag) in self.models.items():

            with mlflow.start_run(run_name=f"{name}_{self.version}"):

                pipeline = CreditScoringPipeline(
                    model,
                    scale_numeric=scale_flag
                )

                mean_cv, std_cv = pipeline.cross_validate(X_train, y_train)

                pipeline.fit(X_train, y_train)
                acc, auc, report = pipeline.evaluate(X_test, y_test)

                mlflow.log_param("model_name", name)
                mlflow.log_metric("cv_f1_macro_mean", mean_cv)
                mlflow.log_metric("test_accuracy", acc)
                mlflow.log_metric("test_auc", auc)

                mlflow.sklearn.log_model(
                    pipeline.pipeline,
                    name="model"
                )

                self.results[name] = {
                    "cv_mean": mean_cv,
                    "cv_std": std_cv,
                    "accuracy": acc,
                    "auc": auc,
                    "pipeline": pipeline   
                }

        self._select_best()
        return self.summary()


    def _select_best(self):

        self.best_model = max(
            self.results.items(),
            key=lambda x: x[1]["auc"]
        )[0]


    def summary(self):

        rows = []

        for name, metrics in self.results.items():
            rows.append({
                "Model": name,
                "CV_F1_macro": metrics["cv_mean"],
                "Test_Accuracy": metrics["accuracy"],
                "Test_AUC": metrics["auc"]
            })

        return pd.DataFrame(rows).sort_values(
            by="Test_AUC",
            ascending=False
        )


    def save_best(self, path):

        if self.best_model is None:
            raise ValueError("No best model selected. Run experiment first.")

        best_pipeline = self.results[self.best_model]["pipeline"]
        best_pipeline.save(path)

        print(f"Best model saved: {self.best_model} → {path}")