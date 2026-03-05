import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.modeling.train_model import CreditScoringPipeline


class Experiment:
    """
    Class responsible for running model training experiments.

    This class manages the full lifecycle of a training experiment:
    - model initialization
    - data splitting
    - pipeline training
    - coefficient extraction
    - feature transformation using the trained preprocessing pipeline
    """

    def __init__(self, version="v1"):
        """
        Initialize the experiment configuration.

        Args:
            version (str, optional): Version identifier for the experiment.
                Useful when tracking experiments with MLflow or versioning models.
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

    def split_data(self, X, y):
        """
        Split the dataset into training and testing subsets.

        The split preserves the class distribution using stratified sampling.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.ndarray): Target labels.

        Returns:
            tuple:
                X_train (pd.DataFrame): Training features
                X_test (pd.DataFrame): Testing features
                y_train (pd.Series): Training labels
                y_test (pd.Series): Testing labels
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def run(self, X_train, y_train):
        """
        Train the credit scoring pipeline.

        This method builds the preprocessing pipeline, fits the logistic
        regression model, and extracts the learned coefficients.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series or np.ndarray): Training target labels.

        Returns:
            tuple:
                coef (np.ndarray): Model coefficients for each class.
                intercept (np.ndarray): Intercept terms for each class.
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

    def transform(self, X):
        """
        Apply the trained preprocessing pipeline to new data.

        This method uses the fitted preprocessing step (e.g., scaling)
        to transform input features before scoring.

        Args:
            X (pd.DataFrame or array-like): Feature matrix to transform.

        Returns:
            np.ndarray: Transformed feature matrix.
        """

        return self.pipeline.pipeline.named_steps["preprocessor"].transform(X)
