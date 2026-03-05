import numpy as np


class CreditScoreModel:
    """
    Credit scoring model based on a logistic regression linear score.

    This class transforms the linear output of a trained logistic regression
    model into a normalized credit score between 300 and 850. It also provides
    classification of observations into credit categories using predefined
    score thresholds.
    """

    def __init__(self, coef, intercept, features):
        """
        Initialize the credit score model.

        Args:
            coef (np.ndarray): Coefficients of the logistic regression model
                corresponding to the selected class.
            intercept (float): Intercept term of the logistic regression model.
            features (list): List of feature names used by the model.
        """

        self.coef = coef
        self.intercept = intercept
        self.features = features

        # Score thresholds used for classification
        self.t1 = 569
        self.t2 = 645

    def score(self, X):
        """
        Compute the linear score (logit) of the model.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            np.ndarray: Linear combination of features and coefficients.
        """

        Xv = np.asarray(X)
        z = np.dot(Xv, self.coef) + self.intercept

        return z

    def credit_score(self, X):
        """
        Convert the model's linear score into a normalized credit score.

        The score is rescaled into the range [300, 850] using min-max
        normalization.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            np.ndarray: Credit scores between 300 and 850.
        """

        z = self.score(X)

        # Reverse sign so higher values correspond to better credit
        z = -z

        z_min = z.min()
        z_max = z.max()

        score_norm = (z - z_min) / (z_max - z_min)

        score_min = 300
        score_max = 850

        score = score_norm * (score_max - score_min) + score_min

        return score.astype(int)

    def predict(self, X):
        """
        Predict credit classes based on computed credit scores.

        Args:
            X (array-like): Input feature matrix.

        Returns:
            np.ndarray: Predicted credit class labels.
        """

        scores = self.credit_score(X)

        return np.array([self.classify(s) for s in scores])

    def classify(self, score):
        """
        Assign a credit category based on score thresholds.

        Args:
            score (int): Credit score value.

        Returns:
            int: Predicted class label
                 0 -> Poor
                 1 -> Standard
                 2 -> Good
        """

        if score < self.t1:
            return 1   # Standard

        elif score < self.t2:
            return 0   # Poor

        else:
            return 2   # Good
