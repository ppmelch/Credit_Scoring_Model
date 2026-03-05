from sklearn.metrics import accuracy_score
from src.modeling.score_model import CreditScoreModel


def train_score_model(experiment, X_train, y_train, X_test, class_idx, features):

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

    scores_train = model.credit_score(X_train)
    scores_test = model.credit_score(X_test)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return acc, scores_train, scores_test