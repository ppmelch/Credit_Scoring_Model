import os
import pickle
from src.data.load_data import load_data
from src.utils.utils import print_results
from src.modeling.trainer import Experiment
from src.modeling.score_pipeline import evaluate_model, score_dataset, train_score_model
from src.visualization.viz import plot_confusion_matrix, plot_real_vs_predicted, plot_score_distribution

def run_pipeline():

    X, y = load_data("data/clean_train.csv")

    experiment = Experiment()

    # Data Splitting
    X_train, X_test, y_train, y_test = experiment.split_data(X, y)

    # Model Training
    model, X_train_scaled, X_test_scaled = train_score_model(
        experiment, X_train, y_train, X_test, X.columns)

    # Save model
    os.makedirs("models", exist_ok=True)

    with open("models/credit_score_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Model Evaluation
    acc, scores_train, scores_test, y_pred_train, y_pred_test = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_test)

    print_results(acc, model)

    plot_confusion_matrix(y_train, y_pred_train, model_name="Train")
    plot_confusion_matrix(y_test, y_pred_test, model_name="Test")

    plot_score_distribution(scores_train, y_train, [model.t1, model.t2], "Train")
    plot_score_distribution(scores_test, y_test, [model.t1, model.t2], "Test")

    plot_real_vs_predicted(scores_train, y_train, y_pred_train, "Train")
    plot_real_vs_predicted(scores_test, y_test, y_pred_test, "Test")

    score_dataset(
        X.copy(),
        model,
        experiment,
        save_path="outputs/scores_full_dataset.csv"
    )

    return model

