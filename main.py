from src.data.load_data import load_data
from src.utils.utils import print_results
from src.modeling.trainer import Experiment
from src.modeling.score_pipeline import evaluate_model, score_dataset, train_score_model
from src.visualization.viz import plot_confusion_matrix, plot_real_vs_predicted, plot_score_distribution


def main():
    """
    Run the end-to-end credit scoring pipeline.

    The workflow includes:
    - Loading the cleaned dataset.
    - Splitting data into train and test sets.
    - Training a logistic-regression-based scorecard model.
    - Evaluating performance and generating predictions.
    - Visualizing confusion matrices and score distributions.
    - Saving scores for the complete dataset.
    """

    X, y = load_data("data/clean_train.csv")

    experiment = Experiment()

    # == Data Splitting ==

    X_train, X_test, y_train, y_test = experiment.split_data(X, y)

    # == Model Training ==

    model, X_train_scaled, X_test_scaled = train_score_model(
        experiment, X_train, y_train, X_test, X.columns)

    # == Model Evaluation ==

    acc, scores_train, scores_test, y_pred_train, y_pred_test = evaluate_model(
        model, X_train_scaled, X_test_scaled, y_test)

    # == Results and Visualization ==

    print_results(acc, scores_test, model)

    plot_confusion_matrix(y_train, y_pred_train, model_name="Train")
    plot_confusion_matrix(y_test, y_pred_test, model_name="Test")

    plot_score_distribution(scores_train, y_train, [
                            model.t1, model.t2], "Train")
    plot_score_distribution(scores_test, y_test, [model.t1, model.t2], "Test")

    plot_real_vs_predicted(scores_train, y_train, y_pred_train, "Train")
    plot_real_vs_predicted(scores_test, y_test, y_pred_test, "Test")

    # === Scores for Full Dataset ===
    score_dataset(X.copy(), model, experiment,
                  save_path="data/scores_full_dataset.csv")


if __name__ == "__main__":
    main()
