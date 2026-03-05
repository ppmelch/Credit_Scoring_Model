from src.data.load_data import load_data
from src.modeling.score_training import evaluate_model, train_score_model
from src.utils.utils import print_results
from src.modeling.experiment import Experiment
from src.visualization.viz import plot_score_distribution

def main():

    class_idx = 1

    X, y = load_data("data/clean_train.csv")

    experiment = Experiment()

    X_train, X_test, y_train, y_test = experiment.split_data(X, y)

    model, X_train_scaled, X_test_scaled = train_score_model(
        experiment,
        X_train,
        y_train,
        X_test,
        class_idx,
        X.columns
    )

    acc, scores_train, scores_test = evaluate_model(
        model,
        X_train_scaled,
        X_test_scaled,
        y_test
    )

    print_results(acc, scores_test, model)

    plot_score_distribution(
        scores_train,
        y_train,
        model_name="Standard Score Model"
    )


if __name__ == "__main__":
    main()