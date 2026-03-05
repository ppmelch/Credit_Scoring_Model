from src.data.load_data import load_data
from src.utils.utils import print_results
from src.modeling.trainer import Experiment
from src.visualization.viz import plot_score_distribution
from src.modeling.score_training import evaluate_model, train_score_model


def main():
    """
    Run the complete credit scoring pipeline.

    This function orchestrates the full workflow of the project:

    1. Load and preprocess the dataset.
    2. Split the data into training and testing sets.
    3. Train the credit scoring model using a logistic regression pipeline.
    4. Transform the features using the fitted preprocessing pipeline.
    5. Generate credit scores for the training and testing datasets.
    6. Evaluate model performance using classification accuracy.
    7. Print summary results.
    8. Visualize the score distribution for the training data.

    The model is trained using the class index corresponding to the
    "Standard" credit category.
    """

    # Index corresponding to the "Standard" class
    class_idx = 1

    # Load dataset and target variable
    X, y = load_data("data/clean_train.csv")

    # Initialize experiment configuration and model
    experiment = Experiment()

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = experiment.split_data(X, y)

    # Train the credit scoring model and transform features
    model, X_train_scaled, X_test_scaled = train_score_model(
        experiment,
        X_train,
        y_train,
        X_test,
        class_idx,
        X.columns
    )

    # Evaluate model performance
    acc, scores_train, scores_test = evaluate_model(
        model,
        X_train_scaled,
        X_test_scaled,
        y_test
    )

    # Print evaluation results
    print_results(acc, scores_test, model)

    # Plot score distribution for training data
    plot_score_distribution(
        scores_train,
        y_train,
        model_name="Standard Score Model"
    )


if __name__ == "__main__":
    main()
