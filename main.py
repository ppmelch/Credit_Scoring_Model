from src.data.load_data import load_data
from src.utils.utils import print_results
from src.modeling.trainer import Experiment
from src.modeling.score_pipeline import evaluate_model, score_dataset, train_score_model
from src.visualization.viz import plot_confusion_matrix, plot_real_vs_predicted, plot_score_distribution

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
        X.columns
    )

    # Evaluate model performance
    acc, scores_train, scores_test, y_pred_train, y_pred_test = evaluate_model(
        model,
        X_train_scaled,
        X_test_scaled,
        y_test
    )

    print_results(acc, scores_test, model)
    
    plot_confusion_matrix(y_train, y_pred_train, model_name="Train")
    plot_confusion_matrix(y_test, y_pred_test, model_name="Test")

    plot_score_distribution(scores_train, y_train, thresholds=[model.t1, model.t2], set_name="Train")
    plot_score_distribution(scores_test, y_test, thresholds=[model.t1, model.t2], set_name="Test")
    
    plot_real_vs_predicted(scores_train, y_train, y_pred_train, "Train")
    plot_real_vs_predicted(scores_test, y_test, y_pred_test, "Test")
    
    score_dataset(X.copy(), model, experiment, save_path="data/scores_full_dataset.csv")
    
if __name__ == "__main__":
    main()
