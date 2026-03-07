import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix


sns.set_theme(style="whitegrid", palette="Greys_r")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100

red_grey = mcolors.LinearSegmentedColormap.from_list(
    "RedGrey",
    ["#bfbfbf96", "#c60f0f"])


def plot_confusion_matrix(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, class_names: tuple[str, str, str] = ("Poor", "Standard", "Good"), model_name: str = "Score Model") -> None:
    """
    Plot the confusion matrix of predicted vs true classes.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        True class labels.
    y_pred : np.ndarray
        Predicted class labels.
    model_name : str
        Name of the model for the plot title.
    """

    matrix = confusion_matrix(y_true, y_pred)
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Greys", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_score_distribution(score_values, true_labels, score_thresholds, dataset_name="Train"):

    score_df = pd.DataFrame({
        "Score": score_values,
        "True_Class": true_labels
    })

    class_map = {
        0: "Poor",
        1: "Standard",
        2: "Good"
    }

    # -------- GENERAL DISTRIBUTION --------
    plt.figure(figsize=(10,5))

    sns.kdeplot(
        x=score_df["Score"],
        fill=True,
        alpha=0.3,
        linewidth=2
    )

    plt.axvline(score_thresholds[0], linestyle="--", color="gray", label="t1")
    plt.axvline(score_thresholds[1], linestyle="--", color="black", label="t2")

    plt.title(f"{dataset_name} Distribution of Credit Score")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()

    plt.show()


    # -------- DISTRIBUTION PER CLASS (LAS 3 JUNTAS) --------
    plt.figure(figsize=(10,5))

    for class_id, class_name in class_map.items():

        sns.kdeplot(
            x=score_df.loc[score_df["True_Class"] == class_id, "Score"],
            fill=True,
            alpha=0.3,
            linewidth=2,
            label=class_name
        )

    plt.axvline(score_thresholds[0], linestyle="--", color="gray")
    plt.axvline(score_thresholds[1], linestyle="--", color="black")

    plt.title(f"{dataset_name} Score Distribution per Class")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()

    plt.show()


    # -------- INDIVIDUAL CLASS DISTRIBUTIONS --------
    for class_id, class_name in class_map.items():

        plt.figure(figsize=(10,5))

        sns.kdeplot(
            x=score_df.loc[score_df["True_Class"] == class_id, "Score"],
            fill=True,
            alpha=0.3,
            linewidth=2,
            bw_adjust=1
        )

        plt.axvline(score_thresholds[0], linestyle="--", color="gray", label="t1")
        plt.axvline(score_thresholds[1], linestyle="--", color="black", label="t2")

        plt.title(f"{dataset_name} Score Distribution - {class_name}")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()

        plt.show()

def plot_real_vs_predicted(score_values: np.ndarray, true_labels: pd.Series | np.ndarray, predicted_labels: np.ndarray, dataset_name: str = "Train") -> None:
    """
    Compare real and predicted score distributions for each class.

    Parameters
    ----------
    score_values : np.ndarray
        Credit scores produced by the model.
    true_labels : array-like
        True class labels.
    predicted_labels : np.ndarray
        Predicted class labels.
    dataset_name : str
        Dataset identifier (Train/Test).
    """

    results_df = pd.DataFrame({
        "Score": score_values,
        "Real_Class": true_labels,
        "Predicted_Class": predicted_labels
    })

    label_names = {
        0: "Poor",
        1: "Standard",
        2: "Good"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for class_id in range(3):

        real_dist = results_df[results_df["Real_Class"] == class_id]["Score"]
        pred_dist = results_df[results_df["Predicted_Class"]
                               == class_id]["Score"]

        sns.kdeplot(
            real_dist,
            ax=axes[class_id],
            label="Real",
            fill=True,
            alpha=0.3
        )

        sns.kdeplot(
            pred_dist,
            ax=axes[class_id],
            label="Predicted",
            linestyle="--"
        )

        axes[class_id].set_title(f"{dataset_name} - {label_names[class_id]}")
        axes[class_id].set_xlabel("Score")
        axes[class_id].set_ylabel("Density")
        axes[class_id].legend()

    plt.tight_layout()
    plt.show()
