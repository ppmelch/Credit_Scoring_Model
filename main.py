from libraries import pd , warnings 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_cleaning import data_preprocessing
from experiment import Experiment

warnings.filterwarnings("ignore")


def main():

    df = pd.read_csv("data/train-3.csv", low_memory=False)
    clean_df = data_preprocessing(df)
    
    X = clean_df.drop("Credit_Score", axis=1)
    y = clean_df["Credit_Score"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    experiment = Experiment(version="v2")

    results = experiment.run(X_train, X_test, y_train, y_test)

    print(results)

    experiment.save_best("best_model.pkl")


if __name__ == "__main__":
    main()