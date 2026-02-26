from libraries import *
from data_cleaning import data_preprocessing

def main():

    df = pd.read_csv("train-3.csv" , low_memory=False)

    data = data_preprocessing(df , save_path="train.csv")

if __name__ == "__main__":
    main()