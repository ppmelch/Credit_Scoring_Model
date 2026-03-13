import pickle
import pandas as pd

def load_model():

    with open("models/credit_score_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


def predict_credit_score(data):

    model = load_model()

    df = pd.DataFrame([data])

    score = model.predict(df)

    return score