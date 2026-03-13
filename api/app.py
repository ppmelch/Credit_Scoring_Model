from fastapi import FastAPI
from src.inference.predict import predict_credit_score

app = FastAPI()

@app.post("/predict")
def predict(data: dict):

    score = predict_credit_score(data)

    return {"credit_score": score}