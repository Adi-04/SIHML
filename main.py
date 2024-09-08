from fastapi import FastAPI
from model.model import predict_pipeline

app = FastAPI()

@app.get("/")
def home():
    return "Everything OK!"

@app.post("/predict")
def predict(data: dict):
    result = predict_pipeline(data)
    return result