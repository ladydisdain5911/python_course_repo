import joblib
import uvicorn
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Real Estate Price Prediction API")

MODEL_PATH = "linear_model.pkl"


def generate_data():
    df = pd.read_csv("realty_data.csv")
    data = df[["price", "floor", "total_square"]]
    return data


def train_model(data):
    X = data[["total_square", "floor"]]
    y = data["price"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


if not os.path.exists(MODEL_PATH):
    model = train_model(generate_data())
else:
    model = joblib.load(MODEL_PATH)


class ModelRequestData(BaseModel):
    total_square: float
    floor: float


class Result(BaseModel):
    predicted_price: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.post("/predict", response_model=Result)
def predict(data: ModelRequestData):
    input_df = pd.DataFrame(data.dict(), index=[0])
    prediction = model.predict(input_df)[0]
    return Result(predicted_price=round(prediction, 2))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
