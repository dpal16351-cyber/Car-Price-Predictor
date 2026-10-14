from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
model = joblib.load(BASE / "model.joblib")
import json
meta = json.loads((BASE / "meta.json").read_text())

app = FastAPI(title="Car Price Predictor", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class CarInput(BaseModel):
    name: str = Field(..., examples=["Maruti Suzuki Swift"])
    company: str = Field(..., examples=["Maruti"])
    year: int = Field(..., ge=1990, le=2030, examples=[2019])
    kms_driven: int = Field(..., ge=0, examples=[40000])
    fuel_type: str = Field(..., examples=["Petrol"])


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    return meta


@app.post("/predict")
def predict(car: CarInput):
    df = pd.DataFrame([car.model_dump()],
                      columns=["name", "company", "year", "kms_driven", "fuel_type"])
    price = float(model.predict(df)[0])
    price = max(price, 0.0)
    return {"predicted_price_inr": round(price), "input": car.model_dump()}
