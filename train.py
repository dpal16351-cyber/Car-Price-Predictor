"""Notebook ka cleaning + training logic, script form me. Run: python train.py"""
import json
import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = ["name", "company", "fuel_type"]


def load_clean(path="quicker_car.csv") -> pd.DataFrame:
    car = pd.read_csv(path)
    car = car[car["year"].astype(str).str.isnumeric()].copy()
    car["year"] = car["year"].astype(int)
    car = car[car["Price"] != "Ask For Price"].copy()
    car["Price"] = car["Price"].str.replace(",", "").astype(int)
    car["kms_driven"] = car["kms_driven"].str.split(" ").str.get(0).str.replace(",", "")
    car = car[car["kms_driven"].str.isnumeric()].copy()
    car["kms_driven"] = car["kms_driven"].astype(int)
    car = car[~car["fuel_type"].isna()].copy()
    car["name"] = car["name"].str.split(" ").str.slice(0, 3).str.join(" ")
    car = car.reset_index(drop=True)
    car = car.drop(columns=["Unnamed: 0"], errors="ignore")
    car = car[car["Price"] < 6e6].reset_index(drop=True)
    return car


def main():
    car = load_clean()
    X, y = car.drop(columns="Price"), car["Price"]

    enc = OneHotEncoder(handle_unknown="ignore").fit(X[CAT_COLS])
    ct = make_column_transformer((enc, CAT_COLS), remainder="passthrough")
    pipe = make_pipeline(ct, LinearRegression())

    # fixed seed (notebook jaisa 1000 seeds me best chunna nahi, taaki score honest rahe)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    print(f"Test R2 (seed 42): {r2_score(y_te, pipe.predict(X_te)):.3f}")

    pipe.fit(X, y)  # final model poore data par
    joblib.dump(pipe, "model.joblib")

    # dropdown options for the API / UI
    meta = {
        "companies": sorted(car["company"].unique().tolist()),
        "fuel_types": sorted(car["fuel_type"].unique().tolist()),
        "models_by_company": {
            c: sorted(g["name"].unique().tolist()) for c, g in car.groupby("company")
        },
        "year_min": int(car["year"].min()),
        "year_max": int(car["year"].max()),
    }
    json.dump(meta, open("meta.json", "w"), indent=2)
    print(f"Saved model.joblib + meta.json ({len(car)} rows)")


if __name__ == "__main__":
    main()
