"""
Train the Nigerian car price model and save it to models/car_price_model.joblib.

Reproduces the cleaning, feature engineering and final model (tuned SVR)
from notebooks/Nigerian_Car_Price_Prediction.ipynb.

Usage:
    python train.py                      # uses data/Nigerian_Car_Prices.csv, or downloads from Kaggle
    python train.py --data path/to.csv   # use a specific CSV
"""
import argparse
import os
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "Nigerian_Car_Prices.csv"
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"
KAGGLE_DATASET = "segunadedipe/nigerian-car-prices"

RANDOM_STATE = 42
MIN_COUNT = 15
LUXURY = ["Lexus", "Mercedes-Benz", "BMW", "Land Rover", "Audi", "Infiniti", "Acura",
          "Porsche", "Rolls-Royce", "Jaguar", "Bentley", "Volvo"]
FEATURES_NUM = ["car_age", "mileage", "engine_size", "mileage_per_year", "is_luxury"]
FEATURES_CAT = ["make", "condition", "fuel", "transmission"]


def load_raw(csv_path=None) -> pd.DataFrame:
    """Load the raw CSV from a path, the data/ folder, or Kaggle (via kagglehub)."""
    if csv_path and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)
    import kagglehub  # only needed when no local CSV is present

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in Kaggle dataset {KAGGLE_DATASET}")
    return pd.read_csv(os.path.join(path, csv_files[0]))


def clean(raw: pd.DataFrame):
    """Same cleaning steps as the notebook. Returns (clean_df, all_raw_makes)."""
    df = raw.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.drop(columns=["Build"], errors="ignore")
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    df = df.rename(columns={"year_of_manufacture": "year"})

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})

    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    for c in ["year", "mileage", "engine_size"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().drop_duplicates()
    df["year"] = df["year"].astype(int)

    # Invalid / implausible values
    df = df[df["price"] > 100_000]
    df = df[df["mileage"] > 0]
    df = df[df["engine_size"].between(800, 8000)]
    cond_counts = df["condition"].value_counts()
    df = df[df["condition"].isin(cond_counts[cond_counts >= 30].index)]

    # Per-make price outliers on log(price), IQR rule
    df["log_price"] = np.log(df["price"])

    def iqr_flag(s, k=1.5):
        if len(s) < 10:
            return pd.Series(False, index=s.index)
        q1, q3 = s.quantile([0.25, 0.75])
        i = q3 - q1
        return (s < q1 - k * i) | (s > q3 + k * i)

    out = df.groupby("make", group_keys=False)["log_price"].apply(iqr_flag)
    df = df[~out.reindex(df.index, fill_value=False)]

    all_makes = sorted(df["make"].unique())

    # Group rare categories
    make_counts = df["make"].value_counts()
    rare = make_counts[make_counts < MIN_COUNT].index
    df["raw_make"] = df["make"]
    df["make"] = df["make"].where(~df["make"].isin(rare), "Other")
    for c in ["fuel", "transmission"]:
        vc = df[c].value_counts()
        small = vc[vc < MIN_COUNT].index
        df[c] = df[c].where(~df[c].isin(small), "Other")

    return df, all_makes


def add_features(df: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = ref_year - df["year"]
    df["mileage_per_year"] = df["mileage"] / df["car_age"].clip(lower=1)
    df["is_luxury"] = df["make"].isin(LUXURY).astype(int)  # after rare makes are grouped, as in the notebook
    return df


def build_model():
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocess), ("model", SVR(C=3, epsilon=0.05))]),
        func=np.log, inverse_func=np.exp,
    )


def train(csv_path=None, model_path=MODEL_PATH, verbose=True) -> dict:
    raw = load_raw(csv_path)
    df, all_makes = clean(raw)
    ref_year = int(df["year"].max()) + 1
    df = add_features(df, ref_year)

    X, y = df[FEATURES_NUM + FEATURES_CAT], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)

    # Evaluate on the 30% hold-out set, then refit on all data for deployment
    model = build_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    pct_err = np.abs(pred - y_test) / y_test
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "mape": float(mean_absolute_percentage_error(y_test, pred)),
        "within_20pct": float((pct_err <= 0.20).mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    final_model = build_model().fit(X, y)

    bundle = {
        "model": final_model,
        "ref_year": ref_year,
        "luxury": LUXURY,
        "model_makes": sorted(df["make"].unique()),     # categories the model knows (incl. "Other")
        "all_makes": all_makes,                          # every make seen after cleaning
        "conditions": sorted(df["condition"].unique()),
        "fuels": sorted(df["fuel"].unique()),
        "transmissions": sorted(df["transmission"].unique()),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
        "engine_range": (int(df["engine_size"].min()), int(df["engine_size"].max())),
        "mileage_median": float(df["mileage"].median()),
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "algorithm": "Support Vector Regression (C=3, epsilon=0.05) on log(price)",
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    if verbose:
        print(f"Clean rows: {len(df)} | reference year: {ref_year}")
        print(f"Hold-out R2: {metrics['r2']:.3f} | MAE: ₦{metrics['mae']:,.0f} | MAPE: {metrics['mape']*100:.1f}%")
        print(f"Saved model to {model_path}")
    return bundle


def predict_price(bundle, make, year, condition, mileage, engine_size_cc,
                  fuel="Petrol", transmission="Automatic") -> float:
    """Predict a single car's price in Naira."""
    ref_year = bundle["ref_year"]
    model_make = make if make in bundle["model_makes"] else "Other"
    age = ref_year - int(year)
    row = pd.DataFrame([{
        "car_age": age,
        "mileage": float(mileage),
        "engine_size": float(engine_size_cc),
        "mileage_per_year": float(mileage) / max(age, 1),
        "is_luxury": int(model_make in bundle["luxury"]),
        "make": model_make,
        "condition": condition,
        "fuel": fuel,
        "transmission": transmission,
    }])
    return float(bundle["model"].predict(row)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to Nigerian_Car_Prices.csv")
    parser.add_argument("--out", default=str(MODEL_PATH), help="Where to save the model")
    args = parser.parse_args()
    train(args.data, args.out)
