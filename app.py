"""
Streamlit app: Nigerian used-car price estimator.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "car_price_model.joblib"

st.set_page_config(page_title="Nigerian Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource(show_spinner=False)
def load_bundle():
    """Load the trained model and its metadata (saved from the notebook's final model)."""
    return joblib.load(MODEL_PATH)


def predict_price(bundle, make, year, condition, mileage, engine_size_cc,
                  fuel="Petrol", transmission="Automatic") -> float:
    """Predict a single car's price in Naira, using the same features as the notebook."""
    model_make = make if make in bundle["model_makes"] else "Other"
    age = bundle["ref_year"] - int(year)
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


def naira(v: float) -> str:
    return f"₦{v:,.0f}"


try:
    bundle = load_bundle()
except Exception as e:
    st.error(f"Couldn't load the model from `models/car_price_model.joblib`.\n\nDetails: {e}")
    st.stop()

metrics = bundle["metrics"]
y_min, y_max = bundle["year_range"]
e_min, e_max = bundle["engine_range"]

# ---------- Header ----------
st.title("🚗 Nigerian Car Price Predictor")
st.write(
    "Estimate the market price of a used car in Nigeria from its make, age, condition, "
    "mileage and engine size. Trained on listings from the "
    "[Nigerian Car Prices dataset](https://www.kaggle.com/datasets/segunadedipe/nigerian-car-prices)."
)

# ---------- Inputs ----------
with st.form("car"):
    c1, c2 = st.columns(2)
    makes = bundle["all_makes"]
    make = c1.selectbox("Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)
    year = c2.number_input("Year of manufacture", min_value=y_min, max_value=y_max,
                           value=min(max(2010, y_min), y_max), step=1)

    conditions = bundle["conditions"]
    condition = c1.selectbox("Condition", conditions,
                             help="Foreign Used = 'Tokunbo' (imported used); Nigerian Used = used locally.")
    mileage = c2.number_input("Mileage (km)", min_value=1, max_value=1_000_000,
                              value=int(round(bundle["mileage_median"], -3)), step=5_000)

    engine = c1.number_input("Engine size (cc)", min_value=max(800, e_min), max_value=min(8000, e_max),
                             value=2400, step=100, help="e.g. 2.4L engine = 2400 cc")
    transmissions = bundle["transmissions"]
    transmission = c2.selectbox("Transmission", transmissions,
                                index=transmissions.index("Automatic") if "Automatic" in transmissions else 0)

    fuels = bundle["fuels"]
    fuel = c1.selectbox("Fuel", fuels, index=fuels.index("Petrol") if "Petrol" in fuels else 0)

    submitted = st.form_submit_button("Estimate price", type="primary", width="stretch")

# ---------- Prediction ----------
if submitted:
    price = predict_price(bundle, make, int(year), condition, mileage, engine, fuel, transmission)
    mape = metrics["mape"]
    low, high = price * (1 - mape), price * (1 + mape)

    st.subheader("Estimated price")
    st.metric(label=f"{int(year)} {make} · {condition}", value=naira(price))
    st.caption(
        f"Typical range: **{naira(low)} – {naira(high)}** "
        f"(± the model's average error of {mape*100:.0f}%)."
    )
    if make not in bundle["model_makes"]:
        st.info(f"{make} has few listings in the data, so it's priced as part of an 'Other' brands group. "
                "Treat this estimate as rougher than usual.")
    if mileage / max(bundle["ref_year"] - int(year), 1) > 60_000:
        st.warning("That mileage is very high for the car's age, which is outside most of the training data.")

# ---------- About ----------
with st.expander("About the model"):
    st.markdown(
        f"""
- **Algorithm:** {bundle['algorithm']}
- **Accuracy on a 30% hold-out test set:** R² = {metrics['r2']:.2f}, average error (MAPE) ≈ {metrics['mape']*100:.0f}%,
  mean absolute error ≈ {naira(metrics['mae'])}; {metrics['within_20pct']*100:.0f}% of test cars were predicted within ±20%.
- **Biggest price drivers:** car age, luxury brand, engine size, make and condition. Mileage matters much less.
- **Limits:** the data has no model/trim (e.g. Camry vs Corolla), location or listing date, and prices reflect the
  period the data was collected, not today's Naira. Use the estimate as a guide, not a valuation.
"""
    )
