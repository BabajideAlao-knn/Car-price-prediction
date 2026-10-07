# 🚗 Nigerian Car Price Prediction

A machine learning project that estimates the market price of used cars in Nigeria (in Naira, ₦) from the make, year, condition, mileage, engine size, fuel type and transmission. It includes the full analysis notebook and a **Streamlit web app** for live predictions.

### 🔗 Live app: **[babajide-car-price-prediction.streamlit.app](https://babajide-car-price-prediction.streamlit.app/)**

**Dataset:** [Nigerian Car Prices – Kaggle (segunadedipe)](https://www.kaggle.com/datasets/segunadedipe/nigerian-car-prices) — 4,095 car listings.

---

## Project structure

```
Car-price-prediction/
├── app.py                         # Streamlit interface
├── notebooks/
│   └── Nigerian_Car_Price_Prediction.ipynb   # Full EDA + model comparison
├── data/Nigerian_Car_Prices.csv   # Raw dataset (4,095 listings)
├── models/car_price_model.joblib  # Trained model used by the app
├── requirements.txt
└── .streamlit/config.toml         # App theme
```

## Workflow (notebook)

1. **Data understanding** – types, missing values (e.g. `Build` is 72% empty), 879 duplicate rows.
2. **Cleaning** – convert text prices to numbers, drop rows with missing core fields, remove impossible values (engine size outside 800–8,000 cc, zero mileage), drop the tiny *Brand New* segment, remove per-make price outliers, group rare makes as *Other*. 2,586 clean rows remain.
3. **Feature engineering** – `car_age`, `mileage_per_year`, `is_luxury`, and a log-transformed price target.
4. **EDA** – price distribution, price by make/condition/year, correlations.
5. **Modelling** – 11 regressors compared on a 70/30 train-test split with 5-fold cross-validation, then the best one is tuned.
6. **Interpretation** – permutation feature importance and error analysis.

## Results

Test-set performance (30% hold-out):

| Model | R² | MAE (₦) | MAPE |
|---|---|---|---|
| **SVR (selected)** | **0.79** | **839,590** | **23.9%** |
| Gradient Boosting | 0.79 | 831,283 | 23.9% |
| XGBoost | 0.78 | 839,085 | 24.0% |
| Random Forest | 0.77 | 850,016 | 24.5% |
| Extra Trees | 0.76 | 882,356 | 25.0% |
| LightGBM | 0.72 | 878,827 | 25.0% |
| KNN | 0.69 | 926,292 | 25.4% |
| Decision Tree | 0.69 | 961,359 | 28.1% |
| Linear Regression | 0.67 | 936,592 | 25.5% |

The tuned **Support Vector Regressor** (C = 3, ε = 0.05, trained on log price) ranked best overall across test R², cross-validated R², MAE and MAPE. About 53% of test cars are predicted within ±20% of their actual price and 74% within ±30%.

### Key insights
- **Age is the main price driver** (~58% of feature importance); each extra year is associated with roughly 10% lower price.
- **Foreign Used ("Tokunbo") cars** sell at about a **1.8×** median premium over Nigerian Used cars.
- **Brand and engine size** add a strong premium (Lexus, Mercedes-Benz, Land Rover, BMW…).
- **Mileage matters less than expected**, likely because odometer readings are unreliable and overlap with age.
- Fuel and transmission add little: almost every listing is Petrol and Automatic.

## Run the app locally

```bash
git clone https://github.com/BabajideAlao-knn/Car-price-prediction.git
cd Car-price-prediction
pip install -r requirements.txt

streamlit run app.py     # opens the app at http://localhost:8501
```

The trained model is already included in `models/`. It was saved with scikit-learn 1.9.1, which is pinned in `requirements.txt` so the model loads correctly. If you retrain it in the notebook, save it with the same scikit-learn version or update the pin.

## Live deployment

The app is hosted on Streamlit Community Cloud: **https://babajide-car-price-prediction.streamlit.app/**

It redeploys automatically whenever changes are pushed to the `main` branch.

## Using the model in Python

```python
import joblib
import pandas as pd

bundle = joblib.load("models/car_price_model.joblib")
car = pd.DataFrame([{
    "car_age": bundle["ref_year"] - 2010,      # year of manufacture 2010
    "mileage": 120_000,
    "engine_size": 2400,
    "mileage_per_year": 120_000 / (bundle["ref_year"] - 2010),
    "is_luxury": 0,
    "make": "Toyota",
    "condition": "Foreign Used",
    "fuel": "Petrol",
    "transmission": "Automatic",
}])
print(f"₦{bundle['model'].predict(car)[0]:,.0f}")
```

## Limitations and next steps
- The data has no **model/trim** (Camry vs Corolla), which hides a lot of price variation within a brand.
- No **location** or **listing date**, so prices aren't adjusted for Naira inflation/devaluation since the data was collected.
- Rare and luxury makes have fewer listings and larger errors.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost / LightGBM (notebook comparison) · Matplotlib / Seaborn · Streamlit

## Author
**Babajide Alao** — [GitHub](https://github.com/BabajideAlao-knn)
