# 🚗 Nigerian Car Price Prediction

A machine learning project that estimates the market price of used cars in Nigeria (in Naira, ₦) from the make, year, condition, mileage, engine size, fuel type and transmission. It includes the full analysis notebook and a **Streamlit web app** for live predictions.

**Dataset:** [Nigerian Car Prices – Kaggle (segunadedipe)](https://www.kaggle.com/datasets/segunadedipe/nigerian-car-prices) — 4,095 car listings.

---

## Project structure

```
Car-price-prediction/
├── app.py                         # Streamlit interface
├── train.py                       # Cleans the data, trains and saves the model
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

A trained model is already included. To retrain it (for example after changing the cleaning steps), run `python train.py`. It reads `data/Nigerian_Car_Prices.csv` and overwrites `models/car_price_model.joblib`.

If the installed scikit-learn version differs from the one the model was saved with, the app retrains automatically from the CSV on start-up (a few seconds).

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Choose repository `BabajideAlao-knn/Car-price-prediction`, branch `main`, main file `app.py`.
4. Click **Deploy**. The app loads the saved model; nothing else to configure.

## Using the model in Python

```python
import joblib
from train import predict_price

bundle = joblib.load("models/car_price_model.joblib")
price = predict_price(bundle, make="Toyota", year=2010, condition="Foreign Used",
                      mileage=120_000, engine_size_cc=2400)
print(f"₦{price:,.0f}")
```

## Limitations and next steps
- The data has no **model/trim** (Camry vs Corolla), which hides a lot of price variation within a brand.
- No **location** or **listing date**, so prices aren't adjusted for Naira inflation/devaluation since the data was collected.
- Rare and luxury makes have fewer listings and larger errors.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost / LightGBM (notebook comparison) · Matplotlib / Seaborn · Streamlit

## Author
**Babajide Alao** — [GitHub](https://github.com/BabajideAlao-knn)
