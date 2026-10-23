# CodeAlpha_CarPricePrediction

Task 3 of the **CodeAlpha Data Science Internship**: predict the resale price of a used vehicle with machine learning.

**Result in one line:** a Gradient Boosting model predicts resale price to within about **0.58 lakh INR (~58,000 rupees) on average** on vehicles it had never seen (R-squared 0.95), compared with 3.56 lakh for a model that always guesses the median price.

> Prices are in **lakh INR** (1 lakh = 100,000 rupees).
> **MAE** (mean absolute error) = the average size of the mistake, in lakh INR. **R-squared** = share of price variation the model explains (1.0 is perfect, 0 is no better than guessing the average).

## 1. Dataset
`data/car_data.csv`: 301 listings, 9 columns, no missing values.

| Column | Meaning |
|---|---|
| `Car_Name` | Vehicle model (98 distinct names) |
| `Year` | Year of manufacture (2003-2018) |
| `Selling_Price` | **Target**: resale price, lakh INR |
| `Present_Price` | Price of the same model new today, lakh INR |
| `Driven_kms` | Distance driven |
| `Fuel_Type`, `Selling_type`, `Transmission`, `Owner` | Petrol/Diesel/CNG, Dealer/Individual, Manual/Automatic, number of previous owners |

The task brief mentions horsepower and mileage (km/l). **This dataset has neither**, so the model uses the information it does have (new price, age, distance driven, fuel, brand and so on).

## 2. What I did
1. **Cleaning**: removed 2 exact duplicate rows (299 remain), fixed stray spaces in names, and checked that no vehicle sells for more than its new price (none does).
2. **Feature engineering**
   - `age` = 2018 - year (2018 is the newest year in the data, so results do not depend on today's date).
   - `vehicle_type` and `brand`: the file mixes **198 cars and 101 two-wheelers** (bikes and scooters) under 98 raw names. I mapped each name to a brand (Maruti Suzuki, Toyota, Honda, Hyundai, Bajaj, Hero, Royal Enfield, ...) and a vehicle type. This is the "brand goodwill" feature from the brief.
   - `owner_prev`: whether there was any previous owner (only 11 listings have one).
   - Prices and distance are log-transformed because they are heavily skewed (price skew 2.5 falls to 0.1 after the transform). The model predicts log price and converts back to lakh INR.
3. **Fair evaluation**: 20% of the data (60 vehicles) was set aside as a **test set** and not touched until the end. Models were compared with 5-fold cross-validation repeated 3 times on the other 80%.
4. **Models compared**: median-price baseline, Linear Regression, Ridge, Random Forest, Extra Trees, Gradient Boosting; the best families were then tuned with grid search.
5. **Final check** on the 60 unseen vehicles, with bootstrap confidence intervals, error breakdowns and permutation importance.

## 3. Results

**Model comparison** (cross-validation on the training set, default settings, errors in lakh INR):

| Model | MAE | R-squared |
|---|---|---|
| Baseline (median price) | 3.25 | -0.07 |
| Linear Regression | 0.72 | 0.92 |
| Ridge | 0.73 | 0.92 |
| Random Forest | 0.68 | 0.87 |
| Gradient Boosting | 0.64 | 0.88 |
| Extra Trees | 0.63 | 0.88 |

After tuning, **Gradient Boosting had the lowest cross-validated MAE (0.605)** and was selected.

**Final test on 60 unseen vehicles:**

| Metric | Value | 95% confidence interval |
|---|---|---|
| R-squared | **0.95** | 0.92 - 0.99 |
| MAE | **0.58 lakh** | 0.34 - 0.93 |
| RMSE | 1.31 lakh | |
| Baseline MAE | 3.56 lakh | |

Typical error by price band on the test set:

| Actual price | Vehicles | MAE (lakh) | Typical error (median % off) |
|---|---|---|---|
| Under 1 lakh | 16 | 0.07 | 17% |
| 1-4 lakh | 15 | 0.35 | 10% |
| 4-10 lakh | 22 | 0.54 | 7% |
| Over 10 lakh | 7 | 2.38 | 7% |

## 4. Key insights
- **The new price is by far the most important feature.** Shuffling it raises test error by 4.3 lakh, versus 0.76 for age and 0.24 for fuel type. Resale price is mainly "new price minus depreciation".
- **Depreciation is steep and similar for cars and two-wheelers**: a vehicle keeps about 90% of its new price after 1 year, about 70% after 3 years and about half after 5-6 years.
- **Brand goodwill exists but is small once the new price is known.** Adjusting for age, Royal Enfield, TVS and Hyundai hold their value best (+9% to +15% versus same-age peers), Toyota and Honda bikes least (about -10%). Removing brand from the model raises error only slightly (MAE 0.605 to 0.632), which is within the noise between folds, because the new price already captures most of what the brand signals.
- **Distance driven barely matters on its own** here. It is correlated with age (0.53), so age already carries most of the signal.
- **Transmission and seller type add essentially nothing** after the other features are known.
- **Where the model struggles:** the most expensive vehicles. The three largest test errors are Fortuners (the model under-predicted a 33-lakh listing by 8.4 lakh), and there are very few vehicles above 20 lakh to learn from. Percentage errors are also larger for cheap two-wheelers (median 18%), though that is only about 11,000 rupees.

## 5. Things to know (limitations)
- **Small dataset** (299 rows, only 60 in the test set), so the test score has a wide confidence interval. The cross-validated R-squared (about 0.90) is a more cautious estimate than the test R-squared (0.95); the test set contained few extreme-priced vehicles.
- **Linear Regression is better on the extremes.** It has the higher cross-validated R-squared (0.92) and lower RMSE because it extrapolates to very expensive cars, while tree models are better on typical vehicles. I selected the model by MAE because most listings are cheap or mid-priced; if your priority is expensive cars, a linear model is a reasonable alternative.
- The saved model was refit on all 299 rows, so the example in `predict.py` is an illustration, not an out-of-sample test. The reported metrics above are all from held-out data.
- CNG appears twice, and some brands appear once, so the model cannot learn them reliably.
- Predictions are for vehicles similar to those in this file; the data is a few years old and price levels will have moved.

## 6. Charts (`outputs/`)
| File | Shows |
|---|---|
| `01_price_distribution.png` | Why a log transform is used |
| `02_new_vs_resale_price.png` | Resale vs new price |
| `03_depreciation_by_age.png` | Value retained by age |
| `04_category_effects.png` | Fuel, transmission, seller and owner effects |
| `05_brand_goodwill.png` | Brand value retention, adjusted for age |
| `06_correlations.png` | Feature correlations |
| `07_model_comparison.png` | Model comparison |
| `08_predicted_vs_actual.png` | Test predictions and errors |
| `09_feature_importance.png` | What drives predictions |

Also: `metrics.json`, `model_comparison_cv.csv`, `test_predictions.csv`, `test_error_by_price_band.csv`, `permutation_importance.csv`, `analysis_log.txt` (full printed output) and the saved model `car_price_model.joblib`.

## 7. Run it
```bash
pip install -r requirements.txt
python car_price_prediction.py     # full pipeline, writes everything to outputs/
python predict.py                  # price one vehicle with the saved model (edit the dictionary inside)
```

## Files
- `car_price_prediction.py`: cleaning, features, EDA, modelling, evaluation
- `predict.py`: predict a price from the saved model
- `data/car_data.csv`: dataset
- `outputs/`: charts, tables, logs, saved model
- `requirements.txt`: dependencies
