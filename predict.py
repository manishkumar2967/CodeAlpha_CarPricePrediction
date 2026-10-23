"""Predict a used-vehicle resale price with the saved model.

Usage:  python predict.py
Edit the dictionary below (prices in lakh INR; 1 lakh = 100,000 INR) and run the file.
"""
import joblib
import pandas as pd

bundle = joblib.load("outputs/car_price_model.joblib")
model, FEATURES, REF_YEAR = bundle["model"], bundle["features"], bundle["ref_year"]

vehicle = {
    "Present_Price": 9.85,        # price of the same model new today, lakh INR
    "Driven_kms": 6900,
    "Year": 2017,                 # manufacturing year
    "Owner": 0,                   # number of previous owners
    "Fuel_Type": "Petrol",        # Petrol / Diesel / CNG
    "Selling_type": "Dealer",     # Dealer / Individual
    "Transmission": "Manual",     # Manual / Automatic
    "vehicle_type": "Car",        # Car / Two-wheeler
    "brand": "Maruti Suzuki",     # Maruti Suzuki, Toyota, Honda, Hyundai, Bajaj, Hero, Royal Enfield, ...
}
row = dict(vehicle)
row["age"] = REF_YEAR - row.pop("Year")
row["owner_prev"] = int(row.pop("Owner") > 0)
X = pd.DataFrame([row])[FEATURES]
print(f"Estimated resale price: {model.predict(X)[0]:.2f} lakh INR")
