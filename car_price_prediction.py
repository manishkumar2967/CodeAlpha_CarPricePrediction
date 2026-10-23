"""
CodeAlpha Data Science Internship - Task 3: Car Price Prediction with Machine Learning
Author: <your name>

Goal: predict the resale price (Selling_Price, in Indian lakh rupees; 1 lakh = 100,000 INR)
of a used vehicle from its specifications and history.

Pipeline
  1. Load, clean and validate the data
  2. Feature engineering (age, brand, vehicle type, log transforms)
  3. Exploratory analysis (charts 01-06)
  4. Hold out a test set, compare models with repeated cross-validation
  5. Tune the best candidates, check what brand information adds (ablation)
  6. Evaluate once on the untouched test set, with bootstrap confidence intervals
  7. Explain the model (permutation importance, error by segment) and save it
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (train_test_split, RepeatedKFold, cross_validate, GridSearchCV)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
REF_YEAR = 2018          # newest model year in the data; "age" = REF_YEAR - Year (independent of today's date)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False,
})
log = []


def note(msg=""):
    print(msg)
    log.append(str(msg))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ====================================================================== 1. LOAD + CLEAN
note("=== 1. DATA CLEANING ===")
df = pd.read_csv("data/car_data.csv")
note(f"Raw shape: {df.shape}; missing values: {int(df.isna().sum().sum())}")
df["Car_Name"] = df["Car_Name"].str.strip().str.replace(r"\s+", " ", regex=True)   # e.g. 'Bajaj  ct 100'
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
note(f"Dropped {n_dup} exact duplicate rows -> {len(df)} rows")
assert (df["Selling_Price"] <= df["Present_Price"]).all(), "resale price above new price?"
note("Sanity check passed: no vehicle sells for more than its current new (showroom) price.")
note(f"Extreme values kept (real vehicles, handled with log transforms): max Driven_kms = {df.Driven_kms.max():,}, "
     f"max Present_Price = {df.Present_Price.max():.1f} lakh (Land Cruiser)")

# ====================================================================== 2. FEATURE ENGINEERING
CAR_MAKE = {}
for make, names in {
    "Maruti Suzuki": ["ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza", "s cross", "alto 800",
                      "alto k10", "ignis", "800", "baleno", "omni", "ertiga", "dzire"],
    "Toyota": ["fortuner", "innova", "corolla altis", "etios cross", "etios g", "etios gd", "etios liva",
               "corolla", "camry", "land cruiser"],
    "Honda": ["city", "amaze", "brio", "jazz"],
    "Hyundai": ["creta", "verna", "i10", "grand i10", "i20", "eon", "xcent", "elantra"],
}.items():
    for n in names:
        CAR_MAKE[n] = make

TWO_WHEELER_PREFIX = [("Bajaj", "Bajaj"), ("Hero", "Hero"), ("Honda", "Honda Bikes"), ("Activa", "Honda Bikes"),
                      ("Royal Enfield", "Royal Enfield"), ("KTM", "KTM"), ("TVS", "TVS"), ("Yamaha", "Yamaha"),
                      ("Suzuki", "Suzuki Bikes"), ("Mahindra", "Mahindra Bikes"), ("UM", "UM"), ("Hyosung", "Hyosung")]


def vehicle_info(name):
    """Car names in this file are lower-case ('city', '800'); two-wheelers start with a capital letter."""
    if name in CAR_MAKE:
        return "Car", CAR_MAKE[name]
    for prefix, make in TWO_WHEELER_PREFIX:
        if name.startswith(prefix):
            return "Two-wheeler", make
    raise ValueError(f"Unmapped vehicle: {name}")


df[["vehicle_type", "brand"]] = df["Car_Name"].apply(lambda n: pd.Series(vehicle_info(n)))
df["age"] = REF_YEAR - df["Year"]
df["owner_prev"] = (df["Owner"] > 0).astype(int)       # only 11 of 299 listings have previous owners; 3 owners occurs once
df["retention"] = df["Selling_Price"] / df["Present_Price"]   # EDA only: NOT used as a model input (it contains the target)

note("\n=== 2. FEATURE ENGINEERING ===")
note("vehicle_type counts: " + str(df.vehicle_type.value_counts().to_dict()))
note("brand counts: " + str(df.brand.value_counts().to_dict()))
note("The file mixes cars and two-wheelers; the 98 raw Car_Name values were reduced to brand + vehicle_type.")
note("The dataset has no horsepower or mileage (km/l) column; Driven_kms is the usage measure.")

# ====================================================================== 3. EDA
note("\n=== 3. EXPLORATORY ANALYSIS ===")
note(df[["Selling_Price", "Present_Price", "Driven_kms", "age"]].describe().round(2).to_string())
note(f"Selling_Price skew: {df.Selling_Price.skew():.2f}; after log1p: {np.log1p(df.Selling_Price).skew():.2f}")
note(f"Correlation of Selling_Price with Present_Price: {df.Selling_Price.corr(df.Present_Price):.2f}; "
     f"with age: {df.Selling_Price.corr(df.age):.2f}; with Driven_kms: {df.Selling_Price.corr(df.Driven_kms):.2f}")

# 01 price distribution
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].hist(df.Selling_Price, bins=30, color=BLUE, edgecolor="white")
axs[0].set_title("Selling price is heavily right-skewed")
axs[0].set_xlabel("Selling price (lakh INR)"); axs[0].set_ylabel("Vehicles")
axs[1].hist(np.log1p(df.Selling_Price), bins=30, color=ORANGE, edgecolor="white")
axs[1].set_title("After a log transform it is far more symmetric")
axs[1].set_xlabel("log(1 + selling price)")
save(fig, "01_price_distribution.png")

# 02 new price vs resale price
fig, ax = plt.subplots(figsize=(7.5, 6))
for vt, c in [("Two-wheeler", ORANGE), ("Car", BLUE)]:
    d = df[df.vehicle_type == vt]
    ax.scatter(d.Present_Price, d.Selling_Price, s=30, color=c, alpha=0.8, edgecolor="white", linewidth=0.7, label=vt)
lim = [0.1, 100]
ax.plot(lim, lim, color=MUTED, ls=":", lw=1)
ax.text(0.3, 1.4, "dotted line:\nresale = new price", color=MUTED, fontsize=8.5, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.25, 100); ax.set_ylim(0.1, 50)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("Present (new) price, lakh INR - log scale"); ax.set_ylabel("Selling price, lakh INR - log scale")
ax.set_title("Resale price tracks the new price closely (r = 0.88)")
ax.legend(loc="upper left")
save(fig, "02_new_vs_resale_price.png")

# 03 depreciation with age
fig, ax = plt.subplots(figsize=(9, 4.8))
for vt, c in [("Car", BLUE), ("Two-wheeler", ORANGE)]:
    g = df[df.vehicle_type == vt].groupby("age")["retention"].agg(["median", "size"])
    g = g[g["size"] >= 3]
    ax.plot(g.index, g["median"] * 100, color=c, lw=2, marker="o", ms=5, label=f"{vt} (median)")
ax.set_xlabel("Vehicle age (years, relative to 2018)")
ax.set_ylabel("Resale price as % of new price")
ax.set_ylim(0, 100)
ax.set_title("Vehicles lose about half of their new price within 5-6 years")
ax.legend()
save(fig, "03_depreciation_by_age.png")
dep = df.groupby(["vehicle_type", "age"])["retention"].median().unstack(0).round(2)
note("\nMedian resale/new price by age (cars | two-wheelers):\n" + dep.head(12).to_string())

# 04 categorical effects on retention
cats = [("Fuel_Type", "Fuel type"), ("Transmission", "Transmission"), ("Selling_type", "Seller type"),
        ("owner_prev", "Previous owners")]
fig, axs = plt.subplots(1, 4, figsize=(14, 4))
for ax, (col, title) in zip(axs, cats):
    g = df.groupby(col)["retention"].agg(["median", "size"])
    labels = [("None" if k == 0 else "1+") if col == "owner_prev" else str(k) for k in g.index]
    ax.bar(labels, g["median"] * 100, color=BLUE, width=0.55)
    for i, (m, n) in enumerate(zip(g["median"], g["size"])):
        ax.text(i, m * 100 + 1.5, f"{m * 100:.0f}%\nn={n}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 95); ax.set_title(title, fontsize=11)
    ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Median resale / new price (%)")
fig.suptitle("Resale value retained, by category (small groups, e.g. CNG n=2, are unreliable)", x=0.01, ha="left",
             fontweight="bold", fontsize=12, y=1.02)
save(fig, "04_category_effects.png")
for col, _ in cats:
    note(f"Median retention by {col}: " + str(df.groupby(col)["retention"].median().round(2).to_dict()))

# 05 brand goodwill (resale retention by brand)
# age-adjusted retention: brand average of (log retention minus what the vehicle type's age curve predicts)
df["log_ret"] = np.log(df["retention"])
df["ret_resid"] = 0.0
for _vt, _g in df.groupby("vehicle_type"):
    _b = np.polyfit(_g["age"], _g["log_ret"], 1)
    df.loc[_g.index, "ret_resid"] = _g["log_ret"] - np.polyval(_b, _g["age"])
bg = df.groupby("brand").agg(retention=("retention", "median"), n=("retention", "size"), vt=("vehicle_type", "first"),
                             med_age=("age", "median"), adj=("ret_resid", "mean"))
bg = bg[bg.n >= 5].copy()
bg["vs_peers_pct"] = (np.exp(bg["adj"]) - 1) * 100       # % above/below the typical resale value of same-age vehicles of that type
bg = bg.sort_values("vs_peers_pct")
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh(bg.index, bg.vs_peers_pct, color=[BLUE if v == "Car" else ORANGE for v in bg.vt], height=0.65)
ax.axvline(0, color=MUTED, lw=1)
for i, (v, r, n) in enumerate(zip(bg.vs_peers_pct, bg.retention, bg.n)):
    ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{int(round(v)):+d}%   (n={n}, keeps {r * 100:.0f}% of new price)" if round(v) != 0 else f"0%   (n={n}, keeps {r * 100:.0f}% of new price)",
            va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
ax.set_xlim(-45, 50)
ax.set_xlabel("Resale value vs typical same-age vehicles of the same type (%)")
ax.set_title("Brand goodwill, adjusted for age (blue = cars, orange = two-wheelers)", fontsize=11)
ax.grid(axis="y", visible=False)
save(fig, "05_brand_goodwill.png")
note("\nBrand resale retention (brands with >= 5 listings); adj = mean log-retention vs the age curve of its vehicle type (+ = holds value better than same-age peers):\n" + (bg[["retention", "n", "med_age", "adj"]].round(2)).to_string())

# 06 correlations
num = df[["Selling_Price", "Present_Price", "Driven_kms", "age", "Owner"]].copy()
num["log_Selling_Price"] = np.log1p(num.pop("Selling_Price"))
num["log_Present_Price"] = np.log1p(num.pop("Present_Price"))
num["log_Driven_kms"] = np.log1p(num.pop("Driven_kms"))
cm = num[["log_Selling_Price", "log_Present_Price", "log_Driven_kms", "age", "Owner"]].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(cm.columns, rotation=30, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(cm.columns)
ax.grid(False)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(cm.values[i, j]) > 0.6 else INK)
ax.set_title("Correlations (log scale for prices and distance)")
fig.colorbar(im, fraction=0.046, pad=0.04)
save(fig, "06_correlations.png")

# ====================================================================== 4. SPLIT + MODEL COMPARISON
note("\n=== 4. MODEL COMPARISON ===")
FEATURES = ["Present_Price", "Driven_kms", "age", "owner_prev", "Fuel_Type", "Selling_type",
            "Transmission", "vehicle_type", "brand"]
NUM = ["Present_Price", "Driven_kms", "age", "owner_prev"]
CAT = ["Fuel_Type", "Selling_type", "Transmission", "vehicle_type", "brand"]
X, y = df[FEATURES], df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=df["vehicle_type"])
note(f"Train {len(X_train)} rows / test {len(X_test)} rows (stratified by vehicle type; test set is untouched until step 6)")


def make_preprocessor(features_cat, scale=True):
    num_steps = [("log", FunctionTransformer(np.log1p, validate=False))]
    # log-transform prices and distance; age and owner are used as-is
    num_pipe_log = Pipeline(num_steps + ([("sc", StandardScaler())] if scale else []))
    num_pipe_raw = Pipeline([("sc", StandardScaler())] if scale else [("id", FunctionTransformer(validate=False))])
    return ColumnTransformer([
        ("num_log", num_pipe_log, ["Present_Price", "Driven_kms"]),
        ("num_raw", num_pipe_raw, ["age", "owner_prev"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), features_cat),
    ])


def make_model(regressor, scale=True, features_cat=CAT):
    pipe = Pipeline([("prep", make_preprocessor(features_cat, scale)), ("reg", regressor)])
    # the model learns log(1 + price); predictions are converted back to lakh INR automatically
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


candidates = {
    "Baseline (median price)": make_model(DummyRegressor(strategy="median")),
    "Linear Regression": make_model(LinearRegression()),
    "Ridge": make_model(Ridge(alpha=1.0)),
    "Random Forest": make_model(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE), scale=False),
    "Extra Trees": make_model(ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE), scale=False),
    "Gradient Boosting": make_model(GradientBoostingRegressor(random_state=RANDOM_STATE), scale=False),
}
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
SCORING = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}


def cv_summary(model):
    r = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    return {"R2": r["test_r2"].mean(), "R2_std": r["test_r2"].std(),
            "MAE": -r["test_mae"].mean(), "MAE_std": r["test_mae"].std(), "RMSE": -r["test_rmse"].mean()}


rows = [{"model": n, **cv_summary(m)} for n, m in candidates.items()]
cmp_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
note("Repeated 5-fold CV (3 repeats) on the training set; errors in lakh INR:\n" + cmp_df.round(3).to_string(index=False))
cmp_df.round(4).to_csv(f"{OUT}/model_comparison_cv.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
order = cmp_df.sort_values("MAE", ascending=False)
axs[0].barh(order["model"], order["MAE"], xerr=order["MAE_std"], color=ORANGE, height=0.6)
axs[0].set_title("Average error (MAE, lakh INR) - lower is better", fontsize=11)
fig.suptitle("Default settings, repeated 5-fold CV on the training set (lines = spread across folds)", x=0.01, ha="left", fontsize=10, color=MUTED, y=1.02)
axs[1].barh(order["model"], order["R2"], xerr=order["R2_std"], color=BLUE, height=0.6)
axs[1].set_xlim(-0.15, 1.1)
axs[1].axvline(0, color=MUTED, lw=1)
axs[1].set_title("R-squared - higher is better", fontsize=11)
for ax in axs:
    ax.grid(axis="y", visible=False)
axs[1].set_yticklabels([])
save(fig, "07_model_comparison.png")

# ====================================================================== 5. TUNING + BRAND ABLATION
note("\n=== 5. TUNING ===")
grids = {
    "Random Forest": (make_model(RandomForestRegressor(random_state=RANDOM_STATE), scale=False),
                      {"regressor__reg__n_estimators": [300, 600], "regressor__reg__max_depth": [None, 8, 12],
                       "regressor__reg__min_samples_leaf": [1, 2, 4]}),
    "Extra Trees": (make_model(ExtraTreesRegressor(random_state=RANDOM_STATE), scale=False),
                    {"regressor__reg__n_estimators": [300, 600], "regressor__reg__max_depth": [None, 8, 12],
                     "regressor__reg__min_samples_leaf": [1, 2, 4]}),
    "Gradient Boosting": (make_model(GradientBoostingRegressor(random_state=RANDOM_STATE), scale=False),
                          {"regressor__reg__n_estimators": [200, 400], "regressor__reg__learning_rate": [0.03, 0.05, 0.1],
                           "regressor__reg__max_depth": [2, 3, 4], "regressor__reg__subsample": [0.8, 1.0]}),
    "Ridge": (make_model(Ridge()), {"regressor__reg__alpha": [0.01, 0.1, 1, 3, 10]}),
}
tuned = {}
for name, (model, grid) in grids.items():
    gs = GridSearchCV(model, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs
    note(f"{name}: best CV MAE {-gs.best_score_:.3f} lakh, params { {k.split('__')[-1]: v for k, v in gs.best_params_.items()} }")
best_name = max(tuned, key=lambda n: tuned[n].best_score_)
best_model = tuned[best_name].best_estimator_
note(f"Selected model (lowest CV MAE): {best_name}")

# ablation: does brand information help?
abl = {}
for label, cat_cols in [("with brand", CAT), ("without brand", [c for c in CAT if c != "brand"])]:
    m = clone(best_model).set_params(regressor__prep=make_preprocessor(cat_cols, scale=(best_name == "Ridge")))
    abl[label] = cv_summary(m)
note("\nBrand ablation (CV on training set, same tuned model):\n" + pd.DataFrame(abl).T.round(3).to_string())

# ====================================================================== 6. FINAL TEST EVALUATION
note("\n=== 6. FINAL EVALUATION ON HELD-OUT TEST SET ===")
pred = best_model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
dummy = candidates["Baseline (median price)"].fit(X_train, y_train).predict(X_test)
note(f"{best_name}: R2 = {r2:.3f}, MAE = {mae:.3f} lakh, RMSE = {rmse:.3f} lakh")
note(f"Baseline (always predict the median price): R2 = {r2_score(y_test, dummy):.3f}, "
     f"MAE = {mean_absolute_error(y_test, dummy):.3f} lakh")
rng = np.random.default_rng(RANDOM_STATE)
yt, yp = y_test.values, pred
boot = []
for _ in range(2000):
    i = rng.integers(0, len(yt), len(yt))
    boot.append((r2_score(yt[i], yp[i]), mean_absolute_error(yt[i], yp[i])))
boot = np.array(boot)
ci_r2, ci_mae = np.percentile(boot[:, 0], [2.5, 97.5]), np.percentile(boot[:, 1], [2.5, 97.5])
note(f"Bootstrap 95% CI (2000 resamples of the {len(yt)} test rows): R2 [{ci_r2[0]:.3f}, {ci_r2[1]:.3f}], "
     f"MAE [{ci_mae[0]:.3f}, {ci_mae[1]:.3f}] lakh")

res = X_test.copy()
res["actual"], res["predicted"] = yt, yp
res["error"] = res.predicted - res.actual
res["abs_pct_error"] = (res.error.abs() / res.actual * 100)
bands = pd.cut(res.actual, [0, 1, 4, 10, 100], labels=["< 1 lakh", "1-4 lakh", "4-10 lakh", "> 10 lakh"])
by_band = res.groupby(bands).agg(n=("actual", "size"), MAE_lakh=("error", lambda s: s.abs().mean()),
                                 median_abs_pct_error=("abs_pct_error", "median")).round(2)
by_type = res.groupby("vehicle_type").agg(n=("actual", "size"), MAE_lakh=("error", lambda s: s.abs().mean()),
                                          median_abs_pct_error=("abs_pct_error", "median")).round(2)
note("\nError by price band:\n" + by_band.to_string())
note("\nError by vehicle type:\n" + by_type.to_string())
by_band.to_csv(f"{OUT}/test_error_by_price_band.csv")
res.round(3).sort_values("abs_pct_error", ascending=False).to_csv(f"{OUT}/test_predictions.csv")
note("\nLargest test errors:\n" + res.assign(vehicle=df.loc[res.index, "Car_Name"])
     .sort_values("error", key=abs, ascending=False).head(5)[["vehicle", "actual", "predicted", "error"]].round(2).to_string())

# 08 predicted vs actual + residuals
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for vt, c in [("Two-wheeler", ORANGE), ("Car", BLUE)]:
    m_ = res.vehicle_type == vt
    axs[0].scatter(res.actual[m_], res.predicted[m_], s=36, color=c, alpha=0.85, edgecolor="white", linewidth=0.8, label=vt)
axs[0].plot([0.1, 40], [0.1, 40], color=MUTED, ls=":", lw=1)
axs[0].set_xscale("log"); axs[0].set_yscale("log")
for a_ in (axs[0].xaxis, axs[0].yaxis):
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
axs[0].set_xlabel("Actual selling price (lakh INR, log scale)"); axs[0].set_ylabel("Predicted (lakh INR, log scale)")
axs[0].set_title(f"Predicted vs actual on unseen test data (R2 = {r2:.2f})", fontsize=11)
axs[0].legend(loc="upper left")
axs[1].scatter(res.predicted, res.error, s=36, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.8)
axs[1].axhline(0, color=MUTED, lw=1)
axs[1].set_xlabel("Predicted selling price (lakh INR)"); axs[1].set_ylabel("Error = predicted - actual (lakh INR)")
axs[1].set_title("Errors are small and grow with price", fontsize=11)
save(fig, "08_predicted_vs_actual.png")

# 09 permutation importance (on raw input columns)
pi = permutation_importance(best_model, X_test, y_test, scoring="neg_mean_absolute_error",
                            n_repeats=50, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES, "mae_increase": pi.importances_mean, "std": pi.importances_std}) \
    .sort_values("mae_increase")
note("\nPermutation importance (increase in test MAE, lakh INR, when the column is shuffled):\n"
     + imp.sort_values("mae_increase", ascending=False).round(3).to_string(index=False))
imp.round(4).to_csv(f"{OUT}/permutation_importance.csv", index=False)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.barh(imp.feature, imp.mae_increase, xerr=imp["std"], color=BLUE, height=0.6)
ax.set_xlabel("Increase in test error (MAE, lakh INR) when the feature is shuffled")
ax.set_title("What drives the predictions: the new price dominates")
ax.grid(axis="y", visible=False)
save(fig, "09_feature_importance.png")

# ====================================================================== 7. SAVE MODEL + SUMMARY
# refit on ALL data for the saved model (more data helps a small dataset); reported metrics above stay train/test-honest
final = clone(best_model).fit(X, y)
joblib.dump({"model": final, "features": FEATURES, "ref_year": REF_YEAR}, f"{OUT}/car_price_model.joblib")
note("\nSaved outputs/car_price_model.joblib (refit on all 299 rows; use predict.py to try it).")

with open(f"{OUT}/analysis_log.txt", "w") as f:
    f.write("\n".join(log))
json.dump({
    "selected_model": best_name,
    "best_params": {k.split("__")[-1]: v for k, v in tuned[best_name].best_params_.items()},
    "cv_mae_lakh": round(-tuned[best_name].best_score_, 3),
    "test_r2": round(r2, 3), "test_mae_lakh": round(mae, 3), "test_rmse_lakh": round(rmse, 3),
    "test_r2_95ci": [round(v, 3) for v in ci_r2], "test_mae_95ci": [round(v, 3) for v in ci_mae],
    "baseline_test_mae_lakh": round(mean_absolute_error(y_test, dummy), 3),
    "rows_used": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
}, open(f"{OUT}/metrics.json", "w"), indent=2)
print("\nDone. Outputs in ./outputs")
