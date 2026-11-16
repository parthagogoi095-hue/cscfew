"""
End-to-End ML Project: Solar Power Generation Forecasting & Inverter Diagnostics
Student Name: Chirantan Paul
Course/Batch: AI/ML Boosting Minor Project (Unlox Academy)
Dataset: Solar Power Generation Data (https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    USE_XGB = False

sns.set_theme(style="whitegrid", palette="viridis")


def load_data():
    """Step 1 & 2: Load Plant 1 and Plant 2 generation and weather sensor datasets."""
    p1_gen = pd.read_csv("Plant_1_Generation_Data.csv")
    p1_wea = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
    p2_gen = pd.read_csv("Plant_2_Generation_Data.csv")
    p2_wea = pd.read_csv("Plant_2_Weather_Sensor_Data.csv")

    print(f"Plant 1 Generation: {p1_gen.shape} | Weather: {p1_wea.shape}")
    print(f"Plant 2 Generation: {p2_gen.shape} | Weather: {p2_wea.shape}")
    return p1_gen, p1_wea, p2_gen, p2_wea


def audit_data_quality(p1_gen, p1_wea):
    """Step 3: Perform data understanding, missing value, and duplicate checks."""
    print("\n--- PLANT 1 GENERATION INFO ---")
    p1_gen.info()

    print("\n--- MISSING VALUES CHECK ---")
    print("P1 Gen Nulls:\n", p1_gen.isna().sum())
    print("\nP1 Weather Nulls:\n", p1_wea.isna().sum())

    print("\n--- DUPLICATE ROWS CHECK ---")
    print("P1 Gen Duplicates:", p1_gen.duplicated().sum())
    print("P1 Weather Duplicates:", p1_wea.duplicated().sum())

    print("\n--- NUMERICAL SUMMARY (PLANT 1 GENERATION) ---")
    print(p1_gen.describe())

    print("\n--- NUMERICAL SUMMARY (PLANT 1 WEATHER) ---")
    print(p1_wea.describe())


def prepare_plant_data(gen_df, wea_df, plant_label):
    """Step 4: Standardize timestamps, merge generation & weather data, and engineer domain features."""
    g = gen_df.copy()
    w = wea_df.copy()

    # Standardize DATE_TIME across both files
    g["DATE_TIME"] = pd.to_datetime(g["DATE_TIME"], dayfirst=True, format="mixed")
    w["DATE_TIME"] = pd.to_datetime(w["DATE_TIME"], format="mixed")

    # Drop redundant ID columns from weather before merging
    w_clean = w[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    # Inner merge on timestamp
    merged = pd.merge(g, w_clean, on="DATE_TIME", how="inner")
    merged["PLANT"] = plant_label

    # Temporal & physical domain features
    merged["HOUR"] = merged["DATE_TIME"].dt.hour
    merged["MINUTE"] = merged["DATE_TIME"].dt.minute
    merged["TIME_OF_DAY"] = merged["HOUR"] + merged["MINUTE"] / 60.0
    merged["TEMP_DIFF"] = merged["MODULE_TEMPERATURE"] - merged["AMBIENT_TEMPERATURE"]

    # Encode inverter ID as integer category
    merged["INVERTER_ID"] = merged["SOURCE_KEY"].astype("category").cat.codes

    return merged


def run_eda_visualizations(df_p1):
    """Step 5: Generate 4 meaningful EDA visualizations and save as PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Chart 1: Irradiation vs AC Power (Non-linear saturation & thermal spread)
    sample_p1 = df_p1.sample(n=min(5000, len(df_p1)), random_state=42)
    sc = axes[0, 0].scatter(
        sample_p1["IRRADIATION"],
        sample_p1["AC_POWER"],
        c=sample_p1["MODULE_TEMPERATURE"],
        cmap="inferno",
        alpha=0.5,
        s=12,
    )
    plt.colorbar(sc, ax=axes[0, 0], label="Module Temp (°C)")
    axes[0, 0].set_title("1. Solar Irradiation vs. AC Power Output (Plant 1)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Irradiation (kW/m²)")
    axes[0, 0].set_ylabel("AC Power (kW)")

    # Chart 2: Average Diurnal Generation & Irradiation Curve by Hour
    hourly_profile = df_p1.groupby("HOUR")[["AC_POWER", "IRRADIATION"]].mean()
    ax2_twin = axes[0, 1].twinx()
    axes[0, 1].plot(
        hourly_profile.index,
        hourly_profile["AC_POWER"],
        color="#1b9e77",
        marker="o",
        linewidth=2.5,
        label="Mean AC Power (kW)",
    )
    ax2_twin.plot(
        hourly_profile.index,
        hourly_profile["IRRADIATION"],
        color="#d95f02",
        linestyle="--",
        marker="s",
        linewidth=2,
        label="Mean Irradiation",
    )
    axes[0, 1].set_title("2. Average Diurnal Power & Irradiation Profile", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Hour of Day (0–23)")
    axes[0, 1].set_ylabel("Mean AC Power (kW)", color="#1b9e77")
    ax2_twin.set_ylabel("Mean Irradiation", color="#d95f02")
    axes[0, 1].set_xticks(range(0, 24, 2))

    # Chart 3: Ambient vs Module Temperature Heating Effect
    axes[1, 0].hexbin(
        df_p1["AMBIENT_TEMPERATURE"],
        df_p1["MODULE_TEMPERATURE"],
        gridsize=30,
        cmap="YlOrRd",
        mincnt=1,
    )
    axes[1, 0].plot([20, 40], [20, 40], "k--", linewidth=1.5, label="Ambient = Module Temp")
    axes[1, 0].set_title("3. Solar Panel Heating: Ambient vs. Module Temp", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Ambient Temperature (°C)")
    axes[1, 0].set_ylabel("Module Temperature (°C)")
    axes[1, 0].legend()

    # Chart 4: Total Daily Yield Comparison Across Inverters (Fault / Underperformance Detection)
    inverter_totals = df_p1.groupby("SOURCE_KEY")["AC_POWER"].sum().sort_values() / 4.0
    inverter_totals.plot(kind="barh", ax=axes[1, 1], color="#386cb0")
    axes[1, 1].set_title("4. Cumulative AC Energy Generated per Inverter (Plant 1)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Total Energy Generated over 34 Days (kWh)")
    axes[1, 1].set_ylabel("Inverter ID (SOURCE_KEY)")
    axes[1, 1].tick_params(axis="y", labelsize=8)

    plt.suptitle(
        "Exploratory Data Analysis: Solar Power Plant Performance & Weather Drivers",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig("eda_visualizations.png", dpi=300, bbox_inches="tight")
    plt.show()


def train_and_evaluate(df_p1):
    """Step 6 & 7: Chronological split, model training, evaluation metrics, and anomaly plots."""
    df_model = df_p1.sort_values("DATE_TIME").reset_index(drop=True)

    features = [
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "TEMP_DIFF",
        "TIME_OF_DAY",
        "INVERTER_ID",
    ]
    target = "AC_POWER"

    # 80% earliest timestamps for training, 20% latest days for testing
    split_idx = int(len(df_model) * 0.80)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"\nTraining period: {train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()} ({len(X_train)} rows)")
    print(f"Testing period : {test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()} ({len(X_test)} rows)")

    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest (Bagging)": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    if USE_XGB:
        models["XGBoost (Gradient Boosting)"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        models["HistGradientBoosting (Sklearn)"] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        )

    eval_results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = np.clip(model.predict(X_test), 0, None)
        predictions[name] = preds

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        eval_results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "RMSE (kW)": round(rmse, 2),
            "MAE (kW)": round(mae, 2),
        })

    results_df = pd.DataFrame(eval_results)
    print("\n--- MODEL EVALUATION COMPARISON ---")
    print(results_df.to_string(index=False))

    # Evaluation & Residual Anomaly Detection Plots
    best_model_name = list(models.keys())[-1]
    test_df["PRED_AC_POWER"] = predictions[best_model_name]
    test_df["RESIDUAL"] = test_df["AC_POWER"] - test_df["PRED_AC_POWER"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Actual vs Predicted Time Series (3 Sample Test Days for 1 Inverter)
    sample_inv = test_df["SOURCE_KEY"].unique()[0]
    inv_slice = test_df[test_df["SOURCE_KEY"] == sample_inv].iloc[:288]
    axes[0].plot(inv_slice["DATE_TIME"], inv_slice["AC_POWER"], label="Actual AC Power", color="#1b9e77", linewidth=2)
    axes[0].plot(inv_slice["DATE_TIME"], inv_slice["PRED_AC_POWER"], label="Boosting Forecast", color="#d95f02", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"Actual vs. Predicted Output (3 Test Days)\nInverter: {sample_inv[:10]}...", fontweight="bold")
    axes[0].set_ylabel("AC Power (kW)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    # 2. Actual vs Predicted Scatter
    axes[1].scatter(test_df["AC_POWER"], test_df["PRED_AC_POWER"], alpha=0.25, s=10, color="#386cb0")
    axes[1].plot([0, 1450], [0, 1450], "r--", linewidth=1.5, label="Perfect Prediction (y = x)")
    axes[1].set_title(f"Parity Plot ({best_model_name})", fontweight="bold")
    axes[1].set_xlabel("Actual AC Power (kW)")
    axes[1].set_ylabel("Predicted AC Power (kW)")
    axes[1].legend()

    # 3. Anomaly Detection: Mean Daytime Residual per Inverter
    daytime_test = test_df[test_df["IRRADIATION"] > 0.1]
    inv_residuals = daytime_test.groupby("SOURCE_KEY")["RESIDUAL"].mean().sort_values()
    colors = ["#e34a33" if val < -25 else "#2b8cbe" for val in inv_residuals.values]
    inv_residuals.plot(kind="barh", ax=axes[2], color=colors)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("Inverter Health Check (Mean Daytime Residual)", fontweight="bold")
    axes[2].set_xlabel("Actual minus Predicted AC Power (kW)")
    axes[2].set_ylabel("Inverter ID")
    axes[2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig("model_evaluation_and_anomalies.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    p1_gen, p1_wea, p2_gen, p2_wea = load_data()
    audit_data_quality(p1_gen, p1_wea)

    df_p1 = prepare_plant_data(p1_gen, p1_wea, "Plant_1")
    df_p2 = prepare_plant_data(p2_gen, p2_wea, "Plant_2")
    print(f"\nMerged Plant 1 Shape: {df_p1.shape} | Missing values: {df_p1.isna().sum().sum()}")
    print(f"Merged Plant 2 Shape: {df_p2.shape} | Missing values: {df_p2.isna().sum().sum()}")

    run_eda_visualizations(df_p1)
    train_and_evaluate(df_p1)
