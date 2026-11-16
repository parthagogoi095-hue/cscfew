# ==============================================================================
# UNLOX AI/ML Minor Project: Solar Power Generation Prediction
# Student Name: Chirantan Paul
# Dataset Name: Solar Power Generation Data (Plant 1)
# Kaggle Link : https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # --------------------------------------------------------------------------
    # 1. Data Loading & Understanding
    # --------------------------------------------------------------------------
    # Ensure both CSV files from Kaggle are in the same working directory
    gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
    weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

    print("Generation Data Shape:", gen_df.shape)
    print("Weather Data Shape:", weather_df.shape)
    print("\nGeneration Columns & Types:\n", gen_df.dtypes)
    print("\nWeather Columns & Types:\n", weather_df.dtypes)

    # --------------------------------------------------------------------------
    # 2. Data Quality Check & Cleaning
    # --------------------------------------------------------------------------
    print("\nMissing values in Generation:\n", gen_df.isnull().sum())
    print("\nMissing values in Weather:\n", weather_df.isnull().sum())
    print("\nDuplicates in Generation:", gen_df.duplicated().sum())
    print("Duplicates in Weather:", weather_df.duplicated().sum())

    # Convert DATE_TIME to uniform datetime format
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # Aggregate power output across all 22 inverters per 15-minute timestamp
    gen_grouped = gen_df.groupby('DATE_TIME')[['DC_POWER', 'AC_POWER', 'DAILY_YIELD']].sum().reset_index()

    # Merge Generation and Weather datasets on DATE_TIME
    df = pd.merge(
        gen_grouped,
        weather_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']],
        on='DATE_TIME',
        how='inner'
    )

    # Feature Engineering: Extract Hour of Day
    df['HOUR'] = df['DATE_TIME'].dt.hour

    print("\nCleaned & Merged Dataset Shape:", df.shape)
    print(df.head())

    # --------------------------------------------------------------------------
    # 3. Exploratory Data Analysis (EDA) & 4 Meaningful Visualizations
    # --------------------------------------------------------------------------
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Plot 1: Irradiation vs AC Power
    sns.scatterplot(data=df, x='IRRADIATION', y='AC_POWER', alpha=0.5, ax=axes[0, 0], color='orange')
    axes[0, 0].set_title('1. Solar Irradiation vs. AC Power Output')
    axes[0, 0].set_xlabel('Irradiation')
    axes[0, 0].set_ylabel('Total AC Power (kW)')

    # Plot 2: Average AC Power by Hour of Day
    hourly_power = df.groupby('HOUR')['AC_POWER'].mean()
    axes[0, 1].plot(hourly_power.index, hourly_power.values, marker='o', color='green', linewidth=2)
    axes[0, 1].set_title('2. Average AC Power Generation by Hour of Day')
    axes[0, 1].set_xlabel('Hour of Day (0-23)')
    axes[0, 1].set_ylabel('Average AC Power (kW)')

    # Plot 3: Ambient Temperature vs Module Temperature
    sns.regplot(
        data=df,
        x='AMBIENT_TEMPERATURE',
        y='MODULE_TEMPERATURE',
        scatter_kws={'alpha': 0.3},
        line_kws={'color': 'red'},
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('3. Ambient Temperature vs. Module Temperature')

    # Plot 4: Correlation Heatmap
    corr = df[['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1, 1])
    axes[1, 1].set_title('4. Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300)
    plt.show()

    # --------------------------------------------------------------------------
    # 4. Approach, Model Selection & Evaluation
    # --------------------------------------------------------------------------
    # DC_POWER is excluded to prevent data leakage
    X = df[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR']]
    y = df['AC_POWER']

    # 80-20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model 1: Linear Regression (Baseline)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Model 2: Random Forest Regressor (Non-linear ensemble)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    def evaluate_model(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n=== {name} ===")
        print(f"R2 Score : {r2:.4f}")
        print(f"MAE      : {mae:.2f} kW")
        print(f"RMSE     : {rmse:.2f} kW")

    evaluate_model("Linear Regression", y_test, lr_preds)
    evaluate_model("Random Forest Regressor", y_test, rf_preds)

    # Plot Actual vs Predicted AC Power for Random Forest
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, rf_preds, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted AC Power (Random Forest)')
    plt.xlabel('Actual AC Power (kW)')
    plt.ylabel('Predicted AC Power (kW)')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
