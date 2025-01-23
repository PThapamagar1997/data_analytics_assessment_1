# Insurance Cost Prediction using Linear Regression

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the insurance dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("\n--- Dataset Overview ---")
    print(data.head())
    print("\nColumns:", data.columns)
    print("\nSummary Statistics:")
    print(data.describe())

    print("\n--- Checking for Missing Values ---")
    print(data.isnull().sum())

    print("\n--- Visualizing Numeric Features ---")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
    plt.suptitle("Histograms of Numeric Features")
    plt.show()

    print("\n--- Correlation Heatmap ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

# Model Building and Evaluation
def build_and_evaluate_model(data, target):
    # Encode categorical variables (if any)
    data = pd.get_dummies(data, drop_first=True)

    # Splitting data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model initialization and training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # Plotting actual vs predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

if __name__ == "__main__":
    # Load the dataset
    file_path = "/mnt/data/insurance.csv"
    data = load_data(file_path)

    if data is not None:
        perform_eda(data)
        build_and_evaluate_model(data, target="charges")
