# preprocessing.py
# Description: This script provides a function to preprocess the diabetes dataset.
# It handles missing values by imputing them with the mean and standardizes the features.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def clean_and_preprocess_data(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocesses the input diabetes dataset by handling missing values and scaling features.

    Args:
        raw_data (pd.DataFrame): The raw diabetes dataset.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed and scaled dataset.
            - StandardScaler: The fitted scaler object used for standardization.
    """
    print("--- Starting Data Preprocessing ---")

    # Make a copy to avoid modifying the original DataFrame
    processed_data = raw_data.copy()

    # Columns where a value of 0 is physiologically improbable and should be treated as missing data
    columns_with_improbable_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    print("\n--- Replacing 0 values with NaN for imputation ---")
    for col in columns_with_improbable_zeros:
        processed_data[col] = processed_data[col].replace(0, np.nan)

    print("--- Imputing NaN values with the mean of each column ---")
    for col in columns_with_improbable_zeros:
        mean_value = processed_data[col].mean()
        processed_data[col] = processed_data[col].fillna(mean_value)

    print("\n--- Data after imputation ---")
    print(processed_data.head())

    # Separate features and target variable
    features = processed_data.drop('Outcome', axis=1)
    target = processed_data['Outcome']

    # Standardize the features
    print("\n--- Standardizing feature data ---")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create a new DataFrame with the scaled features and original column names
    scaled_feature_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Combine scaled features with the target variable
    final_dataset = pd.concat([scaled_feature_df, target.reset_index(drop=True)], axis=1)

    print("\n--- Standardized Dataset Summary ---")
    print(final_dataset.describe().round(2))

    return final_dataset, scaler

if __name__ == '__main__':
    try:
        raw_diabetes_df = pd.read_csv('diabetes.csv')
        processed_df, fitted_scaler = clean_and_preprocess_data(raw_diabetes_df)
        print("\n--- Preprocessing Complete ---")
        print("Processed DataFrame head:")
        print(processed_df.head())
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found. Please ensure the dataset is in the current directory.")
