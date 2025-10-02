# main.py
# Description: This script handles the complete machine learning pipeline:
# 1. Loads and preprocesses the dataset.
# 2. Splits the data into training and testing sets.
# 3. Trains an XGBoost classifier.
# 4. Evaluates the model's performance.
# 5. Generates and saves a SHAP feature importance plot.
# 6. Saves the trained model and the data scaler for future use.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import clean_and_preprocess_data
import os
import shap

def train_evaluate_and_save_model():
    """
    Main function to run the diabetes prediction model training, evaluation, and saving pipeline.
    """
    # 1. Load and Preprocess Data
    print("🚀 Starting the model training pipeline...")
    try:
        diabetes_dataset = pd.read_csv('diabetes.csv')
    except FileNotFoundError:
        print("❌ Error: 'diabetes.csv' not found. Please place the dataset in the root directory.")
        return

    processed_data, scaler = clean_and_preprocess_data(diabetes_dataset)
    print("✅ Data preprocessing complete.")
    print("-" * 50)

    # 2. Split Dataset
    print("🔪 Splitting the dataset into training and testing sets...")
    features = processed_data.drop('Outcome', axis=1)
    target = processed_data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print("✅ Dataset splitting complete.")
    print("-" * 50)

    # 3. Train XGBoost Model
    print("🧠 Training the XGBoost model...")
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    print("-" * 50)

    # 4. Evaluate the Model
    print("📊 Evaluating the model's performance...")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png') # Save for potential use in the UI
    plt.show()
    print("✅ Model evaluation complete.")
    print("-" * 50)

    # 5. Generate and Save SHAP Feature Importance Plot
    print("🔍 Generating SHAP feature importance plot...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    # Save the plot to the static directory to be displayed on the web page
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/shap_feature_importance.png')
    plt.show()
    print("✅ SHAP plot saved to 'static/shap_feature_importance.png'.")
    print("-" * 50)

    # 6. Save the Model and Scaler
    print("💾 Saving the trained model and scaler...")
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/diabetes_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Model and scaler saved successfully in the 'models' directory.")
    print("\n🎉 Pipeline finished successfully!")

if __name__ == '__main__':
    train_evaluate_and_save_model()
