# utils.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import chardet


# Load Data with auto-encoding

def load_data(path=r"C:\Users\harshit choudhary\OneDrive\Desktop\dihhdihh\RandomForest\Crop_recommendation.csv"):
    """
    Load CSV dataset from same folder and split into features (X) and labels (y)
    Automatically detects encoding to prevent Unicode errors.
    """
    # Detect encoding
    with open(path, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']

    df = pd.read_csv(path, encoding=encoding)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


# Train Random Forest Model

def train_model(X, y, max_depth=12, estimator_range=[10, 50, 100, 150, 200, 250, 300]):
    """
    Train Random Forest classifier, plot train vs test accuracy for different n_estimators,
    run cross-validation, and return final model and label encoder.
    Overfitting is reduced by limiting tree depth and other parameters.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    train_accuracies = []
    test_accuracies = []

    print("🔹 Training Random Forest with different n_estimators...")

    for n in estimator_range:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"n_estimators={n}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(estimator_range, train_accuracies, marker='o', label="Training Accuracy")
    plt.plot(estimator_range, test_accuracies, marker='s', label="Test Accuracy")
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Accuracy vs Number of Trees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)

    # Overfitting warning
    if train_accuracies[-1] - test_accuracies[-1] > 0.05:
        print("⚠️ Warning: Model may still be overfitting!")

    # Train final model
    final_rf = RandomForestClassifier(
        n_estimators=estimator_range[-1],
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )
    final_rf.fit(X_train, y_train)

    # 5-Fold Cross-Validation
    cv_scores = cross_val_score(final_rf, X_train, y_train, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

    return final_rf, label_encoder


# Save Model and Encoder

def save_model(model, encoder, model_path="crop_rf_model.pkl", encoder_path="label_encoder.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"💾 Model saved as {model_path}, encoder saved as {encoder_path}")


# Load Model and Encoder

def load_model(model_path="crop_rf_model.pkl", encoder_path="label_encoder.pkl"):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


# Predict Crop

def predict_crop(model, encoder, data_dict):
    """
    Predict crop name from input dictionary of features
    Example:
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 25.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 210.0
    }
    """
    df = pd.DataFrame([data_dict])
    pred = model.predict(df)
    crop_name = encoder.inverse_transform(pred)[0]
    return crop_name
