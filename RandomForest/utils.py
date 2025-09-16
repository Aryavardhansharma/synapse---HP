import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(path="Crop_recommendation.csv"):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

def train_model(X, y):
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("✅ Training Accuracy:", acc)

    return rf, label_encoder

def save_model(model, encoder, model_path="crop_rf_model.pkl", encoder_path="label_encoder.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"💾 Model saved as {model_path}, encoder saved as {encoder_path}")

def load_model(model_path="crop_rf_model.pkl", encoder_path="label_encoder.pkl"):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

def predict_crop(model, encoder, data_dict):
    df = pd.DataFrame([data_dict])
    pred = model.predict(df)
    crop_name = encoder.inverse_transform(pred)[0]
    return crop_name
