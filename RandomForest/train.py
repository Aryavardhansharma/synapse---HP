# train.py

from utils import load_data, train_model, save_model, predict_crop

def run_training():
    # Load dataset
    X, y = load_data()  # CSV must be in the same folder

    # Train model
    model, encoder = train_model(X, y)

    # Save model and encoder
    save_model(model, encoder)

    # Test prediction
    sample = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 25.5,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 210.0
    }
    crop = predict_crop(model, encoder, sample)
    print(f"Predicted Crop: {crop}")

if __name__ == "__main__":
    run_training()
