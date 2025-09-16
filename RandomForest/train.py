from utils import load_data, train_model, save_model

def run_training():
    X, y = load_data("Crop_recommendation.csv")
    model, encoder = train_model(X, y)
    save_model(model, encoder)

if __name__ == "__main__":
    run_training()
