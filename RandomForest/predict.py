from utils import load_model, predict_crop

def run_prediction():
    model, encoder = load_model()

    # Example new data (replace with real satellite/sensor values)
    new_data = {
    "N": 120,
    "P": 70,
    "K": 200,
    "temperature": 30.0,
    "humidity": 80.0,
    "ph": 6.5,
    "rainfall": 2000.0
}




    
    

    crop = predict_crop(model, encoder, new_data)
    print("🌱 Recommended Crop:", crop)

if __name__ == "__main__":
    run_prediction()
