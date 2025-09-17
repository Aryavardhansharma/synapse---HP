from utils import load_model, predict_crop

def run_prediction():
    model, encoder = load_model()

    
    new_data ={
    "N": 63,
    "P": 43,
    "K": 17,
    "temperature": 19.23,
    "humidity": 65.47,
    "ph": 6.80,
    "rainfall": 71.31
  }

    crop = predict_crop(model, encoder, new_data)
    print("🌱 Recommended Crop:", crop)

if __name__ == "__main__":
    run_prediction()
