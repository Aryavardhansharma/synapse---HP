import argparse
from train import run_training
from predict import run_prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Recommendation System")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="Choose whether to train a new model or make a prediction")

    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "predict":
        run_prediction()
