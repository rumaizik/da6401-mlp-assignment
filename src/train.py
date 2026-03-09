"""
Main Training Script
Entry point for training neural networks
"""

import argparse
import numpy as np
import json
import wandb

from sklearn.metrics import f1_score

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


# --------------------------------------------------
# CLI Arguments
# --------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train Neural Network")

    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-o", "--optimizer", type=str, default="sgd")

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument("-nhl", "--num_layers", type=int, default=2)

    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 64])

    parser.add_argument("-a", "--activation", type=str, default="relu")

    parser.add_argument("-wi", "--weight_init", type=str, default="xavier")

    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401-mlp")

    parser.add_argument("--model_save_path", type=str, default="src/best_model.npy")

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_arguments()

    # Hidden layer configuration
    args.hidden_layers = args.hidden_size

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    args.input_size = X_train.shape[1]
    args.output_size = y_train.shape[1]

    # Validation split
    val_split = int(0.9 * X_train.shape[0])

    X_val = X_train[val_split:]
    y_val = y_train[val_split:]

    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    print("Building model...")

    model = NeuralNetwork(args)

    print("Training model...")

    model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Evaluating on test set...")

    y_pred = model.forward(X_test)

    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    f1 = f1_score(y_true, y_pred_labels, average="macro")

    print("Test F1 Score:", f1)

    # Save best model
    best_weights = model.get_weights()
    np.save(args.model_save_path, best_weights, allow_pickle=True)

    print("Best model saved at:", args.model_save_path)

    # Save config
    with open("src/best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Best config saved")


# --------------------------------------------------

if __name__ == "__main__":
    main()