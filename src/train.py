"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
import os
import wandb
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


# --------------------------------------------------
# Argument parser
# --------------------------------------------------
def parse_arguments():

    parser = argparse.ArgumentParser(description="Train Neural Network")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_size", nargs="+", type=int)

    parser.add_argument("--hidden_layers", nargs="+", type=int,
                        default=[128, 64])

    parser.add_argument("--activation", type=str, default="relu")

    parser.add_argument("--loss", type=str, default="cross_entropy")

    parser.add_argument("--weight_init", type=str, default="xavier")

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--model_save_path", type=str,
                        default="src/best_model.npy")

    return parser.parse_args()

# --------------------------------------------------
# Save model
# --------------------------------------------------

def save_model(model, args):

    weights = [(layer.W.copy(), layer.b.copy()) for layer in model.layers]

    arr = np.empty(len(weights), dtype=object)
    for i in range(len(weights)):
        arr[i] = weights[i]

    np.save(args.model_save_path, arr, allow_pickle=True)

    print("Model saved")

    # Save config
    config = vars(args)

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("Model saved as model.npy")
    print("Config saved as config.json")


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main():

    args = parse_arguments()
    if args.hidden_size is not None:
       args.hidden_layers = [int(x) for x in args.hidden_size]
    
    wandb.init(
    project="da6401-mlp",
    config={
        "dataset": args.dataset,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "activation": args.activation
    }
)
    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    # Set input/output sizes
    args.input_size = X_train.shape[1]
    args.output_size = y_train.shape[1]

    # ----------------------------
    # Create validation split (90/10)
    # ----------------------------
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

    print("Evaluating model on test set...")

    test_accuracy = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    save_model(model, args)

# --------------------------------------------------

if __name__ == "__main__":
    main()