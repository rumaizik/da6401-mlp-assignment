"""
Inference Script
Fully compatible with model.npy and config.json
DA6401 Assignment Version
"""

import argparse
import numpy as np
import json

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


# =====================================================
# Parse arguments
# =====================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="model.npy",
        help="Path to saved model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion"]
    )

    return parser.parse_args()


# =====================================================
# Load model correctly
# =====================================================

def load_model(model_path):

    print("Loading model...")

    # Load weights
    model_data = np.load(model_path, allow_pickle=True).item()

    weights = model_data["weights"]
    biases = model_data["biases"]

    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Create args object
    class Args:
        pass

    args = Args()

    # Required fields
    args.input_size = config["input_size"]
    args.output_size = config["output_size"]

    args.hidden_layers = config["hidden_layers"]

    args.activation = config["activation"]
    args.loss = config["loss"]
    args.optimizer = config["optimizer"]

    args.learning_rate = config["learning_rate"]
    args.weight_decay = config.get("weight_decay", 0.0)

    # IMPORTANT FIX
    args.weight_init = config.get("weight_init", "xavier")

    # Create model
    model = NeuralNetwork(args)

    # Load weights into layers
    for i, layer in enumerate(model.layers):

        layer.W = weights[i]
        layer.b = biases[i]

    return model


# =====================================================
# Metrics
# =====================================================

def compute_loss(y_true, y_pred):

    eps = 1e-12

    y_pred = np.clip(y_pred, eps, 1 - eps)

    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    return loss


def compute_metrics(y_true, y_pred):

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = np.mean(y_true == y_pred)

    num_classes = np.max(y_true) + 1

    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):

        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)

    return accuracy, precision, recall, f1


# =====================================================
# Evaluate
# =====================================================

def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")

    logits = model.forward(X_test)

    loss = compute_loss(y_test, logits)

    accuracy, precision, recall, f1 = compute_metrics(y_test, logits)

    return loss, accuracy, precision, recall, f1


# =====================================================
# Main
# =====================================================

def main():

    args = parse_arguments()

    model = load_model(args.model_path)

    print("Loading dataset...")

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    loss, accuracy, precision, recall, f1 = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nResults:")
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nEvaluation complete!")


# =====================================================

if __name__ == "__main__":
    main()