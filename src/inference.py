"""
Inference Script
DA6401 Assignment compatible (logits output + robust weight/config loading)
"""

import argparse
import json
import os
import numpy as np

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


# =====================================================
# Parse arguments (same as train.py + model path)
# =====================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Neural Network")

    parser.add_argument("--model_path", type=str, default="src/best_model.npy")

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

    return parser.parse_args()


# =====================================================
# Load model
# =====================================================

def _resolve_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_model(model_path, config_path=None):
    print("Loading model...")

    resolved_model = _resolve_existing_path([
        model_path,
        "src/best_model.npy",
        "best_model.npy",
    ])
    if resolved_model is None:
        raise FileNotFoundError("Could not find model file")

    loaded = np.load(resolved_model, allow_pickle=True)
    try:
        weights_obj = loaded.item()
    except Exception:
        weights_obj = loaded

    resolved_config = _resolve_existing_path([
        config_path,
        "src/best_config.json",
        "best_config.json",
        "src/config.json",
        "config.json",
    ])
    if resolved_config is None:
        raise FileNotFoundError("Could not find config file")

    with open(resolved_config, "r") as f:
        config = json.load(f)

    class Args:
        pass

    args = Args()
    args.input_size = config["input_size"]
    args.output_size = config["output_size"]
    args.hidden_layers = config.get("hidden_layers", config.get("hidden_size", [128, 64]))
    args.activation = config.get("activation", "relu")
    args.loss = config.get("loss", "cross_entropy")
    args.optimizer = config.get("optimizer", "sgd")
    args.learning_rate = config.get("learning_rate", 0.001)
    args.weight_decay = config.get("weight_decay", 0.0)
    args.weight_init = config.get("weight_init", "xavier")

    model = NeuralNetwork(args)
    model.set_weights(weights_obj)

    return model


# =====================================================
# Metrics
# =====================================================

def compute_loss(y_true, logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return -np.sum(y_true * np.log(probs)) / y_true.shape[0]


def compute_metrics(y_true, logits):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(logits, axis=1)

    accuracy = np.mean(y_true_labels == y_pred_labels)

    num_classes = np.max(y_true_labels) + 1
    precision_list, recall_list, f1_list = [], [], []

    for c in range(num_classes):
        tp = np.sum((y_pred_labels == c) & (y_true_labels == c))
        fp = np.sum((y_pred_labels == c) & (y_true_labels != c))
        fn = np.sum((y_pred_labels != c) & (y_true_labels == c))

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
    dataset_name = "fashion_mnist" if args.dataset == "fashion" else args.dataset
    _, _, X_test, y_test = load_data(dataset_name)

    loss, accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

    print("\nResults:")
    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
