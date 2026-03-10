import argparse
import json
import os
import numpy as np
from sklearn.metrics import f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.npy")
    parser.add_argument("--config_path", type=str, default="best_config.json")
    parser.add_argument("--dataset", type=str, default="mnist")
    return parser.parse_args()


def resolve_path(primary, fallback):
    if os.path.exists(primary):
        return primary
    return fallback


def main():
    args = parse_args()

    model_path = resolve_path(args.model_path, "src/best_model.npy")
    config_path = resolve_path(args.config_path, "src/best_config.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    nn_args = argparse.Namespace(
        dataset=cfg.get("dataset", "mnist"),
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 64),
        loss=cfg.get("loss", "cross_entropy"),
        optimizer=cfg.get("optimizer", "sgd"),
        weight_decay=cfg.get("weight_decay", 0.0),
        learning_rate=cfg.get("learning_rate", 0.001),
        num_layers=cfg.get("num_layers", 2),
        hidden_layers=cfg.get("hidden_layers", cfg.get("hidden_size", [128, 64])),
        activation=cfg.get("activation", "relu"),
        weight_init=cfg.get("weight_init", "xavier"),
        input_size=cfg.get("input_size", 784),
        output_size=cfg.get("output_size", 10),
    )

    model = NeuralNetwork(nn_args)
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)

    dataset_name = "fashion_mnist" if args.dataset == "fashion" else args.dataset
    _, _, X_test, y_test = load_data(dataset_name)

    y_pred = model.forward(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("F1 Score:", f1_score(y_true_labels, y_pred_labels, average="macro"))


if __name__ == "__main__":
    main()
