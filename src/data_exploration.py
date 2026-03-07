import wandb
import numpy as np
from utils.data_loader import load_data

wandb.init(project="da6401-mlp")

X_train, y_train, X_test, y_test = load_data("mnist")

labels = np.argmax(y_train, axis=1)

table = wandb.Table(columns=["image", "label"])

count = {i:0 for i in range(10)}

for i in range(len(X_train)):

    label = labels[i]

    if count[label] < 5:
        img = X_train[i].reshape(28,28)
        table.add_data(wandb.Image(img), label)
        count[label] += 1

    if all(v == 5 for v in count.values()):
        break

wandb.log({"MNIST Samples": table})