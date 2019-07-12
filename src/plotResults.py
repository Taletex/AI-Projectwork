### STEP 6, 7, 8: VIDEO CLASSIFICATION ###
# Script usato per plottare i risultati della classificazione video #
### --------------------------------------- ###

import torch
from matplotlib import pyplot as plt
import os
from PIL import Image
import pandas as pd
import numpy as np


def main():
    # Initialize history
    history_loss = {"train": [], "val": [], "test": []}
    history_accuracy = {"train": [], "val": [], "test": []}

    trainResults = pd.read_csv('../datasets/UCF-101-Std/results/train.log', delimiter = "\t", dtype={'epoch': "int64", 'lr': "float64"})
    valResults = pd.read_csv('../datasets/UCF-101-Std/results/val.log', delimiter = "\t", dtype={'epoch': "int64", 'lr': "float64"})

    # Update history
    for index, row in trainResults.iterrows():
        history_loss["train"].append(float(row['loss'][7:13]))
        history_accuracy["train"].append(float(row['acc'][7:13]))

    for index, row in valResults.iterrows():
        history_loss["val"].append(float(row['loss'][7:13]))
        history_accuracy["val"].append(float(row['acc'][7:13]))

    bestAccuracy = max(history_accuracy["val"])
    bestConfigIdx = history_accuracy["val"].index(bestAccuracy)
    print(f"Best configuration at epoch {bestConfigIdx} with validation accuracy {bestAccuracy}\n")

    # Plot loss
    plt.title("Loss")
    for split in ["train", "val", "test"]:
        plt.plot(history_loss[split], label=split)
    plt.legend()
    plt.savefig("../progress/video_classifier/loss.png")
    plt.show()

    # Plot accuracy
    plt.title("Accuracy")
    for split in ["train", "val", "test"]:
        plt.plot(history_accuracy[split], label=split)
    plt.legend()
    plt.savefig("../progress/video_classifier/accuracy.png")
    plt.show()


if __name__ == '__main__':
    main()
