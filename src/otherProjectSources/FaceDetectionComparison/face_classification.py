import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import os
import argparse
import numpy as np
import shutil
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from glob import glob
from random import shuffle
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define class
class CNN(nn.Module):

    # Constructor
    def __init__(self):
        # Call parent constructor
        super().__init__();
        # Create convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Create fully-connected layers
        self.fc_layers = nn.Sequential(
            # FC layer
            nn.Linear(4096, 1024),
            nn.ReLU(),
            # Classification layer
            nn.Linear(1024, 2)
        )

    # Forward
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Class map
class_names = {0: 'Black', 1: 'White'}

# Params
resize = T.Resize(32)
center_crop = T.CenterCrop(28)
to_tensor = T.ToTensor()
normalize = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))

# Compose transforms
test_transform = T.Compose([resize, center_crop, to_tensor, normalize])

# Select device
print(f"CUDA is available? {torch.cuda.is_available()}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load("./models/face_detection_model")
model.eval()

def face_classifier(imgPath):
    # Load image
    img = Image.open(imgPath).convert("RGB")
    # Transform image
    img = test_transform(img)
    # View as a batch
    img.unsqueeze_(0)
    # Copy to device
    img = img.to(dev)
    with torch.no_grad():
        # Model forward
        pred = model(img)
    # Get prediction
    _,pred = pred.max(1)
    return pred.item()
