import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as f
from torch.utils.data import RandomSampler, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
    RandomCrop,
    ToTensor,
)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def vectorize():
    pass


def preprocess():
    pass


if __name__ == '__main__':
    pass
