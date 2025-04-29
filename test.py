import torch
import torchvision
from torchvision import datasets
import matlab
import matplotlib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

exampleDataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
