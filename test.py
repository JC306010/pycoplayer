import torch
import torchvision
from torchvision import datasets
import matlab
import matplotlib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

example_dataset: torch.Tensor = datasets.FashionMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor())
figure = plt.figure(figsize=(8, 8))
rows, cols = 3, 3
print(torch.randint(len(example_dataset), size=(1, 0)))
for i in range(1, rows * cols + 1):
    sample_index = torch.randint(len(example_dataset), size=(1, )).item()
    img, label = example_dataset[sample_index]
    print(example_dataset[sample_index])