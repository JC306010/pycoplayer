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
labels_maps = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Glasses",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

for i in range(1, rows * cols + 1):
    sample_index = torch.randint(len(example_dataset), size=(1, )).item()
    img, label = example_dataset[sample_index]
    plt.subplot(rows, cols, i)
    plt.title(labels_maps[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()