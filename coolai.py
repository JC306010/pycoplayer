import torch
import torch.optim.optimizer
import torchvision.models

class Optimizer:
    def __init__(self, model, optimizer_type="sgd"):
        self.model = model
        self.loss_function = torch.nn.CrossEntropyLoss()
        if optimizer_type == "sgd": 
            self.optimizer = torch.optim.SGD() 
        else: 
            self.optimizer = torch.optim.Adam()

class NeuralNetwork(torch.nn.Module):
    def __init__(self, batch_size, output_size, resnet):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(batch_size, output_size)
        self.resnet = resnet
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.current_accelerator() else "cpu"
        self.nn = torch.nn
        self.sequential = self.nn.Sequential()
        
    def forward(self, imageInput, textInput):
        x = self.nn.Flatten(x)