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
    def __init__(self, state_dims, output_dims):
        super(NeuralNetwork, self).__init__()
        channels, height, width = state_dims
        if (height != 240):
            raise ValueError("wrong height")
        
        if (width != 240):
            raise ValueError("wrong width")
        
        self.linear1 = torch.nn.Linear(state_dims)
        # self.resnet = resnet
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.current_accelerator() else "cpu"
        self.sequential = self.nn.Sequential()
        
    def forward(self, x, textInput):
        x = torch.nn.Conv2d(1, 32, 8, 1)