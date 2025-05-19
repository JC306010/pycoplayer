import torch
import torch.optim.optimizer
import torchvision.models

class Optimizer:
    def __init__(self, model: torch.nn.Module, optimizer_type="adam"):
        self.model = model
        self.loss_function = torch.nn.SmoothL1Loss()
        if optimizer_type == "sgd": 
            self.optimizer = torch.optim.SGD() 
        else: 
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
            
    def estimate(self):
        self.model.

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(NeuralNetwork, self).__init__()
        # self.resnet = resnet
        self.input_dims = input_dims
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.current_accelerator() else "cpu"
        channels = input_dims[0]
        self.output = torch.nn.functional.log_softmax(self.forward(channels, output_dims))
        
    def forward(self, channels, output_dims):
        return torch.nn.Sequential(torch.nn.Conv2d(channels, 32, 8, 4),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(32, 64, 4, 2),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(64, 64, 3, 1),
                                   torch.nn.ReLU(),
                                   torch.nn.Flatten(),
                                   torch.nn.Linear(5184, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(512, output_dims))