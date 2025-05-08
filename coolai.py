import torch
import torch.optim.optimizer

class Optimizer:
    def __init__(self, model, optimizer_type="sgd"):
        self.loss_function = torch.nn.CrossEntropyLoss()
        if optimizer_type == "sgd": 
            self.optimizer = torch.optim.SGD() 
        else: 
            self.optimizer = torch.optim.Adam()

class NeuralNetwork:
    def __init__(self):
        self.layer1 = torch.nn.Linear(28 * 28)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.current_accelerator() else "cpu"
        