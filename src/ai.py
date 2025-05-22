import torch
import torch.optim.optimizer
import torchvision.models
import gymnasium
from collections import defaultdict
import numpy

class Optimizer:
    def __init__(self, model: torch.nn.Module, optimizer_type="adam"):
        self.model = model
        self.loss_function = torch.nn.SmoothL1Loss()
        if optimizer_type == "sgd": 
            self.optimizer = torch.optim.SGD() 
        else: 
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
            
    def estimate(self):
        pass

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
                                   torch.nn.Linear(64 * 11 * 11, 512),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(512, output_dims))
        
class DeepQLearning():
    def __init__(self, 
                 env: gymnasium.Env, 
                 learning_rate: float, 
                 initial_epsilon: float, 
                 final_epsilon: float, 
                 epsilon_decay: float, 
                 discount_rate: float):
        self.env = env
        self.q_values = defaultdict(lambda: numpy.zeros(env.action_space.n))
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.discount_rate = discount_rate
        self.training_error = []
        
    def do_action(self, obs: tuple[int, int, bool]) -> int:
        if (numpy.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return int(numpy.max(self.q_values[obs]))
        
    def step(self, obs: tuple[int, int, bool], action: int, reward: float, next_obs: tuple[int, int, bool]):
        next_obs_key = tuple(next_obs.flatten())
        q_value = numpy.max(self.q_values[next_obs_key])
        obs_key = tuple(obs.flatten())
        current_q_value = numpy.max(self.q_values[obs_key])
        temporal_difference = current_q_value + self.epsilon * (reward + self.discount_rate * q_value) - q_value
        self.q_values[obs_key][action] = self.q_values[obs_key][action] + self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)