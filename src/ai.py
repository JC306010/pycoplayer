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
    def __init__(self, input_dims, hidden_dims, output_dims, dropout):
        # 96x96 RGB (to be Grayscaled) image
        # The reward is -0.1 every frame and +1000/N for every track tile visited, 
        # where N is the total number of tiles visited in the track. 
        # For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

        super(NeuralNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_dims, hidden_dims)
        self.layer2 = torch.nn.Linear(hidden_dims, output_dims)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = torch.nn.ReLU()
        x = self.layer2(x)
        
        return x
    
    def calculate_stepwise_return(self, rewards, discount_factor):
        returns = []
        reward = 0
        
        for r in rewards:
            reward = r + reward * discount_factor
            returns.append(reward)
            
        returns = torch.tensor(returns)
        normalized_return = (returns - returns.mean()) / returns.std()
        
        return normalized_return
    
        
class DeepQLearning():
    def __init__(self, 
                 env: gymnasium.Env, 
                 learning_rate: float, 
                 initial_epsilon: float, 
                 final_epsilon: float, 
                 epsilon_decay: float, 
                 discount_rate: float):
        self.env = env
        # self.q_values = defaultdict(lambda: numpy.zeros(env.action_space.n))
        self.actions = 5
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
            # return int(numpy.max(self.q_values[tuple(obs.flatten())]))
            pass
        
    def step(self, obs: tuple[int, int, bool], action: int, reward: float, next_obs: tuple[int, int, bool]):
        # q_value = numpy.max(self.q_values[tuple(next_obs.flatten())])
        # current_q_value = numpy.max(self.q_values[tuple(obs.flatten())])
        # temporal_difference = current_q_value + self.epsilon * (reward + self.discount_rate * q_value) - q_value
        # self.q_values[tuple(obs.flatten())][action] = self.q_values[tuple(obs.flatten())][action] + self.learning_rate * temporal_difference
        # self.training_error.append(temporal_difference)
        pass
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)