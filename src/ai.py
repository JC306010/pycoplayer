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
    def __init__(self, state_dims, action_dims):
        # 96x96 RGB (to be Grayscaled) image
        # The reward is -0.1 every frame and +1000/N for every track tile visited, 
        # where N is the total number of tiles visited in the track. 
        # For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

        super(NeuralNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(state_dims, 16, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.in_features = 32 * 9 * 9
        self.fc1 = torch.nn.Linear(self.in_features, 256)
        self.fc2 = torch.nn.Linear(256, action_dims)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)

        return x
    
class ReplayBuffer():
    def __init__(self, state_dims, action_dims, max_size=int(1e5)):
        self.state = numpy.zeros((max_size, *state_dims), dtype=numpy.float32)
        self.action = numpy.zeros((max_size, *action_dims), dtype=numpy.int64)
        self.reward = numpy.zeros((max_size, 1), dtype=numpy.float32)
        self.next_state = numpy.zeros((max_size, *state_dims), dtype=numpy.float32)
        self.terminated = numpy.zeros((max_size, 1), dtype=numpy.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.terminated[self.ptr] = terminated

        self.ptr = (1 + self.ptr) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        random_index = numpy.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.state[random_index]),
            torch.FloatTensor(self.action[random_index]),
            torch.FloatTensor(self.reward[random_index]),
            torch.FloatTensor(self.next_state[random_index]),
            torch.FloatTensor(self.terminated[random_index])
        )
    
class DeepQLearning():
    def __init__(
            self, 
            state_dims,
            action_dims,
            learning_rate=0.00025, 
            epsilon=1.0, 
            epsilon_min=0.1, 
            gamma=0.99,
            batch_size=32, 
            warmup_steps=5000,
            buffer_size=int(1e5),
            target_update_interval=10000):
        self.action_dims = action_dims
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.warmup_steps = warmup_steps

        self.network = NeuralNetwork(state_dims[0], self.action_dims)
        self.target_network = NeuralNetwork(state_dims[0], self.action_dims)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(state_dims, (1,), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)

        if (training and (numpy.random.random() > self.epsilon) or (self.total_steps < self.warmup_steps)):
            action = numpy.random.randint(0, self.action_dims)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q_value = self.network(x)
            action = torch.argmax(q_value).item()

        return action
    
    def learn(self):
        state, action, reward, next_state, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        next_q_value = self.target_network(next_state).detach()
        td_target = reward + (1 - terminated) * self.gamma * next_q_value.max(dim=1, keepdim=True).values
        loss = torch.nn.functional.mse_loss(self.network(state).gather(1, action.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }

        return result
    
    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon -= self.epsilon_decay

        return result