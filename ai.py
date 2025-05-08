import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import mss
import torchvision.transforms as T
from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
import time
import random
from collections import deque

# Define a small CNN (example model)
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # First convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # Second convolution layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # Third convolution layer
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Fully connected layer
        self.out = nn.Linear(512, num_actions)  # Output layer for Q-values of actions

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply relu activation after first conv layer
        x = F.relu(self.conv2(x))  # Apply relu after second conv layer
        x = F.relu(self.conv3(x))  # Apply relu after third conv layer
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = F.relu(self.fc1(x))    # Apply relu to fully connected layer
        return self.out(x)  # Return Q-values for each action

# Image preprocessing
transform = T.Compose([
    T.ToPILImage(),        # Convert from numpy to PIL image
    T.Grayscale(),         # Convert to grayscale (simplifies input)
    T.Resize((84, 84)),     # Resize to 84x84 (standard for many RL setups)
    T.ToTensor()           # Convert to tensor (from PIL)
])

# Screenshot function
def grab_screen(region=None):
    with mss.mss() as sct:
        screen = np.array(sct.grab(region))
        screen = screen[..., :3]  # Drop alpha channel (transparency)
        return screen

# Setup keyboard and mouse controllers
keyboard = KeyboardController()
mouse = MouseController()

# Define action mappings (for example)
action_keys = ['up', 'down', 'left', 'right']

# Store experiences
experience_replay = deque(maxlen=10000)  # Experience replay buffer
batch_size = 32  # Batch size for training
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration factor
epsilon_min = 0.01
epsilon_decay = 0.995

# Training setup
model = DQN(num_actions=4)  # 4 possible actions (e.g., up, down, left, right)
target_model = DQN(num_actions=4)  # Target model for stability in Q-learning
target_model.load_state_dict(model.state_dict())  # Initialize target model with same weights
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def perform_action(action, hold_time=1.0):
    if action == 0:  # 'up'
        keyboard.press('w')
        time.sleep(hold_time)
        keyboard.release('w')
    elif action == 1:  # 'down'
        keyboard.press('s')
        time.sleep(hold_time)
        keyboard.release('s')
    elif action == 2:  # 'left'
        keyboard.press('a')
        time.sleep(hold_time)
        keyboard.release('a')
    elif action == 3:  # 'right'
        keyboard.press('d')
        time.sleep(hold_time)
        keyboard.release('d')
    elif action == 4: # jump
        keyboard.press(' ')
        time.sleep(hold_time)
        keyboard.release(' ')
    elif action == 5: # action
        keyboard.press('j')
        time.sleep(hold_time)
        keyboard.release('j')

def epsilon_greedy_action(state):
    """Choose action based on epsilon-greedy policy."""
    if np.random.rand() <= epsilon:
        # Explore: choose random action
        return random.randrange(4)
    else:
        # Exploit: choose best action based on model
        with torch.no_grad():
            q_values = model(state)
            return torch.argmax(q_values, dim=1).item()

def train():
    if len(experience_replay) < batch_size:
        return  # Not enough experiences to sample a batch

    # Sample a batch of experiences
    batch = random.sample(experience_replay, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to torch tensors
    states = torch.stack(states)
    next_states = torch.stack(next_states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones)

    # Get Q-values for current states from the model
    q_values = model(states)
    next_q_values = target_model(next_states)
    next_q_values = next_q_values.max(1)[0]

    # Get the Q-value for the taken action
    target_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute the target Q-value: reward + gamma * max(next_state_Q_values)
    target = rewards + (gamma * next_q_values * (1 - dones))

    # Compute loss and update the model
    loss = F.mse_loss(target_q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the epsilon value for exploration
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Example usage of the model
while True:
    img = grab_screen({"top": 100, "left": 100, "width": 640, "height": 480})  # Capture a screenshot
    img = transform(img).unsqueeze(0)  # Preprocess and add batch dimension

    # Choose action using epsilon-greedy policy
    action = epsilon_greedy_action(img)
    
    # Perform the action in the game
    perform_action(action)

    # Get reward and next state (this is game-dependent, so you'll need to integrate it)
    reward = 1  # Placeholder, replace with actual game feedback (reward from the game)
    done = False  # Placeholder, check if the game is over
    next_img = grab_screen({"top": 100, "left": 100, "width": 640, "height": 480})  # Capture next screenshot
    next_img = transform(next_img).unsqueeze(0)

    # Store experience in replay buffer
    experience_replay.append((img, action, reward, next_img, done))

    # Train the model with the stored experiences
    train()

    # Optionally update the target model every few steps
    if random.random() < 0.05:
        target_model.load_state_dict(model.state_dict())
