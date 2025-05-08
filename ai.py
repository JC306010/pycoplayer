import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import mss
import time
import random
from pynput.keyboard import Controller as KeyboardController, Key
from collections import deque

# === CONFIGURATION ===
FPS = 60
FRAME_TIME = 1 / FPS
SCREEN_REGION = {"top": 100, "left": 100, "width": 640, "height": 480}
MOVEMENT_KEYS = ['w', 'a', 's', 'd']
ACTION_KEYS = [' ', 'j']

# Define multi-key movement combinations
ACTIONS = [
    [],                 # No movement
    ['w'],
    ['a'],
    ['s'],
    ['d'],
    ['w', 'a'],
    ['w', 'd'],
    ['s', 'a'],
    ['s', 'd'],
    ['w', ' '],
    ['d', 'j'],
    ['w', 'd', ' '],
    ['a', ' '],
    ['w', 'j'],
]
NUM_ACTIONS = len(ACTIONS)

# === KEYBOARD SETUP ===
keyboard = KeyboardController()
held_keys = set()

# === NEURAL NETWORK (DQN) ===
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# === SCREEN CAPTURE & PREPROCESSING ===
transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((84, 84)),
    T.ToTensor()
])

def grab_screen(region):
    with mss.mss() as sct:
        screen = np.array(sct.grab(region))[:, :, :3]
        return screen

# === KEY MANAGEMENT ===
def press_key(key):
    try:
        keyboard.press(key)
        held_keys.add(key)
    except:
        pass

def release_key(key):
    try:
        keyboard.release(key)
        if key in held_keys:
            held_keys.remove(key)
    except:
        pass

def perform_action(keys_to_press):
    global held_keys

    # Separate movement keys and action keys
    new_movement_keys = [k for k in keys_to_press if k in MOVEMENT_KEYS]
    new_action_keys = [k for k in keys_to_press if k in ACTION_KEYS]

    # Release movement keys not in the new set
    for key in held_keys.copy():
        if key in MOVEMENT_KEYS and key not in new_movement_keys:
            release_key(key)

    # Press new movement keys
    for key in new_movement_keys:
        if key not in held_keys:
            press_key(key)

    # Tap action keys briefly
    for key in new_action_keys:
        press_key(key)
        time.sleep(0.05)
        release_key(key)

# === AGENT DECISION MAKING ===
def select_action(state, epsilon=0.05):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    with torch.no_grad():
        q_values = model(state)
        return torch.argmax(q_values, dim=1).item()

# === MODEL SETUP ===
model = DQN()
model.eval()

# === MAIN LOOP ===
print("Starting AI control loop...")
try:
    while True:
        start_time = time.time()

        # Capture and process screen
        screen = grab_screen(SCREEN_REGION)
        image_tensor = transform(screen).unsqueeze(0)

        # Decide action
        action_index = select_action(image_tensor)
        keys_to_press = ACTIONS[action_index]

        # Execute action
        perform_action(keys_to_press)

        # Maintain 60 FPS
        elapsed = time.time() - start_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

except KeyboardInterrupt:
    print("Stopping...")
    for key in held_keys.copy():
        release_key(key)
