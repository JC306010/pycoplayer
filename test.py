import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import pyautogui
from pynput.keyboard import Controller

keyboard = Controller()

# Basic screen capture
def capture_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if region is None else region
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img = cv2.resize(img, (128, 128))
        img = img[:, :, :3]  # Drop alpha
        return img.transpose(2, 0, 1) / 255.0  # CHW format for PyTorch

# Simple CNN
class GameAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(32 * 30 * 30, 3)  # 3 actions: left, right, jump (example)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

agent = GameAgent()

# Dummy interaction loop
while True:
    frame = capture_screen()
    tensor = torch.tensor([frame], dtype=torch.float32)
    with torch.no_grad():
        action_logits = agent(tensor)
        action = torch.argmax(action_logits, dim=1).item()

    if action == 0:
        keyboard.press('a')
        keyboard.release('a')
    elif action == 1:
        keyboard.press('d')
        keyboard.release('d')
    elif action == 2:
        keyboard.press('space')
        keyboard.release('space')
