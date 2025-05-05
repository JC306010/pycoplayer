import mss
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Initialize the screen capture
monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}  # Adjust for your screen resolution
sct = mss.mss()

# PyTorch image transforms (you can adjust depending on your model requirements)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet pre-trained model normalization
])

def capture_and_process():
    while True:
        # Capture the screen
        screenshot = sct.grab(monitor)

        # Convert to PIL image
        image = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

        # Apply transformations to convert the image into a PyTorch tensor
        tensor = transform(image)

        # Add a batch dimension (batch size of 1)
        tensor = tensor.unsqueeze(0)

        # At this point, you can feed this tensor into your model
        # For demonstration, let's just print its shape
        print(tensor.shape)

        # Add a small delay (to prevent the CPU from maxing out)
        # You can use time.sleep(0.01) if you need a delay between captures

if __name__ == "__main__":
    capture_and_process()
