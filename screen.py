import mss
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional

class AIScreenshot:
    def __init__(self):
        self.sct = mss.mss()
        self.screenshot = None
        self.PILImage = None
        self.tensor = None

    def take_screenshot(self, region=False, region_area=(1920, 1080)):
        if (region == True):
            self.screenshot = self.sct.grab(region_area)
        else:
            self.screenshot = self.sct.grab(self.sct.monitors[0])

    def process_to_PIL(self):
        self.PILImage = Image.frombytes("RGB", self.screenshot.size, self.screenshot.bgra, "raw", "BGRX")

    def transform_to_tensor(self):
        self.tensor = torchvision.transforms.functional.pil_to_tensor(self.PILImage)