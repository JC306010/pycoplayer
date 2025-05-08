import mss
import easyocr
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy

class AIScreenshot:
    def __init__(self):
        self.sct = mss.mss()
        self.screenshot = None
        self.PILImage = None
        self.tensor = None
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.04465), (0.2470, 0.2435, 0.2616))])

    def take_screenshot(self, region=False, region_area=(150, 100, 640, 480)):
        if (region == True):
            self.screenshot = self.sct.grab(region_area)
        else:
            self.screenshot = self.sct.grab(self.sct.monitors[0])

    def process_to_PIL(self):
        self.PILImage = Image.frombytes("RGB", self.screenshot.size, self.screenshot.bgra, "raw", "BGRX")

    def transform_to_tensor(self):
        self.tensor = self.transform(self.PILImage)
        self.tensor.unsqueeze(0)
        
    def show_screenshot(self):
        self.PILImage.show()

class CharacterRecognition:
    def __init__(self):
        pass

    def read_screen(self, image: Image):
        reader = easyocr.Reader(['en'], gpu=True)
        image = numpy.array(image)
        
        results = reader.readtext(image)
        for (bbox, text, prob) in results:
            print(f"Text: {text}, Probability: {prob}")
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left, top_right = (int(top_left[0]), int(top_left[1])), (int(top_right[0]), int(top_right[1]))
            bottom_left, bottom_right = (int(bottom_left[0]), int(bottom_left[1])), (int(bottom_right[0]), int(bottom_right[1]))