import mss
from PIL import Image
from torchvision import transforms
import numpy
import cv2
from enum import Enum

class ImageType(Enum):
    PILImage = 0,
    GrayscaleImage = 1,
    
Region = {
    "top": 300,
    "left": 1150,
    "width": 1500,
    "height": 500
}

class AIScreenshot:
    def __init__(self):
        self.sct = mss.mss()
        self.screenshot = None
        self.PILImage = None
        self.tensor = None
        self.array = None
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(20),
                                        transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))])
        self.image_grayscale = None
        
    def take_screenshot(self, region=False, region_area=Region):
        if (region == True):
            self.screenshot = self.sct.grab(region_area)
            self.process_to_PIL()
            self.process_to_array()
        else:
            self.screenshot = self.sct.grab(self.sct.monitors[1])
            self.process_to_PIL()
            self.process_to_array()

    def process_to_PIL(self):
        self.PILImage = Image.frombytes("RGB", self.screenshot.size, self.screenshot.rgb)
        
    def process_to_array(self):
        self.array = numpy.array(self.screenshot)
    
    def transform_to_tensor(self, image_type: ImageType):

        if (image_type == ImageType.PILImage):
            self.tensor = self.transform(self.PILImage)
        else:
            self.tensor = self.transform(self.PILImage.convert("L"))
        
        return self.tensor
        
    def show_screenshot(self):
        self.PILImage.show()
        
    def transform_to_grayscale(self):
        self.image_grayscale = cv2.cvtColor(self.array, cv2.COLOR_RGB2GRAY)
        
        return self.image_grayscale
        
class ScreenRecorder:
    def __init__():
        pass

class CharacterRecognition:
    def __init__(self):
        pass

    def read_screen(self, image: Image):
        image = numpy.array(image)
        