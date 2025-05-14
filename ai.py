import screen
import cv2
import torchvision.models
import torch

screens = screen.AIScreenshot()
screens.take_screenshot()
screens.transform_to_grayscale()
resnet_model = torchvision.models.resnet18(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children()))[:-1]
resnet_model.eval()

random_data = torch.randn([5, 2, 3])

with torch.no_grad():
    features = resnet_model(random_data)
    
print(resnet_model.shape)

cv2.destroyAllWindows()