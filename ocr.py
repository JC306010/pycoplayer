import easyocr
import cv2
import matplotlib.pyplot as plt

# Load the OCR model
reader = easyocr.Reader(['en'])

# Read the image
image_path = '/Users/Shakopee Robotics/Pictures/Capture.png'
image = cv2.imread(image_path)

# Perform OCR
results = reader.readtext(image)

# Print and visualize results
for (bbox, text, prob) in results:
    print(f"Text: {text}, Probability: {prob}")
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left, top_right = (int(top_left[0]), int(top_left[1])), (int(top_right[0]), int(top_right[1]))
    bottom_left, bottom_right = (int(bottom_left[0]), int(bottom_left[1])), (int(bottom_right[0]), int(bottom_right[1]))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
