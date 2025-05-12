import screen
import cv2

screens = screen.AIScreenshot()
ocr = screen.CharacterRecognition()
videoCapture = cv2.VideoCapture(0)

while True:
    isFrame, frame = videoCapture.read()
    videoCapture.set(cv2.CAP_PROP_FPS, 30)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()