import screen
import cv2

screens = screen.AIScreenshot()
ocr = screen.CharacterRecognition()
videoCapture = cv2.VideoCapture(0)

while True:
    isFrame, frame = videoCapture.read()
    cv2.imshow("frame", frame)
    videoCapture.set(cv2.CAP_PROP_FPS, 30)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()