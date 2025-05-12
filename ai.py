import screen
import cv2

screens = screen.AIScreenshot()
videoCapture = cv2.VideoCapture(0)

while True:
    screens.take_screenshot()
    videoCapture.set(cv2.CAP_PROP_FPS, 30)
    cv2.imshow("screen", screens.array)
    
    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        break
    
videoCapture.release()
cv2.destroyAllWindows()