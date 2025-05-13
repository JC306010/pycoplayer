import screen
import cv2

screens = screen.AIScreenshot()

while True:
    screens.take_screenshot()
    cv2.imshow("screen", screens.array)
    
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    
cv2.destroyAllWindows()