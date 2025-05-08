import screen

screens = screen.AIScreenshot()
ocr = screen.CharacterRecognition()

while True:
    screens.take_screenshot()
    screens.process_to_PIL()
    ocr.read_screen(image=screens.PILImage)