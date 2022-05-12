import cv2
import os 
import numpy as np 
import time 

DEBUG = True 
FOLDER_PATH = "Header"

BRUSH_THICKNESS = 25 
ERASER_THICKNESS = 100
DRAW = False 

drawColor = np.zeros(1, np.uint8) 

listHeader = os.listdir(FOLDER_PATH)
if DEBUG:
    print(listHeader)



cap = cv2.VideoCapture(0)
width = 1920
height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
prevTime = time.time()
fps = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print('Не удалось получить кадр с web-камеры')
        continue
    image = cv2.flip(image, 1) # зеркально отражаем изображение
    if DEBUG:
        cv2.putText(image, f'FPS: {int(fps)}', (1700, 1000), cv2.FONT_HERSHEY_PLAIN, 3, (240, 100, 0), 3)
    cv2.imshow('window', image)
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break