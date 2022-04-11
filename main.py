import cv2 
import mediapipe as mp 
import time  

cap = cv2.VideoCapture(0) # подключение к камере
mp_Hands = mp.solutions.hands # хотим распозновать руки (hands)
hands = mp_Hands.Hands(max_num_hands = 2) # характеристики для распознования 
mpDraw = mp.solutions.drawing_utils # иннициализация утилит для рисования

finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)] # координаты "суставов" пальцев, кроме большого
thumb_Coord = (4, 3) # координаты "суставов" большого пальца 

while cap.isOpened():
    success, image = cap.read()
    prevTime = time.time()
    if not success:
        print('Не удалось получить кадр с web-камеры')
        continue 
    image = cv2.flip(image, 1) # зеркально отражаем изображение 
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_image)
    multiLandMarks = result.multi_hand_landmarks 

    if multiLandMarks: 
        for idx, handLms in enumerate(multiLandMarks):
            lbl = result.multi_handedness[idx].classification[0].label
            #print(lbl) 
        upCount = 0 #счетчик пальцев 
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            fingersList = [] # список ключевых точек в пикселях 
            for lm in handLms.landmark:
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  
                fingersList.append((cx, cy))
            side = 'left'
            if fingersList[5][0] > fingersList[17][0]:
               side = 'right'

            for coordinate in finger_Coord:
                if fingersList[coordinate[0]][1] < fingersList[coordinate[1]][1]:
                    upCount +=1

            if side == 'left':        
                if fingersList[thumb_Coord[0]][0] < fingersList[thumb_Coord[1]][0]:
                    upCount += 1
            else:
                if fingersList[thumb_Coord[0]][0] > fingersList[thumb_Coord[1]][0]:
                    upCount += 1 

        cv2.putText(image, str(upCount), (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 200, 50), 12)
        print(upCount) 

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    cv2.putText(image, f'FPS: {int(fps)}', (400, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (240, 100, 0), 3)
    cv2.imshow('image', image)   
    if cv2.waitKey(1) & 0xFF == 27: # ESC
        break 
    
cap.release() 