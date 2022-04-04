import cv2 
import mediapipe as mp 

cap = cv2.VideoCapture(0) #подключение к камере
mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands(max_num_hands = 1) 
mpDraw = mp.solutions.drawing_utils

finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4, 2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print('Не удалось получить кадр с web-камеры')
        continue 
    image = cv2.flip(image, 1)
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(RGB_image)
    multiLandMarks = result.multi_hand_landmarks 

    if multiLandMarks: 
        for idx, handLms in enumerate(multiLandMarks):
            lbl = result.multi_handedness[idx].classification[0].label
            print(lbl)