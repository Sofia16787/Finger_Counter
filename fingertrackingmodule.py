import cv2
import mediapipe as mp
import time
import math 

class fingerDetector():
    def __init__(self, mode=False, maxHands=2, complexity=0, detectinCon=0.75, trackCon=0.75):
        self.mpHands = mp.solutions.hands # инициализация модуля распознавания рук
        self.hands = self.mpHands.Hands(mode, maxHands, complexity, detectinCon, trackCon) # характеристики для распознавания
        self.mpDraw = mp.solutions.drawing_utils # инициализация утилит для рисования
        self.fingertips = [4, 8, 12, 16, 20] # кончики пальцев
        self.handList = {} #координаты ключевых точек рук 
        self.fingers = {} #поднятые и опущенные пальцы

    def findHands(self, img, draw=False):
        RGB_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        RGB_image.flags.writeable = False
        self.result = self.hands.process(RGB_image)
        RGB_image.flags.writeable = True
        if draw:
            if self.result.multi_hand_landmarks:
                for handLms in self.result.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            
        return img
    
    def findPosition(self, img, handNumber=0, draw=False):
        xList = []
        yList = []
        self.handList[handNumber] = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNumber]
            for lm in myHand.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.handList[handNumber].append([cx, cy])
                
            if draw:
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                #print('(',xmin ,',', ymin, ')', '(',xmax ,',', ymax, ')')
                offset = 20
                cv2.rectangle(img, (xmin-offset, ymin-offset), (xmax+offset, ymax+offset), (0, 255, 0), 2)

    def fingersUp(self, draw=False):
        if self.result.multi_hand_landmarks:
            handCount = len(self.result.multi_hand_landmarks)
            for i in range(handCount):
                self.fingers[i] = []
                side = 'left'
                if self.handList[i][5][0] > self.handList[i][17][0]:
                    side = 'right'
                
                if side == 'left':
                    if self.handList[i][self.fingertips[0]][0] < self.handList[i][self.fingertips[0] - 1][0]:
                        self.fingers[i].append(1)
                    else:
                        self.fingers[i].append(0)
                else:
                    if self.handList[i][self.fingertips[0]][0] > self.handList[i][self.fingertips[0] - 1][0]:
                        self.fingers[i].append(1)
                    else:
                        self.fingers[i].append(0)

                for id in range(1, 5):
                    if self.handList[i][self.fingertips[id]][1] < self.handList[i][self.fingertips[id] - 2][1]:
                        self.fingers[i].append(1)
                    else:
                        self.fingers[i].append(0)
                if draw:
                    print(self.fingers[i])

    def findDistance(self, p1, p2, handNumber=0, draw=False, img=None, r=10, t=3):
        x1, y1 = self.handList[handNumber][p1][0], self.handList[handNumber][p1][1]             
        x2, y2 = self.handList[handNumber][p2][0], self.handList[handNumber][p2][1]         
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)         
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED) 
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED) 
            cv2.line(img, (x1, y1), (x2, y2), (50, 150, 250), t)

        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)   
        return length

def main():
    cap = cv2.VideoCapture(0)
    detector = fingerDetector()
    while cap.isOpened():
        success, image = cap.read()
        prevTime = time.time()
        if not success:
            print('Не удалось получить кадр с web-камеры')
            continue
        image = cv2.flip(image, 1) # зеркально отражаем изображение
        image = detector.findHands(image, True)
        if detector.result.multi_hand_landmarks:
            handCount = len(detector.result.multi_hand_landmarks)
            for i in range(handCount):
                detector.findPosition(image, i, True)
                l = detector.findDistance(4, 8, i, True, image)
                print(l)
        detector.fingersUp()
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        cv2.putText(image, f'FPS: {int(fps)}', (400, 150), cv2.FONT_HERSHEY_PLAIN, 3, (240, 100, 0), 3)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

if __name__ == "__main__":
    main()