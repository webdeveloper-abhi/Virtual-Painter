import cv2
import numpy as np
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

cap.set(3, 1920)
cap.set(4, 1109)

currentTime = 0
previousTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

lmList = []

header = cv2.imread("header.png")

drawcolor = (210, 52, 135)

xp = 0
yp = 0

drawing = np.zeros((720, 1280, 3), np.uint8) 

while True:

    ref, frame = cap.read()

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    header = cv2.resize(header, (w, 132))

    frame[0:132, 0:w] = header

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(frame, "FPS: " + str(int(fps)), (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    fingertips = [4, 8, 12, 16, 20]

    

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = []

            if lmList[fingertips[0]][1] < lmList[fingertips[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmList[fingertips[id]][2] < lmList[fingertips[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers[1] and fingers[2]:
                if y1 <= 132:
                    if 204 <= x1 <= 380:
                        drawcolor = (7, 176, 237)
                    elif 502 <= x1 <= 671:
                        drawcolor = (201, 23, 232)
                    elif 732 <= x1 <= 877:
                        drawcolor = (219, 57, 71)
                    elif 900 <= x1 <= 1036:
                        drawcolor = (0, 0, 0)

                cv2.rectangle(frame, (x2, y2 - 25), (x2 - 25, y2 + 25), drawcolor, cv2.FILLED)

                xp,yp=0,0

            if fingers[1] and not fingers[2]:
                if y1 <= 132:
                    if 204 <= x1 <= 380:
                        drawcolor = (7, 176, 237)
                    elif 502 <= x1 <= 671:
                        drawcolor = (201, 23, 232)
                    elif 732 <= x1 <= 877:
                        drawcolor = (219, 57, 71)
                    elif 900 <= x1 <= 1036:
                        drawcolor = (0, 0, 0)

                cv2.circle(frame, (x1, y1), 10, drawcolor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp = x1
                    yp = y1

                if drawcolor == (0, 0, 0):
                    cv2.line(drawing, (xp, yp), (x1, y1), drawcolor, 150)
                else:
                    cv2.line(drawing, (xp, yp), (x1, y1), drawcolor, 10)

                xp = x1
                yp = y1

    

    frame = cv2.addWeighted(frame, 1, drawing, 0.7, 0)  

    cv2.imshow("Virtual Painter", frame)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

    lmList = []

cap.release()
cv2.destroyAllWindows()
