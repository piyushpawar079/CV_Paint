import cv2
from cvzone.HandTrackingModule import HandDetector as hd
import os
import numpy as np
import time

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)
detector = hd()

folder = 'Images'
myList = os.listdir(folder)
v = []
for i in myList:
    img = cv2.imread(f'{folder}/{i}')
    v.append(img)

header = v[0]

xp, yp = 0, 0
brushThickness = 30
eraserThickness = 100
color1 = (255, 192, 203)
color2 = color3 = (0, 0, 0)
selected = ''
circleFlag = False
done = False
doneL = False
gone = False
lineFlag = False

circleX1, circleY1, circleX2, circleY2, radius = 0, 0, 0, 0, 0

imgC = np.zeros((720, 1280, 3), np.uint8)

while True:

    _, img = cam.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
    lmList = detector.findPosition(img)

    if lmList:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp(hands[0])

        if fingers[1] and fingers[2]:
            if y1 < 130:

                if 10 < x1 < 100:
                    header = v[0]
                    color1 = (255, 192, 203)
                    color3 = (255, 192, 203)
                    selected = 'brush1'

                elif 200 < x1 < 300:
                    header = v[1]
                    header = cv2.resize(header, (1007, 104), interpolation=cv2.INTER_AREA)
                    color1 = (0, 0, 255)
                    color3 = (0, 0, 255)
                    selected = 'brush2'

                elif 450 < x1 < 550:
                    color2 = (0, 0, 0)
                    color1 = (0, 0, 0)
                    color3 = (255, 0, 0)
                    selected = 'circle'
                    circleFlag = True
                    done = False
                    # print('C')

                elif 600 < x1 < 700:
                    color2 = (0, 0, 0)
                    color1 = (0, 0, 0)
                    color3 = (0, 255, 0)
                    selected = 'line'
                    lineFlag = True
                    doneL = False

                elif 800 < x1 < 900:
                    color1 = (0, 0, 0)
                    selected = 'eraser'

            cv2.line(img, (x1, y1), (x2, y2), color3, 3)

        elif fingers[1] and not fingers[2]:

            cv2.circle(img, (x1, y1), 10, (255, 255, 255), -1)
            xp, yp = 0, 0

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), color1, brushThickness)
            cv2.line(imgC, (xp, yp), (x1, y1), color1, brushThickness)
            xp, yp = x1, y1

            if selected == 'eraser':
                cv2.line(img, (xp, yp), (x1, y1), color1, eraserThickness)
                cv2.line(imgC, (xp, yp), (x1, y1), color1, eraserThickness)

            if selected == 'circle':
                if len(hands) == 2 and circleFlag:
                    circleX1, circleY1 = x1, y1
                    # print(hands, hands[0]['type'], hands[1]['type'])
                    # hand = hands[1]
                    id = 8
                    lmList2 = detector.findPosition(img, 1)
                    thumbX, thumbY = lmList2[id][1], lmList2[id][2]

                    radius = int(((thumbX - x1) ** 2 + (thumbY - y1) ** 2) ** 0.5)
                    x3, y3 = lmList[4][1], lmList[4][2]
                    length = int(((x3 - x1) ** 2 + (y3- y1) ** 2) ** 0.5)
                    if length < 160:
                        circleFlag = False
                        done = True
                        color2 = (255, 0, 0)
                        radius = int(((thumbX - x1) ** 2 + (thumbY - y1) ** 2) ** 0.5)
                        cv2.circle(img, (circleX1, circleY1), radius, color2, 5)
                        cv2.circle(imgC, (circleX1, circleY1), radius, color2, 5)

            if not done:
                cv2.circle(img, (circleX1, circleY1), radius, color2, 5)
                cv2.circle(imgC, (circleX1, circleY1), radius, color2, 5)

            if selected == 'line':
                if len(hands) == 2 and lineFlag:
                    gone = True
                    p1, p2 = x1, y1
                    # print(hands, hands[0]['type'], hands[1]['type'])
                    # hand = hands[1]
                    id = 8
                    lmList2 = detector.findPosition(img, 1)
                    p3, p4 = lmList2[id][1], lmList2[id][2]
                    x3, y3 = lmList[4][1], lmList[4][2]
                    length = int(((x3 - x1) ** 2 + (y3- y1) ** 2) ** 0.5)
                    if length < 160:
                        lineFlag = False
                        doneL = True
                        color2 = (255, 0, 0)
                        cv2.line(img, (p1, p2), (p3, p4), color2, 5)
                        cv2.line(imgC, (p1, p2), (p3, p4), color2, 5)

            if not doneL and gone:
                cv2.line(img, (p1, p2), (p3, p4), color2, 5)
                cv2.line(imgC, (p1, p2), (p3, p4), color2, 5)

    imgG = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgG, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgC)

    img[:104, :1007] = header

    cv2.imshow('img', img)
    # cv2.imshow('imgC', imgC)
    cv2.waitKey(1)

