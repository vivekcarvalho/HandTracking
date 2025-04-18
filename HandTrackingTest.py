import mediapipe as mp
import cv2
import HandTrackingModule as htm
import time


cap = cv2.VideoCapture(1)
pTime = 0
cTime = 0
handTracker = htm.HandTrackingModule()
while True:
    success, img = cap.read()
    img = handTracker.findHands(img, draw=True)

    # Get the coordinates of the nth landmark (0 = wrist, 4 = thumb tip, 8 = index finger tip, etc.)
    lmList = handTracker.findPosition(img, draw=True)
    if len(lmList) != 0:
        # Print the coordinates of the wrist
        print('Wrist:', lmList[0])
        # Print the coordinates of the thumb tip
        print('Thumb Tip:', lmList[4])
        # Print the coordinates of the index finger tip
        print('Index Tip:', lmList[8])
        # Print the coordinates of the middle finger tip
        print('Middle Tip:', lmList[12])
        # Print the coordinates of the ring finger tip
        print('Ring Tip:', lmList[16])
        # Print the coordinates of the pinky finger tip
        print('Pinky Tip:', lmList[20])
    

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps: '+ str(int(fps)), (10, 50), cv2.FONT_HERSHEY_TRIPLEX,
                1, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
