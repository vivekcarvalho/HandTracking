import cv2
import mediapipe as mp
import time

# Createing a class for hand tracking and initializing the parameters + Webcam
class HandTrackingModule():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands

        # Initialize MediaPipe Hands with the parameters
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        
        # Drawing the hand landmarks directly instead of calculting the coordinates and drawing them
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and find hands
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks
                    self.mpDraw.draw_landmarks(img, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo = 0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):

                # Mapping the landmark coordinates to the image dimensions
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    # Draw a circle at the landmark position (Wrist + Finger tips)
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return lmList
    

        for id, lm in enumerate(self.results.multi_hand_landmarks[handNo].landmark):
            # Mapping the landmark coordinates to the image dimensions
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Draw a circle at the landmark position (Wrist + Finger tips)
            if draw:
                cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=1,
#                       min_detection_confidence=0.7,
#                       min_tracking_confidence=0.5)

# while True:
#     success, img = cap.read()

#     # Converts the image from BGR to RGB
#     imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Process the image and find hands
#     results = hands.process(imageRGB)
#     # print(results.multi_hand_landmarks)

#     # Draw hand landmarks
#     mpdraw = mp.solutions.drawing_utils

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 # print(id, lm)

#                 # Mapping the landmark coordinates to the image dimensions
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # print(id, cx, cy)

#                 # Draw a circle at the landmark position (Wrist + Finger tips)
#                 if id in [0, 4, 8, 12, 16, 20]:
#                     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
#             mpdraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#             # getting the frame rate
#             cTime = time.time()
#             fps = 1 / (cTime - pTime)
#             pTime = cTime
#             cv2.putText(img, 'fps: '+ str(int(fps)), (170, 170), cv2.FONT_HERSHEY_TRIPLEX,
#                         5, (255, 0, 255), 3, cv2.LINE_4)
            
#             # Get the coordinates of the first landmark (wrist)
#             h, w, c = img.shape
#             cx, cy = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

def main():
    # Main function to encapsulate the script
    cap = cv2.VideoCapture(1)
    pTime = 0
    cTime = 0
    handTracker = HandTrackingModule()
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

if __name__ == "__main__":
    main()