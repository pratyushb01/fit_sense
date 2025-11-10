import cv2
import numpy as np
import os
import cvzone
from cvzone.PoseModule import PoseDetector

# --- Setup ---
cap = cv2.VideoCapture(0)
detector = PoseDetector()
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)

# Shirt and overlay parameters
fixedRatio = 262 / 190
shirtRatioHeightWidth = 581 / 440
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonRight = cv2.resize(imgButtonRight, (0,0), fx=0.5, fy=0.5)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

# --- Background Subtractor (Pure CV Segmentation) ---
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

# --- Lucas-Kanade Optical Flow params ---
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
prev_gray = None
prev_points = None
smooth_lm11, smooth_lm12 = None, None
alpha = 0.6  # optical flow blending weight

while True:
    success, frame = cap.read()
    if not success:
        break
    h, w, _ = frame.shape

    # --- STEP 1: CV-BASED SEGMENTATION ---
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    segmented = frame

    # --- STEP 2: Contour detection to keep only largest moving blob (the person) ---
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(segmented, [largest], -1, (0,255,0), 2)

    # --- STEP 3: Pose landmarks ---
    img = detector.findPose(segmented)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        lm11 = np.array(lmList[11][:2], dtype=np.float32).reshape(1,1,2)
        lm12 = np.array(lmList[12][:2], dtype=np.float32).reshape(1,1,2)

        # Track using Lucas-Kanade Optical Flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None and prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
            if status.sum() >= 2:
                p11, p12 = next_points[0], next_points[1]
                # blend detection + LKT for stability
                lm11 = (1-alpha)*lm11 + alpha*p11
                lm12 = (1-alpha)*lm12 + alpha*p12

        # Update smooth positions
        smooth_lm11 = tuple(lm11.ravel().astype(int))
        smooth_lm12 = tuple(lm12.ravel().astype(int))

        prev_points = np.array([[smooth_lm11, smooth_lm12]], dtype=np.float32).reshape(-1,1,2)
        prev_gray = gray.copy()

        # --- STEP 4: Shirt overlay ---
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        widthOfShirt = int((smooth_lm11[0] - smooth_lm12[0]) * fixedRatio)
        print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (abs(widthOfShirt)+1, abs(int(widthOfShirt * shirtRatioHeightWidth))+1))
        currentScale = (smooth_lm11[0] - smooth_lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (smooth_lm12[0] - offset[0], smooth_lm12[1] - offset[1]))
        except:
            pass

        # --- Buttons and gesture switching ---
        img = cvzone.overlayPNG(img, imgButtonRight, (524, 240))
        img = cvzone.overlayPNG(img, imgButtonLeft, (50, 240))

        if lmList[16][0] < 83:
            counterRight += 1
            cv2.ellipse(img, (83, 273), (33,33), 0, 0, counterRight*selectionSpeed, (0,255,0), 10)
            if counterRight*selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts)-1:
                    imageNumber += 1
        elif lmList[15][0] > 557:
            counterLeft += 1
            cv2.ellipse(img, (557, 273), (33,33), 0, 0, counterLeft*selectionSpeed, (0,255,0), 10)
            if counterLeft*selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
        else:
            counterRight = 0
            counterLeft = 0

    cv2.imshow("Virtual Try-On (CV Segmentation + Optical Flow)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()