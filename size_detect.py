import cv2
import math
import numpy as np
import time
import mediapipe as mp

# CONFIGURATION
FRONT_HOLD_TIME = 10
SIDE_HOLD_TIME = 10
DISTANCE_TO_CAMERA_CM = 133.0
CORRECTION_FACTOR = 1.27
DEPTH_ADJUST = 1.15  # Compensates for underestimation in side view

DETECT_INTERVAL = 30
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

BLUR_KERNEL_SIZE = (5, 5) 


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load calibration data
with np.load("calibration_data.npz") as X:
    camera_matrix, dist_coeffs = [X[i] for i in ("camera_matrix", "dist_coeffs")]
focal_length = camera_matrix[0, 0]


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def distance_px(p1, p2):
    """Euclidean distance between 2 points in pixel coordinates."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def px_to_cm(px):
    """Convert pixels → cm using calibration."""
    return (px * DISTANCE_TO_CAMERA_CM * CORRECTION_FACTOR) / focal_length

def predict_size(chest_cm, torso_cm, arm_cm, chest_depth_cm):
    """Predict T-shirt size from combined front + side measurements."""
    chest_circum = math.pi * (chest_cm + chest_depth_cm) / 2

    if chest_circum < 82:
        size = "XS"
    elif chest_circum < 90:
        size = "S"
    elif chest_circum < 98:
        size = "M"
    elif chest_circum < 106:
        size = "L"
    elif chest_circum < 114:
        size = "XL"
    else:
        size = "XXL"

    # Adjust for long torso or big arms
    if torso_cm > 55 and size in ["XS", "S"]:
        size = "M"
    elif arm_cm > 60 and size in ["S", "M"]:
        size = "L"

    return size, chest_circum


# ----------------------------
# CAPTURE LOGIC
# ----------------------------

def capture_measurements(phase_name, hold_time, direction_text, measure_depth=False):
    """
    Capture stable pose measurements using a hybrid DL (MediaPipe) + CV (Lucas-Kanade) approach.
    - MediaPipe DETECTS landmarks periodically.
    - Lucas-Kanade TRACKS landmarks between detections.
    """
    cap = cv2.VideoCapture(0)
    start_time = None
    measurements = {}

    # --- State variables for LK Tracking ---
    frame_count = 0
    p0 = np.empty((0, 1, 2), dtype=np.float32) 
    p0_labels = [] 
    
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read from camera.")
        cap.release()
        return {}
    old_frame = cv2.flip(old_frame, 1)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # --- ✅ ADDED BLUR ---
    # We blur the 'old_gray' frame before tracking
    old_gray = cv2.GaussianBlur(old_gray, BLUR_KERNEL_SIZE, 0)
    # --------------------

    track_mask = np.zeros_like(old_frame)
    # ----------------------------------------

    landmarks_to_find = {
        "L_SH": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "R_SH": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "L_WR": mp_pose.PoseLandmark.LEFT_WRIST,
        "R_WR": mp_pose.PoseLandmark.RIGHT_WRIST,
        "L_HIP": mp_pose.PoseLandmark.LEFT_HIP,
        "R_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
        "NOSE": mp_pose.PoseLandmark.NOSE
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- ✅ ADDED BLUR ---
        # We also blur the 'frame_gray' (current) frame before tracking
        frame_gray = cv2.GaussianBlur(frame_gray, BLUR_KERNEL_SIZE, 0)
        # --------------------

        display_frame = frame.copy() 
        frame_count += 1

        landmarks_available = False
        current_landmarks_px = {}
        
        # --------------------------------
        # --- 1. HYBRID LOGIC: DETECT or TRACK ---
        # --------------------------------

        if frame_count % DETECT_INTERVAL == 0 or p0.shape[0] == 0:
            # --- PHASE: DETECT (Heavy) ---
            cv2.putText(display_frame, "DETECTING...", (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Note: MediaPipe runs on the *original* RGB frame, not the blurred one.
            # This is good, as we want the detector to have the cleanest image.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            pose_results = pose.process(rgb)
            track_mask = np.zeros_like(old_frame) 
            
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(display_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                p0_list = []
                p0_labels_new = []

                for name, id_val in landmarks_to_find.items():
                    lmk = lm[id_val.value]
                    if lmk.visibility > 0.5:
                        px, py = int(lmk.x * w), int(lmk.y * h)
                        current_landmarks_px[name] = (px, py)
                        p0_list.append([px, py])
                        p0_labels_new.append(name)
                
                if p0_list:
                    p0 = np.array(p0_list, dtype=np.float32).reshape(-1, 1, 2)
                    p0_labels = p0_labels_new
                    landmarks_available = True
                else:
                    p0 = np.empty((0, 1, 2), dtype=np.float32)
                    p0_labels = []

        elif p0.shape[0] > 0:
            # --- PHASE: TRACK (Light) ---
            cv2.putText(display_frame, "TRACKING (LK)", (w - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # LK now runs on the *blurred* gray frames (old_gray and frame_gray)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None and st is not None and np.sum(st) > 0:
                good_new_points = p1[st == 1]
                good_old_points = p0[st == 1]
                good_new_labels = [label for i, label in enumerate(p0_labels) if st[i] == 1] 
                
                for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    current_landmarks_px[good_new_labels[i]] = (a, b)
                    track_mask = cv2.line(track_mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    display_frame = cv2.circle(display_frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                
                p0 = good_new_points.reshape(-1, 1, 2)
                p0_labels = good_new_labels
                landmarks_available = True
            else:
                p0 = np.empty((0, 1, 2), dtype=np.float32)
                p0_labels = []
        
        # --------------------------------
        # --- 2. MEASUREMENT LOGIC ---
        # --------------------------------
        
        cv2.putText(display_frame, f"{phase_name.upper()} VIEW", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, direction_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if landmarks_available:
            cl = current_landmarks_px 
            chest_cm = torso_cm = arm_cm = chest_depth_cm = 0.0

            try:
                if "L_SH" in cl and "R_SH" in cl:
                    chest_px = distance_px(cl["L_SH"], cl["R_SH"])
                    chest_cm = px_to_cm(chest_px)
                    cv2.putText(display_frame, f"Chest: {chest_cm:.1f}cm", (30, h - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if "L_SH" in cl and "L_HIP" in cl:
                    torso_px = distance_px(cl["L_SH"], cl["L_HIP"])
                    torso_cm = px_to_cm(torso_px)
                    cv2.putText(display_frame, f"Torso: {torso_cm:.1f}cm", (30, h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if "L_SH" in cl and "L_WR" in cl and "R_SH" in cl and "R_WR" in cl:
                    arm_px = (distance_px(cl["L_SH"], cl["L_WR"]) + distance_px(cl["R_SH"], cl["R_WR"])) / 2
                    arm_cm = px_to_cm(arm_px)
                    cv2.putText(display_frame, f"Arm: {arm_cm:.1f}cm", (30, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if measure_depth:
                    if "R_SH" in cl and "R_HIP" in cl and "NOSE" in cl:
                        front_x, back_x, belly_x = cl["NOSE"][0], cl["R_SH"][0], cl["R_HIP"][0]
                    elif "L_SH" in cl and "L_HIP" in cl and "NOSE" in cl:
                        front_x, back_x, belly_x = cl["NOSE"][0], cl["L_SH"][0], cl["L_HIP"][0]
                    else:
                        raise KeyError("Missing landmarks for depth measurement")
                    
                    chest_depth_px = abs(front_x - back_x)
                    belly_depth_px = abs(front_x - belly_x) * 0.3
                    chest_depth_px += belly_depth_px
                    chest_depth_cm = px_to_cm(chest_depth_px * DEPTH_ADJUST)

                    cv2.putText(display_frame, f"Depth: {chest_depth_cm:.1f}cm", (300, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= hold_time:
                    measurements = {
                        "chest_cm": chest_cm,
                        "torso_cm": torso_cm,
                        "arm_cm": arm_cm,
                        "chest_depth_cm": chest_depth_cm
                    }
                    break 
            
            except (KeyError, ZeroDivisionError):
                start_time = None
                cv2.putText(display_frame, "Missing points, hold steady", (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        else:
            start_time = None
            p0 = np.empty((0, 1, 2), dtype=np.float32)
            p0_labels = []

        # --------------------------------
        # --- 3. DISPLAY AND UPDATE ---
        # --------------------------------
        
        img = cv2.add(display_frame, track_mask)
        cv2.imshow(f"{phase_name} Capture", img)
        
        # We update old_gray with the *current* blurred frame
        # to be used as the 'previous' frame in the next loop.
        old_gray = frame_gray.copy()
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            measurements = {}
            break

    cap.release()
    cv2.destroyAllWindows()
    return measurements


# ----------------------------
# MAIN PROGRAM FLOW
# ----------------------------

print("\n🧍 STEP 1: FACE THE CAMERA (Front View)")
print(f"Hold steady for {FRONT_HOLD_TIME} seconds when ready...\n")
front_data = capture_measurements("Front", FRONT_HOLD_TIME, "Stand facing camera — hold still...")
if not front_data:
    print("❌ Front view capture failed.")
    exit()

print(f"\n↩️ STEP 2: TURN SIDEWAYS (Side View)")
print(f"Hold steady for {SIDE_HOLD_TIME} seconds when ready...\n")
side_data = capture_measurements("Side", SIDE_HOLD_TIME, "Turn 90° to your side — hold still...", measure_depth=True)
if not side_data:
    print("❌ Side view capture failed.")
    exit()

# Combine data
chest_depth_cm = side_data["chest_depth_cm"]
size, chest_circum = predict_size(front_data["chest_cm"], front_data["torso_cm"], front_data["arm_cm"], chest_depth_cm)

print("\n============================")
print("📏 FINAL MEASUREMENTS (cm)")
print(f"Front Chest Width  : {front_data['chest_cm']:.1f}")
print(f"Front Torso Length : {front_data['torso_cm']:.1f}")
print(f"Avg Arm Length     : {front_data['arm_cm']:.1f}")
print(f"Side Chest Depth   : {chest_depth_cm:.1f}")
print(f"Estimated Chest Circumference: {chest_circum:.1f}")
print("----------------------------")
print(f"👕 Predicted T-shirt Size: {size}")
print("============================\n")