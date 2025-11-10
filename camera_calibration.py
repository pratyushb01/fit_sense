import cv2
import numpy as np

# === Adjustable settings ===
CHESSBOARD_SIZE = (9, 6)   # inner corners
SQUARE_SIZE = 2.4          # cm per square (your chessboard square size)
NUM_FRAMES_TO_CAPTURE = 15 # number of valid patterns before calibration
# ============================

# Prepare real-world points for the chessboard corners
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n🎯 Move the chessboard around the frame until 15 detections are collected.")
print("   Keep it visible at different angles and distances.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret_corners:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_corners)

        if len(objpoints) < NUM_FRAMES_TO_CAPTURE:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"✅ Captured pattern {len(objpoints)} / {NUM_FRAMES_TO_CAPTURE}")
        else:
            print("✅ Enough samples collected. Calibrating...")
            break

    cv2.putText(frame, f"Captured: {len(objpoints)} / {NUM_FRAMES_TO_CAPTURE}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Real-Time Camera Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Calibration cancelled by user.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# === Perform calibration ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n✅ Calibration Complete!")
print("Camera Matrix (K):\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save calibration
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\n💾 Saved as calibration_data.npz\n")
