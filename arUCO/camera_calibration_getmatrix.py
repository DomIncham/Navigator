import cv2
import numpy as np
import glob

# Chessboard size (adjust based on your pattern)
CHESSBOARD_SIZE = (10, 15)  # (columns, rows)
SQUARE_SIZE = 0.025  # Real size in meters (e.g., 25mm = 0.025m)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale to real-world size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load all images
images = glob.glob("calibration_images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f"✔ Chessboard detected in {fname}")
    else:
        print(f"❌ Chessboard NOT found in {fname}!")

cv2.destroyAllWindows()

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save parameters
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("Calibration complete!")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)