import cv2
import numpy as np

# Load camera calibration parameters
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.15, camera_matrix, dist_coeffs)
        
        # Loop through detected markers and draw the axes
        for i, marker_id in enumerate(ids):
            rvec, tvec = rvecs[i], tvecs[i]
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Extract the X, Y, Z values (camera-read values in meters)
            x, y, z = tvec.flatten()

            # Calculate real-world coordinates for x, y, z using the formula:
            # Real-world X, Y, Z = (Camera-read value Y + 9.0667) / 1.6275
            real_x = (x*100 - 0.1286) / 1.6391 / 100 # Real-world X in meters
            real_y = y#*100 + 9.0667) / 1.6275 / 100 # Real-world Y in meters
            real_z = ((z*100 + 12.91) / 1.6267 / 100)-6.6/100 # Real-world Z in meters

            # Display position of each marker (ID, X, Y, Z) in camera space (in meters)
            cv2.putText(frame, f"ID {marker_id[0]}: X={x:.2f}m Y={y:.2f}m Z={z:.2f}m",
                        (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display real-world coordinates (calculated in meters)
            cv2.putText(frame, f"Real-world X={real_x:.2f}m Y={real_y:.2f}m Z={real_z:.2f}m",
                        (10, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame with detected markers and pose
    cv2.imshow("ArUco Pose Estimation", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
