import cv2
import os

# Create a directory to save images
save_dir = "calibration_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)  # Change index if needed
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Calibration - Press 's' to Save", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = os.path.join(save_dir, f"image_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()