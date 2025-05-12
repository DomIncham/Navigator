import cv2

# เปิดกล้องจาก /dev/video0
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ตั้งค่าความกว้างและความสูงของภาพ
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # แสดงภาพจากกล้อง
    cv2.imshow("USB Camera", frame)
    
    # รอการกดปุ่ม 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
