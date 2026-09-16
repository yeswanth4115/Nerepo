import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Display webcam feed
    cv2.imshow("Gaze Tracking - Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()