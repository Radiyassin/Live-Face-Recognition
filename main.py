import threading
import cv2
from deepface import DeepFace

# Initialize camera without CAP_DSHOW
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

# Load reference image and check if valid
reference_img = cv2.imread("test1.jpg")
if reference_img is None:
    print("Error: Reference image not found.")
    cap.release()
    exit()

face_match = False

def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
        face_match = result['verified']
    except Exception as e:
        print(f"Face verification error: {e}")
        face_match = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process face check every 30 frames
    if counter % 30 == 0:
        try:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        except Exception as e:
            print(f"Threading error: {e}")

    counter += 1

    # Display match status
    status_text = "MATCH!" if face_match else "NO MATCH!"
    color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, status_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    
    cv2.imshow('video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()