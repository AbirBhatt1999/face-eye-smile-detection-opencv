import cv2

# Load models
face_cascade = cv2.CascadeClassifier(
    "face_eye_smile_detect/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    "face_eye_smile_detect/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(
    "face_eye_smile_detect/haarcascade_smile.xml")

# Check cascades
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    print("Error loading cascade files")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # ROI inside loop ✅
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frame, "Eyes Detected", (x, y-30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 2)

        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 20)
        if len(smiles) > 0:
            cv2.putText(frame, "Smile Detected", (x, y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Face Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()