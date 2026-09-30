import cv2
import mediapipe as mp
import winsound  # Windows ke liye built-in sound library
from EAR_detector import get_avg_ear
from MAR_detector import calculate_mar

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Thresholds and Counters
EYE_AR_THRESH = 0.23      # EAR limit for closed eyes
EYE_AR_CONSEC_FRAMES = 30 # Kitne frames tak aankh band rehne par alert bajna chahiye
MOUTH_AR_THRESH = 0.5     # MAR limit for yawning

COUNTER = 0  # Frame counter for eye closure

print("Advanced Drowsiness Detection with Alarm Started... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_height, image_width, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    drowsy_detected = False
    yawn_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get EAR and MAR from modules
            avg_ear = get_avg_ear(face_landmarks, image_width, image_height)
            mar = calculate_mar(face_landmarks, image_width, image_height)

            # Display values on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Check for Eye Drowsiness with Consecutive Frames logic
            if avg_ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    drowsy_detected = True
            else:
                COUNTER = 0  # Reset counter if eyes are open

            # Check for Yawning
            if mar > MOUTH_AR_THRESH:
                yawn_detected = True

    # Handle Alerts and Sound
    if drowsy_detected:
        cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Beep sound (Frequency: 2500Hz, Duration: 100ms)
        try:
            winsound.Beep(2500, 3000)
        except:
            pass

    if yawn_detected:
        cv2.putText(frame, "YAWNING DETECTED! FEELING TIRED?", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow('Driver Drowsiness Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()