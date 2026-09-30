import cv2
import mediapipe as mp
from EAR_detector import get_avg_ear
from MAR_detector import calculate_mar

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

print("Drowsiness Detection System Started... Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_height, image_width, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get EAR and MAR from respective module files
            avg_ear = get_avg_ear(face_landmarks, image_width, image_height)
            mar = calculate_mar(face_landmarks, image_width, image_height)

            # Display values on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Drowsiness Alerts
            if avg_ear < 0.23:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if mar > 0.5:
                cv2.putText(frame, "YAWNING ALERT!", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow('Driver Drowsiness Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()