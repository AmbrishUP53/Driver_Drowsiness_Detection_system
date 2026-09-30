import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye_landmarks, face_landmarks, image_width, image_height):
    coords = []
    for idx in eye_landmarks:
        landmark = face_landmarks.landmark[idx]
        coords.append(np.array([int(landmark.x * image_width), int(landmark.y * image_height)]))
    
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h = np.linalg.norm(coords[0] - coords[3])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_avg_ear(face_landmarks, image_width, image_height):
    left_ear = calculate_ear(LEFT_EYE, face_landmarks, image_width, image_height)
    right_ear = calculate_ear(RIGHT_EYE, face_landmarks, image_width, image_height)
    return (left_ear + right_ear) / 2.0