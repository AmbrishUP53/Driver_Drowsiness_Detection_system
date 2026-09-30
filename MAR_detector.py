import numpy as np

# Mouth landmarks for MAR
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

def calculate_mar(face_landmarks, image_width, image_height):
    top_lip = np.array([int(face_landmarks.landmark[MOUTH_TOP].x * image_width), 
                        int(face_landmarks.landmark[MOUTH_TOP].y * image_height)])
    bottom_lip = np.array([int(face_landmarks.landmark[MOUTH_BOTTOM].x * image_width), 
                           int(face_landmarks.landmark[MOUTH_BOTTOM].y * image_height)])
    
    left_corner = np.array([int(face_landmarks.landmark[MOUTH_LEFT].x * image_width), 
                            int(face_landmarks.landmark[MOUTH_LEFT].y * image_height)])
    right_corner = np.array([int(face_landmarks.landmark[MOUTH_RIGHT].x * image_width), 
                             int(face_landmarks.landmark[MOUTH_RIGHT].y * image_height)])
    
    v_dist = np.linalg.norm(top_lip - bottom_lip)
    h_dist = np.linalg.norm(left_corner - right_corner)
    
    mar = v_dist / h_dist
    return mar