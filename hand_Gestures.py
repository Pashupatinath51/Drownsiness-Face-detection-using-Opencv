""" This project leverages OpenCV, MediaPipe, and Pycaw to control system volume using hand gestures detected via webcam. By measuring the distance between thumb and index fingertips, 
it translates finger movements into volume adjustments, providing a touchless, intuitive interface. Ideal for exploring computer vision and gesture-based control applications """


import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Pycaw setup for system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume, max_volume = volume_range[0], volume_range[1]
print(f"Volume range: min = {min_volume}, max = {max_volume}")

# Initialize the webcam
cap = cv2.VideoCapture(0)

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural interaction
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert landmark coordinates to pixel values
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

            # Calculate distance between thumb and index fingertip
            distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)
            print(f"Distance: {distance}")

            # Normalize distance to a volume level
            normalized_distance = np.clip(distance / 200, 0.0, 1.0)
            print(f"Normalized Distance: {normalized_distance}")

            # Calculate volume level and handle NaN cases
            try:
                volume_level = min_volume + normalized_distance * (max_volume - min_volume)
                if np.isnan(volume_level):  # Handle NaN case
                    volume_level = min_volume
                volume.SetMasterVolumeLevel(volume_level, None)
            except Exception as e:
                print(f"Error setting volume level: {e}")

            # Display volume level on screen
            volume_percentage = int((volume_level - min_volume) / (max_volume - min_volume) * 100)
            cv2.putText(frame, f'Volume: {volume_percentage}%', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the line between thumb and index and show distance
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'Distance: {int(distance)}', (thumb_x + 10, thumb_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Volume Control with Hand Gesture", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
