import cv2
import dlib
import numpy as np
import time

# Load face landmark predictor from dlib library
PREDICTOR_PATH = "..\\..\\OpenCV\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# Function to detect eye landmark points
def get_eye_points(shape, eye_indices):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in eye_indices])

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to add outlined text
def putTextWithOutline(frame, text, position, font, scale, text_color, outline_color, thickness, outline_thickness):
    # First draw the text outline
    cv2.putText(frame, text, position, font, scale, outline_color, thickness=outline_thickness)
    # Then draw the text itself
    cv2.putText(frame, text, position, font, scale, text_color, thickness)

# Indices of facial landmarks for the left and right eye in dlib's model
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

# Start video capture from camera
cap = cv2.VideoCapture(0)

eye_blink_count = 0
eye_blink_timestamps = []
eye_closed = False
eye_closure_start = None
last_blink_timestamp = 0.0
blink_cooldown = 0.3  # Cooldown in seconds
minimum_blink_duration = 0.1  # Minimum blink duration
start_time = time.time()

EAR_THRESHOLD = 0.3  # EAR threshold to consider an eye closed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        left_eye_points = get_eye_points(shape, LEFT_EYE_INDICES)
        right_eye_points = get_eye_points(shape, RIGHT_EYE_INDICES)

        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        # Draw points around left eye
        for (x, y) in left_eye_points:
            color = (0, 255, 0) if left_ear > EAR_THRESHOLD else (0, 0, 255)  # Green if open, red if closed
            cv2.circle(frame, (x, y), 1, color, -1)  # Draw small circle for each point

        # Draw points around right eye
        for (x, y) in right_eye_points:
            color = (0, 255, 0) if right_ear > EAR_THRESHOLD else (0, 0, 255)  # Green if open, red if closed
            cv2.circle(frame, (x, y), 1, color, -1)  # Draw small circle for each point

        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if not eye_closed:
                eye_closed = True
                eye_closure_start = time.time()
        else:
            if eye_closed and eye_closure_start is not None:
                eye_closed = False
                blink_duration = time.time() - eye_closure_start
                if blink_duration >= minimum_blink_duration and (time.time() - last_blink_timestamp >= blink_cooldown):
                    eye_blink_count += 1
                    eye_blink_timestamps.append(time.time())
                    last_blink_timestamp = time.time()
                eye_closure_start = None

    current_time = time.time()
    elapsed_time = current_time - start_time
    if len(eye_blink_timestamps) > 1:
        intervals = np.diff(eye_blink_timestamps)
        average_interval = np.mean(intervals)
    else:
        average_interval = 0

    blinks_per_minute = (eye_blink_count / elapsed_time) * 60

    # Display statistics on screen with outlined text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    text_color = (255, 255, 255)  # White
    outline_color = (0, 0, 0)  # Black
    thickness = 2
    outline_thickness = 4  # Adjustable based on requirements

    # Displaying stats with outlined text
    putTextWithOutline(frame, f"Blink count: {eye_blink_count}", (10, 30), font, scale, text_color, outline_color,
                       thickness, outline_thickness)
    putTextWithOutline(frame, f"Blinks per minute: {blinks_per_minute:.2f}", (10, 60), font, scale, text_color,
                       outline_color, thickness, outline_thickness)
    putTextWithOutline(frame, f"Average interval: {average_interval:.2f} s", (10, 90), font, scale, text_color,
                       outline_color, thickness, outline_thickness)
    putTextWithOutline(frame, f"Time [s]: {elapsed_time:.2f}", (10, 120), font, scale, text_color, outline_color,
                       thickness, outline_thickness)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
