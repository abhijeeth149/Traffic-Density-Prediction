import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import pyttsx3
import os

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize Pose and Drawing modules from MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Check if the workout log directory exists, otherwise create it
if not os.path.exists("workout_logs"):
    os.makedirs("workout_logs")

# Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Unable to open camera! Please check your camera access permissions.")
    exit()

counter = 0
stage = None
start_time = time.time()
last_rep_time = None

# Exercise type (push-up or squat)
exercise_type = 'pushup'  # Default

# Cooldown settings
cooldown_threshold = 5  # seconds
cooldown_counter = 0

# Feedback functions
def give_feedback(message):
    engine.say(message)
    engine.runAndWait()
    print("🗣️", message)

def provide_instructions():
    if exercise_type == 'pushup':
        give_feedback("To do a push-up, keep your hands shoulder-width apart and body straight.")
        give_feedback("Lower down until your chest almost touches the ground, then push back up.")
    elif exercise_type == 'squat':
        give_feedback("To do a squat, stand with feet shoulder-width apart.")
        give_feedback("Lower down as if sitting on a chair, then return to standing.")

def give_correction_feedback():
    if exercise_type == 'pushup':
        give_feedback("Incorrect form. Keep your elbows close and avoid arching your back.")
    elif exercise_type == 'squat':
        give_feedback("Incorrect form. Keep your knees behind your toes.")

def log_workout():
    duration = int(time.time() - start_time)
    log_file = f"workout_logs/{datetime.now().strftime('%Y-%m-%d')}_workout_log.txt"
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} | {exercise_type.capitalize()}s: {counter} | Duration: {duration}s\n")
    print(f"✅ Workout logged to {log_file}")

# Start session
provide_instructions()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    h, w, _ = frame.shape

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        def get_coords(idx): return [lm[idx.value].x, lm[idx.value].y]

        # Common key points
        r_shoulder = get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        r_elbow = get_coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
        r_wrist = get_coords(mp_pose.PoseLandmark.RIGHT_WRIST)

        l_shoulder = get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
        l_elbow = get_coords(mp_pose.PoseLandmark.LEFT_ELBOW)
        l_wrist = get_coords(mp_pose.PoseLandmark.LEFT_WRIST)

        # Push-up logic
        if exercise_type == 'pushup':
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            avg_angle = (r_angle + l_angle) / 2

            if avg_angle > 160:
                stage = "up"
            if avg_angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                last_rep_time = time.time()
                give_feedback(f"Great job! Push-up {counter} completed.")

            # Correction
            if avg_angle > 160 and r_angle > 170 and l_angle > 170:
                give_correction_feedback()

        # Squat logic
        elif exercise_type == 'squat':
            r_hip = get_coords(mp_pose.PoseLandmark.RIGHT_HIP)
            r_knee = get_coords(mp_pose.PoseLandmark.RIGHT_KNEE)
            r_ankle = get_coords(mp_pose.PoseLandmark.RIGHT_ANKLE)

            l_hip = get_coords(mp_pose.PoseLandmark.LEFT_HIP)
            l_knee = get_coords(mp_pose.PoseLandmark.LEFT_KNEE)
            l_ankle = get_coords(mp_pose.PoseLandmark.LEFT_ANKLE)

            r_squat_angle = calculate_angle(r_hip, r_knee, r_ankle)
            l_squat_angle = calculate_angle(l_hip, l_knee, l_ankle)
            avg_squat_angle = (r_squat_angle + l_squat_angle) / 2

            if avg_squat_angle > 160:
                stage = "up"
            if avg_squat_angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                last_rep_time = time.time()
                give_feedback(f"Nice work! Squat {counter} completed.")

            # Correction
            if avg_squat_angle > 170:
                give_correction_feedback()

        # Cooldown timer
        if last_rep_time:
            idle_time = time.time() - last_rep_time
            if idle_time > cooldown_threshold:
                if cooldown_counter == 0:
                    give_feedback("Take a short break. You've been inactive for a while.")
                    cooldown_counter += 1
        else:
            cooldown_counter = 0

    # Overlay counter and exercise
    cv2.putText(frame, f'{exercise_type.capitalize()} Reps: {counter}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.imshow("Gym Tracker", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        log_workout()
        break
    elif key == ord('p'):
        exercise_type = 'pushup'
        counter = 0
        stage = None
        provide_instructions()
    elif key == ord('s'):
        exercise_type = 'squat'
        counter = 0
        stage = None
        provide_instructions()

cap.release()
cv2.destroyAllWindows()
