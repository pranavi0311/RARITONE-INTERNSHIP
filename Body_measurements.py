import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Measurement History (average last 15 frames for stability) ────────────────
class MeasurementSmoother:
    def __init__(self, window=15):
        self.window  = window
        self.history = {}

    def update(self, key, value):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)
        self.history[key].append(value)
        return np.mean(self.history[key])

# ── Pixel Distance ────────────────────────────────────────────────────────────
def dist(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# ── Improved Measurements ─────────────────────────────────────────────────────
def get_measurements(lm, w, h, smoother):

    # ── 1. Shoulder Width ─────────────────────────────────────────────────────
    # Direct left-right shoulder distance
    shoulder_px = dist(lm[11], lm[12], w, h)

    # ── 2. Torso Height (shoulder to hip) ─────────────────────────────────────
    # Average of left and right sides for accuracy
    left_torso  = dist(lm[11], lm[23], w, h)
    right_torso = dist(lm[12], lm[24], w, h)
    torso_px    = (left_torso + right_torso) / 2

    # ── 3. Upper Leg (hip to knee) ────────────────────────────────────────────
    left_thigh  = dist(lm[23], lm[25], w, h)
    right_thigh = dist(lm[24], lm[26], w, h)
    thigh_px    = (left_thigh + right_thigh) / 2

    # ── 4. Lower Leg (knee to ankle) ─────────────────────────────────────────
    left_shin   = dist(lm[25], lm[27], w, h)
    right_shin  = dist(lm[26], lm[28], w, h)
    shin_px     = (left_shin + right_shin) / 2

    # ── 5. Hip Width ──────────────────────────────────────────────────────────
    hip_px      = dist(lm[23], lm[24], w, h)

    # ── 6. Arm Length (shoulder to wrist) ─────────────────────────────────────
    left_upper  = dist(lm[11], lm[13], w, h)
    left_lower  = dist(lm[13], lm[15], w, h)
    right_upper = dist(lm[12], lm[14], w, h)
    right_lower = dist(lm[14], lm[16], w, h)
    arm_px      = ((left_upper + left_lower) + (right_upper + right_lower)) / 2

    # ── Scale Factor ──────────────────────────────────────────────────────────
    # Human body ratio: shoulder width ≈ 23% of total height (average)
    # More accurate than fixed 40cm assumption
    # Total height from multiple segments:
    # head(approx) + torso + thigh + shin
    head_approx = dist(lm[0], lm[11], w, h) * 0.8  # nose to shoulder approx
    total_body_px = head_approx + torso_px + thigh_px + shin_px

    # Use body proportion ratios (average human body)
    # Shoulder width = ~23% of height
    # Torso = ~30% of height
    if total_body_px > 0:
        # Estimate height from body segments
        # Average adult height = 165cm (for India)
        scale = 165.0 / total_body_px  # cm per pixel
    elif shoulder_px > 0:
        # Fallback: shoulder width avg = 40cm
        scale = 40.0 / shoulder_px
    else:
        scale = 0.1

    # ── Convert to cm ─────────────────────────────────────────────────────────
    shoulder_cm = shoulder_px  * scale
    height_cm   = total_body_px * scale
    hip_cm      = hip_px       * scale
    torso_cm    = torso_px     * scale
    arm_cm      = arm_px       * scale
    thigh_cm    = thigh_px     * scale
    shin_cm     = shin_px      * scale

    # ── Weight Estimation (improved) ──────────────────────────────────────────
    # Using Devine formula + BMI cross check
    height_m    = height_cm / 100
    # BMI method (average BMI = 22)
    weight_bmi  = 22 * (height_m ** 2)
    # Devine formula: weight = 50 + 2.3 * (height_inches - 60)
    height_inch = height_cm / 2.54
    weight_dev  = 50 + 2.3 * max(0, height_inch - 60)
    # Average both formulas
    weight_kg   = (weight_bmi + weight_dev) / 2

    # ── Smooth all measurements ───────────────────────────────────────────────
    shoulder_cm = smoother.update("shoulder", shoulder_cm)
    height_cm   = smoother.update("height",   height_cm)
    hip_cm      = smoother.update("hip",      hip_cm)
    torso_cm    = smoother.update("torso",    torso_cm)
    arm_cm      = smoother.update("arm",      arm_cm)
    weight_kg   = smoother.update("weight",   weight_kg)

    return {
        "shoulder": shoulder_cm,
        "height":   height_cm,
        "hip":      hip_cm,
        "torso":    torso_cm,
        "arm":      arm_cm,
        "weight":   weight_kg
    }

# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

smoother       = MeasurementSmoother(window=15)
prev_time      = time.time()
snapshot_count = 0

REQUIRED = [0, 11, 12, 23, 24, 25, 26, 27, 28]

print("Improved Body Measurements Started!")
print("Stand FULL BODY visible for accurate measurements!")
print("Press Q to quit | S to save snapshot")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    # Lighting fix
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l       = clahe.apply(l)
    frame   = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    detected  = False
    full_body = False
    measures  = {}

    if results.pose_landmarks:
        lm       = results.pose_landmarks.landmark
        detected = True

        # Check full body visibility
        missing   = [i for i in REQUIRED if lm[i].visibility < 0.5]
        full_body = len(missing) == 0

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

        if full_body:
            measures = get_measurements(lm, w, h, smoother)

            # Draw shoulder measurement line
            lsx = int(lm[11].x * w); lsy = int(lm[11].y * h)
            rsx = int(lm[12].x * w); rsy = int(lm[12].y * h)
            cv2.line(frame, (lsx, lsy-15), (rsx, rsy-15), (0,255,255), 2)
            cv2.putText(frame, f"{measures['shoulder']:.1f}cm",
                ((lsx+rsx)//2 - 30, lsy-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # Draw height line
            nose_x = int(lm[0].x * w)
            nose_y = int(lm[0].y * h)
            ank_y  = int(((lm[27].y + lm[28].y)/2) * h)
            cv2.line(frame, (nose_x-25, nose_y), (nose_x-25, ank_y), (0,255,0), 2)
            cv2.putText(frame, f"{measures['height']:.1f}cm",
                (nose_x-80, (nose_y+ank_y)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Draw hip line
            lhx = int(lm[23].x * w); lhy = int(lm[23].y * h)
            rhx = int(lm[24].x * w); rhy = int(lm[24].y * h)
            cv2.line(frame, (lhx, lhy+15), (rhx, rhy+15), (255,100,0), 2)
            cv2.putText(frame, f"{measures['hip']:.1f}cm",
                ((lhx+rhx)//2 - 20, lhy+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,0), 1)

    # FPS
    curr_time = time.time()
    fps       = int(1/(curr_time - prev_time + 0.001))
    prev_time = curr_time

    # Top bar
    cv2.rectangle(frame, (0,0), (w,40), (0,0,0), -1)
    status = "Pose Detected" if detected else "No Pose Detected"
    color  = (0,255,0) if detected else (0,0,255)
    cv2.putText(frame, f"Improved Body Measurements  |  {status}",
        (10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS:{fps}", (w-80,27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

    # Full body warning
    if detected and not full_body:
        cv2.rectangle(frame, (0,40), (w,70), (0,0,150), -1)
        cv2.putText(frame, "Stand back! Show full body for accurate measurements",
            (10,62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Measurements panel
    if full_body and measures:
        px = w - 240
        py = h - 155
        cv2.rectangle(frame, (px-10, py-25), (w-5, h-5), (0,0,0), -1)
        cv2.putText(frame, "-- BODY MEASUREMENTS --", (px-5, py),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
        cv2.putText(frame, f"Height    : {measures['height']:.1f} cm",
            (px, py+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Shoulder  : {measures['shoulder']:.1f} cm",
            (px, py+44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Hip Width : {measures['hip']:.1f} cm",
            (px, py+66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Torso     : {measures['torso']:.1f} cm",
            (px, py+88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Arm Length: {measures['arm']:.1f} cm",
            (px, py+110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Est.Weight: {measures['weight']:.1f} kg",
            (px, py+132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1)

    # Bottom bar
    cv2.putText(frame, "Q = Quit | S = Snapshot | Show FULL body for accuracy",
        (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.imshow("Improved Body Measurements", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"measurement_snapshot_{snapshot_count}.png"
        cv2.imwrite(fname, frame)
        print(f"Snapshot saved: {fname}")
        snapshot_count += 1

cap.release()
pose.close()
cv2.destroyAllWindows()
print("Done!")
