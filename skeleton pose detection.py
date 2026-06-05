import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
snapshot_count = 0

print("Skeleton Tracing Started!")
print("Press Q to quit | Press S to save snapshot")

with mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())
            cv2.rectangle(image, (0,0), (w,40), (0,0,0), -1)
            cv2.putText(image, "Skeleton Tracing  |  Pose Detected",
                (10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.rectangle(image, (0,0), (w,40), (0,0,0), -1)
            cv2.putText(image, "Skeleton Tracing  |  No Pose Detected",
                (10,27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(image, "Q = Quit | S = Snapshot",
            (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("Skeleton Tracing", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"skeleton_{snapshot_count}.png", image)
            print(f"Snapshot saved: skeleton_{snapshot_count}.png")
            snapshot_count += 1

cap.release()
cv2.destroyAllWindows()
print("Done!")
