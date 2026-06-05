import cv2
import mediapipe as mp
import os

# ── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# ── Image paths ───────────────────────────────────────────────────────────────
# Add your image filenames here
IMAGE_FOLDER = "C:/Users/dell/Desktop/aiml/"
IMAGE_FILES  = ["image1.jpg", "image2.jpg", "image3.jpg"]

print("Pose Detection on Images Started!")
print("Press any key to go to next image | Press Q to quit")

with mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    for idx, filename in enumerate(IMAGE_FILES):
        path = IMAGE_FOLDER + filename

        # Check if file exists
        if not os.path.exists(path):
            print(f"File not found: {path} — skipping")
            continue

        # Read image
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read: {path} — skipping")
            continue

        print(f"Processing: {filename}")

        # Resize for display if too large
        h, w = image.shape[:2]
        if w > 1280:
            scale  = 1280 / w
            image  = cv2.resize(image, (1280, int(h * scale)))
            h, w   = image.shape[:2]

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

            lm = results.pose_landmarks.landmark

            # Count visible landmarks
            visible = sum(1 for l in lm if l.visibility > 0.5)

            # Top bar
            cv2.rectangle(image, (0, 0), (w, 45), (0, 0, 0), -1)
            cv2.putText(image,
                f"Image {idx+1}/{len(IMAGE_FILES)}: {filename}  |  Pose Detected  |  Landmarks: {visible}/33",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            print(f"  Pose Detected! Visible landmarks: {visible}/33")

        else:
            # Top bar - no pose
            cv2.rectangle(image, (0, 0), (w, 45), (0, 0, 0), -1)
            cv2.putText(image,
                f"Image {idx+1}/{len(IMAGE_FILES)}: {filename}  |  No Pose Detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            print(f"  No pose detected in {filename}")

        # Bottom instruction
        cv2.putText(image, "Press any key for next image | Q to quit",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Save output image
        output_name = f"output_{filename}"
        output_path = IMAGE_FOLDER + output_name
        cv2.imwrite(output_path, image)
        print(f"  Saved output: {output_name}")

        # Show image
        cv2.imshow("Pose Detection on Images", image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quit!")
            break

cv2.destroyAllWindows()
print("Done! Check your aiml folder for output images.")
