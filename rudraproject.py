import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam and Face Mesh model
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables to smooth cursor movement
prev_x, prev_y = 0, 0
smooth_factor = 0.7  # Lower values mean smoother movements

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Mirror the image for easier control
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Process landmarks for eye tracking (IDs 474 to 478)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if id == 1:  # Use only one point to move the cursor
                screen_x = int(screen_w / frame_w * x)
                screen_y = int(screen_h / frame_h * y)

                # Exponential smoothing for smoother cursor movement
                curr_x = smooth_factor * prev_x + (1 - smooth_factor) * screen_x
                curr_y = smooth_factor * prev_y + (1 - smooth_factor) * screen_y

                # Move the cursor only if movement is significant
                if abs(curr_x - prev_x) > 5 or abs(curr_y - prev_y) > 5:
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

        # Check for blink detection using left eye landmarks (145, 159)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Blink detection logic
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)  # Prevent multiple clicks

    # Display the video feed with annotations
    cv2.imshow('Eye Controlled Mouse', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
