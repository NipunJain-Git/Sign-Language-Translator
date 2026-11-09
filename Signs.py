import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. SETUP MEDIAPIPE & LABELS
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# All 29 target classes: A-Z + SPACE, DELETE, CLEAR
LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["SPACE", "DELETE", "CLEAR"]
DATA_FILE = "asl_dataset.pkl"

# ==========================================
# 2. FEATURE EXTRACTION & HEURISTIC FALLBACK
# ==========================================
def extract_features(landmarks):
    """
    Converts 21 3D hand landmarks into 63 scale-invariant, position-invariant features.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    # Center around the wrist (landmark 0)
    wrist = coords[0]
    relative_coords = coords - wrist
    # Normalize by maximum distance so hand size/distance from camera doesn't matter
    max_val = np.max(np.abs(relative_coords))
    if max_val > 0:
        relative_coords = relative_coords / max_val
    return relative_coords.flatten()

def geometric_fallback_classifier(landmarks):
    """
    Built-in geometric rules so common signs & commands work immediately 
    even before you train the full 29-sign ML model.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Helper: distance between two landmarks
    def dist(i, j):
        return np.linalg.norm(pts[i][:2] - pts[j][:2])

    # Finger extension checks (tip higher than PIP knuckle)
    idx_up = pts[8][1] < pts[6][1]
    mid_up = pts[12][1] < pts[10][1]
    rng_up = pts[16][1] < pts[14][1]
    pnk_up = pts[20][1] < pts[18][1]
    fingers = [idx_up, mid_up, rng_up, pnk_up]
    
    # Thumb extended outwards check
    thumb_out = dist(4, 17) > dist(3, 17) * 1.15

    # Commands first
    if fingers == [True, False, False, True] and not thumb_out:
        return "CLEAR"  # Horns (Index + Pinky, thumb tucked)
    if all(fingers) and thumb_out and dist(8, 12) > dist(5, 9) * 1.2:
        return "SPACE"  # Wide open spread palm
    if not any(fingers) and thumb_out and pts[4][1] > pts[5][1]:
        return "DELETE" # Thumb pointing sideways

    # Alphabet Rules
    if all(fingers) and not thumb_out:
        return "B"
    if fingers == [False, False, False, True] and not thumb_out:
        return "I"
    if fingers == [False, False, False, True] and thumb_out:
        return "Y"
    if fingers == [True, False, False, False] and thumb_out:
        return "L"
    if fingers == [True, False, False, False] and not thumb_out:
        return "D"
    if fingers == [True, True, True, False] and not thumb_out:
        return "W"
    if fingers == [False, True, True, True] and dist(4, 8) < 0.06:
        return "F"
    if fingers == [True, True, False, False] and not thumb_out:
        if dist(8, 12) < dist(5, 9) * 0.9:
            return "U"
        elif pts[12][0] < pts[8][0]: # Crossed fingers (Right hand mirror)
            return "R"
        else:
            return "V"
    if not any(fingers):
        if dist(4, 8) < 0.06 and dist(4, 12) < 0.06:
            return "O"
        elif pts[4][1] < pts[6][1] and pts[4][0] > pts[6][0]:
            return "A"
        else:
            return "S"
    return "?"

# ==========================================
# 3. UI DRAWING HELPERS
# ==========================================
def create_rounded_mask(width, height, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius, height), 255, -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, height - radius), radius, 255, -1)
    cv2.circle(mask, (width - radius, height - radius), radius, 255, -1)
    return mask

def draw_rounded_rect(img, top_left, bottom_right, color, radius):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    # Load or initialize training data
    dataset_X, dataset_y = [], []
    model = None

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            dataset_X, dataset_y = pickle.load(f)
        if len(set(dataset_y)) >= 2:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(dataset_X, dataset_y)

    cap = cv2.VideoCapture(0)
    
    # UI Dimensions matching screenshot
    CANVAS_W, CANVAS_H = 1000, 720
    CAM_W, CAM_H = 540, 440
    CAM_X = (CANVAS_W - CAM_W) // 2
    CAM_Y = 65
    
    BOX_W, BOX_H = 540, 120
    BOX_X = CAM_X
    BOX_Y = CAM_Y + CAM_H + 20

    cam_mask = create_rounded_mask(CAM_W, CAM_H, radius=16)
    cam_mask_3ch = cv2.cvtColor(cam_mask, cv2.COLOR_GRAY2BGR)

    # State variables for typing sentences
    typed_sentence = ""
    current_sign = None
    sign_hold_start = 0.0
    HOLD_DURATION = 1.2  # Seconds to hold a sign to type it

    # Training mode variables
    training_mode = False
    target_idx = 0
    recording_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_resized = cv2.resize(frame, (CAM_W, CAM_H))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_sign = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                features = extract_features(hand_landmarks.landmark)

                # Record samples if inside Training Mode
                if training_mode and recording_frames > 0:
                    dataset_X.append(features)
                    dataset_y.append(LABELS[target_idx])
                    recording_frames -= 1
                    if recording_frames == 0:
                        # Save and retrain model automatically
                        with open(DATA_FILE, "wb") as f:
                            pickle.dump((dataset_X, dataset_y), f)
                        if len(set(dataset_y)) >= 2:
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(dataset_X, dataset_y)

                # Predict the sign
                if model is not None:
                    probs = model.predict_proba([features])[0]
                    if np.max(probs) > 0.55:
                        detected_sign = model.classes_[np.argmax(probs)]
                    else:
                        detected_sign = geometric_fallback_classifier(hand_landmarks.landmark)
                else:
                    detected_sign = geometric_fallback_classifier(hand_landmarks.landmark)

                # Draw live badge inside top-left of webcam feed
                if detected_sign and detected_sign != "?":
                    cv2.rectangle(frame_resized, (12, 12), (160, 52), (21, 16, 17), -1)
                    cv2.putText(frame_resized, f"Sign: {detected_sign}", (22, 39),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (175, 225, 115), 1, cv2.LINE_AA)

        # Handle Hold-to-Type logic (only when not training)
        if not training_mode:
            if detected_sign and detected_sign != "?":
                if detected_sign == current_sign:
                    elapsed = time.time() - sign_hold_start
                    # Draw progress bar at bottom of webcam
                    progress = min(elapsed / HOLD_DURATION, 1.0)
                    cv2.rectangle(frame_resized, (0, CAM_H - 6), (int(CAM_W * progress), CAM_H), (175, 225, 115), -1)
                    
                    if elapsed >= HOLD_DURATION:
                        if detected_sign == "SPACE":
                            typed_sentence += " "
                        elif detected_sign == "DELETE":
                            typed_sentence = typed_sentence[:-1]
                        elif detected_sign == "CLEAR":
                            typed_sentence = ""
                        else:
                            typed_sentence += detected_sign
                        # Reset timer so it doesn't spam
                        sign_hold_start = time.time() + 0.4
                else:
                    current_sign = detected_sign
                    sign_hold_start = time.time()
            else:
                current_sign = None

        # ==========================================
        # 5. RENDER UI CANVAS
        # ==========================================
        canvas = np.full((CANVAS_H, CANVAS_W, 3), (21, 16, 17), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX

        # Title
        title = "ASL Live Interpreter"
        t_size = cv2.getTextSize(title, font, 0.7, 1)[0]
        cv2.putText(canvas, title, ((CANVAS_W - t_size[0]) // 2, 40), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Place rounded webcam feed
        roi = canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W]
        canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = np.where(cam_mask_3ch == 255, frame_resized, roi)

        # Bottom Subtitle Card
        draw_rounded_rect(canvas, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (34, 26, 27), 16)
        center_x = BOX_X + BOX_W // 2
        center_y = BOX_Y + BOX_H // 2

        if typed_sentence:
            display_str = typed_sentence[-22:] # Keep within box bounds
            s_size = cv2.getTextSize(display_str, font, 1.0, 2)[0]
            cv2.putText(canvas, display_str, (center_x - s_size[0] // 2, center_y + s_size[1] // 2),
                        font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        elif detected_sign and detected_sign != "?":
            # Show live preview in mint green when sentence is empty
            s_size = cv2.getTextSize(detected_sign, font, 1.0, 2)[0]
            cv2.putText(canvas, detected_sign, (center_x - s_size[0] // 2, center_y + s_size[1] // 2),
                        font, 1.0, (175, 225, 115), 2, cv2.LINE_AA)
        else:
            # Three mint-green waiting dots (...)
            for offset in [-18, 0, 18]:
                cv2.circle(canvas, (center_x + offset, center_y), 5, (175, 225, 115), -1, cv2.LINE_AA)

        # Footer Instructions for Training / Usage
        if training_mode:
            target_label = LABELS[target_idx]
            count = dataset_y.count(target_label)
            status = f"RECORDING ({recording_frames})..." if recording_frames > 0 else f"TRAIN MODE | Sign: [{target_label}] (Saved: {count}) | [N/P] Next/Prev | [R] Record | [T] Exit"
            cv2.putText(canvas, status, (140, CANVAS_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 220, 255), 1, cv2.LINE_AA)
        else:
            hint = "Hold sign 1.2s to type | Press [T] to Train All 29 Signs | [C] Clear Text | [Q] Quit"
            cv2.putText(canvas, hint, (185, CANVAS_H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 130), 1, cv2.LINE_AA)

        cv2.imshow("ASL Live Interpreter", canvas)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            typed_sentence = ""
        elif key == ord('t'):
            training_mode = not training_mode
        elif training_mode:
            if key == ord('n'):
                target_idx = (target_idx + 1) % len(LABELS)
            elif key == ord('p'):
                target_idx = (target_idx - 1) % len(LABELS)
            elif key == ord('r'):
                recording_frames = 40  # Captures 40 frames (~1.5 seconds) of your hand

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
