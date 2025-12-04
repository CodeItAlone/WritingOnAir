import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Config

MAX_NUM_HANDS = 1
DEFAULT_BRUSH_THICKNESS = 5
ERASER_THICKNESS = 40          # thickness for eraser "line"
SENSITIVITY = 1.8              # >1 = amplify hand motion
MAX_LOST_FRAMES = 2            # frames allowed with no hand before killing prev point
SMOOTH_WINDOW = 4              # points for smoothing

MIN_BRUSH = 1
MAX_BRUSH = 50

# Colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_RED   = (0,   0, 255)
COLOR_GREEN = (0, 255,   0)
COLOR_BLUE  = (255, 0,   0)



# Utility: finger state helpers

def get_fingers_up(lm):
    """
    Returns [index_up, middle_up, ring_up, pinky_up]
    based on tip vs pip y-coordinates.
    """
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    fingers_up = []
    for tip_idx, pip_idx in zip(tips, pips):
        fingers_up.append(lm[tip_idx].y < lm[pip_idx].y)
    return fingers_up


def is_eraser_gesture(hand_landmarks):
    """
    Eraser: ALL four fingers (index, middle, ring, pinky) up.
    """
    lm = hand_landmarks.landmark
    fingers_up = get_fingers_up(lm)
    return all(fingers_up)


def is_index_only_up_gesture(hand_landmarks):
    """
    Pen DOWN: ONLY index up, others down.
    """
    lm = hand_landmarks.landmark
    fingers_up = get_fingers_up(lm)

    index_up = fingers_up[0]
    others_up = any(fingers_up[1:])
    return index_up and not others_up


def is_pinch_gesture(hand_landmarks):
    """
    Pinch (thumb + index close) = PEN UP.
    """
    lm = hand_landmarks.landmark

    index_tip = np.array([lm[8].x, lm[8].y])
    thumb_tip = np.array([lm[4].x, lm[4].y])

    distance = np.linalg.norm(index_tip - thumb_tip)
    return distance < 0.05  # tune if needed


def smooth_point(points_window, new_point):
    """
    Add new_point to window, keep last N, return averaged (x, y).
    """
    points_window.append(new_point)
    if len(points_window) > SMOOTH_WINDOW:
        points_window.pop(0)
    xs = [p[0] for p in points_window]
    ys = [p[1] for p in points_window]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))



# Stroke replay (for undo)

def redraw_canvas_from_strokes(canvas, strokes):
    canvas[:] = 0
    for stroke in strokes:
        pts = stroke["points"]
        if len(pts) < 2:
            continue
        mode = stroke["mode"]
        color = stroke["color"]
        thickness = stroke["thickness"]
        if mode == "draw":
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color, thickness)
        elif mode == "erase":
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (0, 0, 0), thickness)



# Main

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

ret, frame = cap.read()
if not ret:
    print("Error: Could not read from webcam.")
    cap.release()
    raise SystemExit

h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

cx, cy = w // 2, h // 2  # center for scaling

prev_x, prev_y = None, None
lost_frames = 0

# Drawing state
current_color = COLOR_WHITE
brush_thickness = DEFAULT_BRUSH_THICKNESS
whiteboard_mode = False

strokes = []           # history of strokes
current_stroke = None  # active stroke
prev_mode = None       # None / "draw" / "erase"
smooth_window = []     # for smoothing

# Recording
recording = False
video_writer = None

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    model_complexity=0,   # faster
)

print("Press 'c' to clear, 'b' to soft clear, 's' to save, 'z' to undo,")
print("'1-4' to change color, '+/-' to change thickness, 'w' whiteboard mode, 'r' record, 'q' quit.")
print("GESTURES:")
print("- ONLY index finger up      = PEN DOWN (draw)")
print("- Pinch (thumb + index)     = PEN UP (no draw)")
print("- Open palm (4 fingers up)  = ERASER")

cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received, exiting.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    index_tip_point = None
    eraser_mode = False
    pen_down = False

    if result.multi_hand_landmarks:
        lost_frames = 0
        hand_landmarks = result.multi_hand_landmarks[0]

        # Comment this out if you need more FPS
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        eraser_mode = is_eraser_gesture(hand_landmarks)
        pinch = is_pinch_gesture(hand_landmarks)
        index_only = is_index_only_up_gesture(hand_landmarks)

        if eraser_mode:
            pen_down = False
            mode = "erase"
        elif pinch:
            pen_down = False
            mode = None
        elif index_only:
            pen_down = True
            mode = "draw"
        else:
            pen_down = False
            mode = None

        lm = hand_landmarks.landmark
        raw_x = int(lm[8].x * w)
        raw_y = int(lm[8].y * h)

        # Scale motion around center
        scaled_x = int(cx + (raw_x - cx) * SENSITIVITY)
        scaled_y = int(cy + (raw_y - cy) * SENSITIVITY)
        scaled_x = max(0, min(w - 1, scaled_x))
        scaled_y = max(0, min(h - 1, scaled_y))

        # Smoothing
        sm_x, sm_y = smooth_point(smooth_window, (scaled_x, scaled_y))
        index_tip_point = (sm_x, sm_y)

        # Show raw and scaled points for debug
        cv2.circle(frame, (raw_x, raw_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (sm_x, sm_y), 4, (255, 0, 0), 1)

    else:
        lost_frames += 1
        mode = None
        if lost_frames > MAX_LOST_FRAMES:
            prev_x, prev_y = None, None
            smooth_window = []

    
    # Handle stroke state (start/end/current)
   
    # Determine current drawing mode: "draw", "erase", or None
    if not result.multi_hand_landmarks:
        mode = None

    if mode != prev_mode:
        # Close previous stroke if existed
        if prev_mode in ("draw", "erase") and current_stroke is not None:
            strokes.append(current_stroke)
            current_stroke = None

        # Start new stroke if drawing/erasing
        if mode in ("draw", "erase") and index_tip_point is not None:
            current_stroke = {
                "mode": mode,
                "color": current_color,
                "thickness": brush_thickness if mode == "draw" else ERASER_THICKNESS,
                "points": []
            }
            smooth_window = []  # reset smoothing for new stroke
        prev_mode = mode

    
    # Drawing / Erasing Logic
    
    if index_tip_point is not None:
        x, y = index_tip_point

        if mode == "erase":
            # Erase with thick black line
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), ERASER_THICKNESS)
            if current_stroke is not None:
                current_stroke["points"].append((x, y))
            prev_x, prev_y = x, y

        elif mode == "draw":
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), current_stroke["color"], current_stroke["thickness"])
            if current_stroke is not None:
                current_stroke["points"].append((x, y))
            prev_x, prev_y = x, y

        else:
            prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

   
    # Overlay + UI / Toolbar
    
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    if mode == "erase":
        mode_text = "MODE: ERASER (Open Palm)"
    elif mode == "draw":
        mode_text = "MODE: DRAW (Index only)"
    else:
        mode_text = "MODE: PEN UP (Pinch / idle)"

    color_name = {
        COLOR_WHITE: "White",
        COLOR_RED: "Red",
        COLOR_GREEN: "Green",
        COLOR_BLUE: "Blue"
    }.get(current_color, "Custom")

    info_lines = [
        mode_text,
        f"Color: {color_name}  | Thickness: {brush_thickness}",
        "Keys: C=Clear, B=Soft Clear, S=Save, Z=Undo, 1-4 Colors, +/- Thickness, W Whiteboard, R Record, Q Quit",
        "Gestures: Index=Draw, Pinch=Pen Up, Open Palm=Eraser"
    ]

    display_frame = canvas if whiteboard_mode else combined

    y0 = 25
    for i, line in enumerate(info_lines):
        cv2.putText(
            display_frame,
            line,
            (10, y0 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Recording
    if recording:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"air_session_{timestamp}.avi"
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
            print(f"[INFO] Recording started: {filename}")
        video_writer.write(display_frame)
    else:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print("[INFO] Recording stopped.")

    cv2.imshow("Air Writing", display_frame)
    cv2.imshow("Canvas", canvas)

    
    # Key handling
   
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        canvas[:] = 0
        strokes.clear()
        current_stroke = None
        print("[INFO] Canvas cleared.")

    if key == ord('b'):
        # Soft clear / fade canvas
        canvas = (canvas * 0.7).astype(np.uint8)
        print("[INFO] Soft clear (fade) applied.")

    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_note_{timestamp}.png"
        cv2.imwrite(filename, canvas)
        print(f"[INFO] Canvas saved as {filename}")

    if key == ord('z'):
        # Finish current stroke before undo
        if current_stroke is not None:
            strokes.append(current_stroke)
            current_stroke = None
            prev_mode = None
        if strokes:
            strokes.pop()
            redraw_canvas_from_strokes(canvas, strokes)
            print("[INFO] Undo last stroke.")

    # Colors
    if key == ord('1'):
        current_color = COLOR_WHITE
    if key == ord('2'):
        current_color = COLOR_RED
    if key == ord('3'):
        current_color = COLOR_GREEN
    if key == ord('4'):
        current_color = COLOR_BLUE

    # Thickness
    if key == ord('+') or key == ord('='):
        brush_thickness = min(MAX_BRUSH, brush_thickness + 1)
    if key == ord('-') or key == ord('_'):
        brush_thickness = max(MIN_BRUSH, brush_thickness - 1)

    if key == ord('w'):
        whiteboard_mode = not whiteboard_mode
        print(f"[INFO] Whiteboard mode: {whiteboard_mode}")

    if key == ord('r'):
        recording = not recording
        # start/stop handled in loop

# Close any remaining active stroke
if current_stroke is not None:
    strokes.append(current_stroke)

hands.close()
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
