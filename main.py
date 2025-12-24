import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import deque

from shape_autocorrect import autocorrect_stroke

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==================== CONFIG ====================
class Config:
    MAX_NUM_HANDS = 1
    DEFAULT_BRUSH_THICKNESS = 3
    ERASER_THICKNESS = 40
    MAX_LOST_FRAMES = 2
    SMOOTH_WINDOW = 4
    MIN_BRUSH = 1
    MAX_BRUSH = 50
    MODE_STABLE_FRAMES = 3
    
    # Colors (BGR)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    
    # Performance
    DETECTION_CONFIDENCE = 0.5
    TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 0
    
    # Gesture thresholds
    PINCH_THRESHOLD = 0.05
    FINGER_TIP_INDICES = [8, 12, 16, 20]
    FINGER_PIP_INDICES = [6, 10, 14, 18]


# ==================== GESTURE DETECTION ====================
class GestureDetector:
    @staticmethod
    def get_fingers_up(landmarks):
        """Returns [index_up, middle_up, ring_up, pinky_up]"""
        fingers = []
        for tip, pip in zip(Config.FINGER_TIP_INDICES, Config.FINGER_PIP_INDICES):
            fingers.append(landmarks[tip].y < landmarks[pip].y)
        return fingers
    
    @staticmethod
    def is_eraser_gesture(hand_landmarks):
        """Eraser: ALL four fingers up"""
        fingers = GestureDetector.get_fingers_up(hand_landmarks.landmark)
        return all(fingers)
    
    @staticmethod
    def is_index_only_up(hand_landmarks):
        """Pen DOWN: ONLY index up"""
        fingers = GestureDetector.get_fingers_up(hand_landmarks.landmark)
        return fingers[0] and not any(fingers[1:])
    
    @staticmethod
    def is_pinch_gesture(hand_landmarks):
        """Pinch = PEN UP"""
        lm = hand_landmarks.landmark
        index_tip = np.array([lm[8].x, lm[8].y])
        thumb_tip = np.array([lm[4].x, lm[4].y])
        return np.linalg.norm(index_tip - thumb_tip) < Config.PINCH_THRESHOLD


# ==================== DRAWING ENGINE ====================
class DrawingEngine:
    def __init__(self, height, width):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.strokes = []
        self.current_stroke = None
    
    def clear(self):
        """Clear canvas completely"""
        self.canvas[:] = 0
        self.strokes.clear()
        self.current_stroke = None
    
    def soft_clear(self):
        """Fade canvas"""
        self.canvas = (self.canvas * 0.7).astype(np.uint8)
    
    def undo(self):
        """Remove last stroke"""
        if self.current_stroke:
            self.strokes.append(self.current_stroke)
            self.current_stroke = None
        if self.strokes:
            self.strokes.pop()
            self.redraw()
    
    def save(self):
        """Save canvas to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_note_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        return filename
    
    def start_stroke(self, mode, color, thickness):
        """Start a new stroke"""
        self.current_stroke = {
            "mode": mode,
            "color": color,
            "thickness": thickness,
            "points": []
        }
    
    def add_point(self, point):
        """Add point to current stroke"""
        if self.current_stroke:
            self.current_stroke["points"].append(point)
    
    def finish_stroke(self, apply_shape_correction=False):
        """Finish current stroke and optionally apply shape correction"""
        if not self.current_stroke:
            return
        
        if apply_shape_correction and self.current_stroke["mode"] == "draw":
            pts = self.current_stroke["points"]
            if len(pts) >= 5:
                shape_type, shape_params = autocorrect_stroke(pts)
                if shape_type:
                    print(f"[SHAPE] Detected {shape_type}!")
                    self.strokes.append({
                        "mode": "shape",
                        "shape_type": shape_type,
                        "shape_params": shape_params,
                        "color": self.current_stroke["color"],
                        "thickness": self.current_stroke["thickness"]
                    })
                    self.current_stroke = None
                    self.redraw()
                    return
                else:
                    print(f"[SHAPE] No shape detected (points: {len(pts)})")
        
        self.strokes.append(self.current_stroke)
        self.current_stroke = None
        self.redraw()
    
    def draw_line_to(self, prev_point, current_point, color, thickness):
        """Draw line segment"""
        if prev_point:
            cv2.line(self.canvas, prev_point, current_point, color, thickness)
    
    def redraw(self):
        """Redraw entire canvas from stroke history"""
        self.canvas[:] = 0
        
        for stroke in self.strokes:
            mode = stroke["mode"]
            
            if mode == "draw":
                self._draw_stroke(stroke)
            elif mode == "erase":
                self._erase_stroke(stroke)
            elif mode == "shape":
                self._draw_shape(stroke)
    
    def _draw_stroke(self, stroke):
        """Draw a freehand stroke"""
        pts = stroke["points"]
        if len(pts) < 2:
            return
        
        color = stroke["color"]
        thickness = stroke["thickness"]
        
        for i in range(1, len(pts)):
            cv2.line(self.canvas, pts[i-1], pts[i], color, thickness)
    
    def _erase_stroke(self, stroke):
        """Apply eraser stroke"""
        pts = stroke["points"]
        if len(pts) < 2:
            return
        
        thickness = stroke["thickness"]
        for i in range(1, len(pts)):
            cv2.line(self.canvas, pts[i-1], pts[i], (0, 0, 0), thickness)
    
    def _draw_shape(self, stroke):
        """Draw recognized shape"""
        shape_type = stroke["shape_type"]
        params = stroke["shape_params"]
        color = stroke["color"]
        thickness = stroke["thickness"]
        
        if shape_type == "line":
            x1, y1, x2, y2 = params
            cv2.line(self.canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                    color, thickness, cv2.LINE_AA)
        
        elif shape_type == "circle":
            cx, cy, r = params
            cv2.circle(self.canvas, (int(cx), int(cy)), int(r),
                      color, thickness, cv2.LINE_AA)
        
        elif shape_type in ("rectangle", "square"):
            corners = [(int(x), int(y)) for x, y in params]
            pts = np.array(corners + [corners[0]], dtype=np.int32)
            cv2.polylines(self.canvas, [pts], True, color, thickness, cv2.LINE_AA)


# ==================== STATE MANAGER ====================
class StateManager:
    def __init__(self):
        self.current_color = Config.COLOR_WHITE
        self.brush_thickness = Config.DEFAULT_BRUSH_THICKNESS
        self.whiteboard_mode = False
        self.recording = False
        self.video_writer = None
        
        # Tracking
        self.prev_point = None
        self.lost_frames = 0
        self.smooth_window = deque(maxlen=Config.SMOOTH_WINDOW)
        
        # Mode stability
        self.prev_stable_mode = None
        self.raw_mode_prev = None
        self.mode_stable_count = 0
    
    def smooth_point(self, new_point):
        """Smooth point using moving average"""
        self.smooth_window.append(new_point)
        xs = [p[0] for p in self.smooth_window]
        ys = [p[1] for p in self.smooth_window]
        return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
    
    def stabilize_mode(self, raw_mode):
        """Apply gesture debouncing"""
        if raw_mode == self.raw_mode_prev:
            self.mode_stable_count = min(self.mode_stable_count + 1, 
                                        Config.MODE_STABLE_FRAMES)
        else:
            self.raw_mode_prev = raw_mode
            self.mode_stable_count = 1
        
        if self.mode_stable_count >= Config.MODE_STABLE_FRAMES:
            return raw_mode
        return self.prev_stable_mode
    
    def reset_tracking(self):
        """Reset hand tracking state"""
        self.prev_point = None
        self.smooth_window.clear()
    
    def cycle_color(self):
        """Cycle through available colors"""
        colors = [Config.COLOR_WHITE, Config.COLOR_RED, 
                 Config.COLOR_GREEN, Config.COLOR_BLUE]
        try:
            idx = colors.index(self.current_color)
            self.current_color = colors[(idx + 1) % len(colors)]
        except ValueError:
            self.current_color = Config.COLOR_WHITE
        return self.get_color_name()
    
    def adjust_thickness(self, delta):
        """Adjust brush thickness"""
        self.brush_thickness = max(Config.MIN_BRUSH, 
                                   min(Config.MAX_BRUSH, 
                                       self.brush_thickness + delta))
    
    def get_color_name(self):
        """Get current color name"""
        color_map = {
            Config.COLOR_WHITE: "White",
            Config.COLOR_RED: "Red",
            Config.COLOR_GREEN: "Green",
            Config.COLOR_BLUE: "Blue"
        }
        return color_map.get(self.current_color, "Custom")


# ==================== UI RENDERER ====================
class UIRenderer:
    @staticmethod
    def draw_info(frame, state, mode):
        """Draw information overlay"""
        mode_text_map = {
            "erase": "MODE: ERASER (Open Palm)",
            "draw": "MODE: DRAW (Index only)",
            None: "MODE: PEN UP (Pinch / idle)"
        }
        
        info_lines = [
            mode_text_map.get(mode, "MODE: Unknown"),
            f"Color: {state.get_color_name()}  | Thickness: {state.brush_thickness}",
            "Keys: C=Clear, B=Soft Clear, S=Save, Z=Undo, 1-4 Colors, +/- Thickness, W Whiteboard, R Record, Q Quit"
        ]
        
        y_pos = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            y_pos += 25
    
    @staticmethod
    def draw_pointer(frame, point):
        """Draw finger pointer"""
        if point:
            cv2.circle(frame, point, 4, (0, 255, 0), 1)


# ==================== MAIN APPLICATION ====================
class AirWritingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("Error: Could not read from webcam.")
        
        h, w = frame.shape[:2]
        self.width, self.height = w, h
        
        self.drawing_engine = DrawingEngine(h, w)
        self.state = StateManager()
        self.gesture_detector = GestureDetector()
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_NUM_HANDS,
            min_detection_confidence=Config.DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.TRACKING_CONFIDENCE,
            model_complexity=Config.MODEL_COMPLEXITY
        )
        
        self.print_instructions()
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("AIR WRITING - Instructions")
        print("="*60)
        print("\nKEYBOARD CONTROLS:")
        print("  C - Clear canvas completely")
        print("  B - Soft clear (fade)")
        print("  S - Save canvas to file")
        print("  Z - Undo last stroke")
        print("  1-4 - Change color (White/Red/Green/Blue)")
        print("  +/- - Adjust brush thickness")
        print("  W - Toggle whiteboard mode")
        print("  R - Start/Stop recording")
        print("  Q - Quit application")
        print("\nGESTURES:")
        print("  Index finger only - Draw mode")
        print("  Pinch (thumb + index) - Pen up (no drawing)")
        print("  Open palm (4 fingers) - Eraser mode")
        print("="*60 + "\n")
    
    def process_hand_detection(self, frame):
        """Process hand detection and return gesture info"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        if not result.multi_hand_landmarks:
            self.state.lost_frames += 1
            if self.state.lost_frames > Config.MAX_LOST_FRAMES:
                self.state.reset_tracking()
            return None, None
        
        self.state.lost_frames = 0
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # Optional: Draw hand landmarks (comment out for better performance)
        # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Detect gestures
        if self.gesture_detector.is_eraser_gesture(hand_landmarks):
            raw_mode = "erase"
        elif self.gesture_detector.is_pinch_gesture(hand_landmarks):
            raw_mode = None
        elif self.gesture_detector.is_index_only_up(hand_landmarks):
            raw_mode = "draw"
        else:
            raw_mode = None
        
        # Get and smooth index finger position
        lm = hand_landmarks.landmark
        raw_x = int(lm[8].x * self.width)
        raw_y = int(lm[8].y * self.height)
        smoothed_point = self.state.smooth_point((raw_x, raw_y))
        
        return raw_mode, smoothed_point
    
    def handle_mode_transition(self, old_mode, new_mode, has_point):
        """Handle transitions between drawing modes"""
        if old_mode == new_mode:
            return
        
        # Finish previous stroke
        if old_mode in ("draw", "erase"):
            self.drawing_engine.finish_stroke(apply_shape_correction=(old_mode == "draw"))
        
        # Start new stroke
        if new_mode in ("draw", "erase") and has_point:
            thickness = (Config.ERASER_THICKNESS if new_mode == "erase" 
                        else self.state.brush_thickness)
            self.drawing_engine.start_stroke(new_mode, self.state.current_color, thickness)
            self.state.smooth_window.clear()
        
        self.state.prev_stable_mode = new_mode
    
    def handle_drawing(self, mode, current_point):
        """Handle drawing/erasing based on current mode"""
        if not current_point or mode not in ("draw", "erase"):
            self.state.prev_point = None
            return
        
        color = (0, 0, 0) if mode == "erase" else self.state.current_color
        thickness = (Config.ERASER_THICKNESS if mode == "erase" 
                    else self.state.brush_thickness)
        
        if self.state.prev_point:
            self.drawing_engine.draw_line_to(self.state.prev_point, current_point, 
                                            color, thickness)
        
        self.drawing_engine.add_point(current_point)
        self.state.prev_point = current_point
    
    def handle_keyboard(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        
        if key == ord('c'):
            self.drawing_engine.clear()
            print("[INFO] Canvas cleared.")
        
        elif key == ord('b'):
            self.drawing_engine.soft_clear()
            print("[INFO] Soft clear applied.")
        
        elif key == ord('s'):
            filename = self.drawing_engine.save()
            print(f"[INFO] Canvas saved as {filename}")
        
        elif key == ord('z'):
            self.drawing_engine.undo()
            print("[INFO] Undo last stroke.")
        
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            colors = [Config.COLOR_WHITE, Config.COLOR_RED, 
                     Config.COLOR_GREEN, Config.COLOR_BLUE]
            self.state.current_color = colors[key - ord('1')]
        
        elif key in [ord('+'), ord('=')]:
            self.state.adjust_thickness(1)
        
        elif key in [ord('-'), ord('_')]:
            self.state.adjust_thickness(-1)
        
        elif key == ord('w'):
            self.state.whiteboard_mode = not self.state.whiteboard_mode
            print(f"[INFO] Whiteboard mode: {self.state.whiteboard_mode}")
        
        elif key == ord('r'):
            self.state.recording = not self.state.recording
            if self.state.recording:
                self.start_recording()
            else:
                self.stop_recording()
        
        return True
    
    def start_recording(self):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"air_session_{timestamp}.mp4"
        self.state.video_writer = cv2.VideoWriter(
            filename, fourcc, 20.0, (self.width, self.height))
        print(f"[INFO] Recording started: {filename}")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.state.video_writer:
            self.state.video_writer.release()
            self.state.video_writer = None
            print("[INFO] Recording stopped.")
    
    def run(self):
        """Main application loop"""
        cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame not received, exiting.")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process hand detection
                raw_mode, current_point = self.process_hand_detection(frame)
                
                # Stabilize mode
                stable_mode = self.state.stabilize_mode(raw_mode)
                
                # Handle mode transitions
                self.handle_mode_transition(self.state.prev_stable_mode, 
                                           stable_mode, current_point is not None)
                
                # Handle drawing/erasing
                self.handle_drawing(stable_mode, current_point)
                
                # Render UI
                combined = cv2.addWeighted(frame, 0.5, self.drawing_engine.canvas, 0.5, 0)
                display_frame = (self.drawing_engine.canvas if self.state.whiteboard_mode 
                               else combined)
                
                UIRenderer.draw_info(display_frame, self.state, stable_mode)
                UIRenderer.draw_pointer(frame if not self.state.whiteboard_mode else display_frame, 
                                       current_point)
                
                # Recording
                if self.state.recording and self.state.video_writer:
                    self.state.video_writer.write(display_frame)
                
                cv2.imshow("Air Writing", display_frame)
                cv2.imshow("Canvas", self.drawing_engine.canvas)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard(key):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        # Finish any active stroke
        if self.drawing_engine.current_stroke:
            self.drawing_engine.finish_stroke(
                apply_shape_correction=(self.drawing_engine.current_stroke["mode"] == "draw"))
        
        self.hands.close()
        self.cap.release()
        if self.state.video_writer:
            self.state.video_writer.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Application closed successfully.")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        app = AirWritingApp()
        app.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        raise