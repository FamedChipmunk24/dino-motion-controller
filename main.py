"""
Dino Motion Controller
Tracks your body via webcam: jump to press Space, squat to hold Down.
Built for the Chrome dinosaur game (chrome://dino).
"""

import collections
import os
import sys
import time

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from pynput.keyboard import Controller, Key

# ---------------------------------------------------------------------------
# Settings (adjust with +/- keys while running)
# ---------------------------------------------------------------------------
JUMP_THRESHOLD = 0.06
COOLDOWN_SEC = 0.15
BASELINE_SAMPLES = 15
# Hips count as "off the ground" above this fraction of the jump threshold.
LIFTOFF_FRACTION = 0.4
# Upward hip speed (in jump-thresholds per second) that counts as a takeoff.
TAKEOFF_SPEED = 6.0
# Squat depth (in jump-thresholds below standing) that starts a duck, and the
# shallower depth that ends it, so the Down key doesn't flicker at the edge.
DUCK_FRACTION = 1.5
DUCK_RELEASE_FRACTION = 1.0

WINDOW_NAME = "Dino Motion Controller"
MODEL_FILENAME = "pose_landmarker_lite.task"


def resource_path(filename: str) -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


class JumpDetector:
    def __init__(self):
        self.baseline_y: float | None = None
        self.setup_samples: collections.deque[float] = collections.deque(maxlen=BASELINE_SAMPLES)
        self.last_jump_time = 0.0
        self.prev_y: float | None = None
        self.prev_time: float | None = None
        self.armed = True
        self.ducking = False
        self.threshold = JUMP_THRESHOLD
        self.jump_count = 0
        self.active = False

    def reset_to_setup(self) -> None:
        self.active = False
        self.baseline_y = None
        self.setup_samples.clear()
        self.prev_y = None
        self.prev_time = None
        self.armed = True
        self.ducking = False

    def track_setup_position(self, hip_y: float | None) -> None:
        if hip_y is not None:
            self.setup_samples.append(hip_y)

    def start_with_current_position(self, hip_y: float | None) -> tuple[bool, str]:
        samples = list(self.setup_samples)
        if hip_y is not None:
            samples.append(hip_y)

        if len(samples) < 3:
            return False, "Stand in frame so I can see you"

        self.baseline_y = float(np.median(samples))
        self.prev_y = None
        self.prev_time = None
        self.armed = True
        self.active = True
        return True, "Tracking active - jump to play!"

    def process(self, hip_y: float | None) -> tuple[bool, str]:
        if not self.active:
            return False, "Waiting to start"

        if hip_y is None:
            self.prev_y = None
            self.ducking = False
            return False, "Lost tracking - stand in frame"

        now = time.time()
        rise = self.baseline_y - hip_y
        speed = 0.0
        if self.prev_y is not None and now > self.prev_time:
            speed = (self.prev_y - hip_y) / (now - self.prev_time)
        self.prev_y = hip_y
        self.prev_time = now

        if rise <= -self.threshold * DUCK_FRACTION:
            self.ducking = True
        elif rise > -self.threshold * DUCK_RELEASE_FRACTION:
            self.ducking = False
        if self.ducking:
            self.armed = True
            return False, "DUCK"

        if rise < self.threshold * LIFTOFF_FRACTION:
            self.armed = True
            return False, "Ready"

        taking_off = speed >= self.threshold * TAKEOFF_SPEED
        if self.armed and (taking_off or rise >= self.threshold):
            if now - self.last_jump_time >= COOLDOWN_SEC:
                self.armed = False
                self.last_jump_time = now
                self.jump_count += 1
                return True, "JUMP!"

        return False, "Airborne..."


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    points = []
    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[conn.start]
        end = points[conn.end]
        cv2.line(frame, start, end, (0, 200, 255), 2)


def draw_baseline_lines(frame, detector: JumpDetector):
    if detector.baseline_y is None:
        return

    h, w = frame.shape[:2]
    baseline_px = int(detector.baseline_y * h)
    cv2.line(frame, (0, baseline_px), (w, baseline_px), (255, 100, 0), 2)
    jump_line = int((detector.baseline_y - detector.threshold) * h)
    cv2.line(frame, (0, jump_line), (w, jump_line), (0, 200, 255), 2)
    duck_line = int((detector.baseline_y + detector.threshold * DUCK_FRACTION) * h)
    cv2.line(frame, (0, duck_line), (w, duck_line), (255, 0, 255), 2)
    cv2.putText(
        frame,
        "duck line",
        (8, duck_line + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
    )
    cv2.putText(
        frame,
        "standing",
        (8, baseline_px - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 100, 0),
        1,
    )
    cv2.putText(
        frame,
        "jump line",
        (8, jump_line - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        1,
    )


def draw_setup_overlay(frame, has_body: bool):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 150), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    title = "GET IN POSITION"
    cv2.putText(frame, title, (w // 2 - 140, h - 118), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)

    if has_body:
        msg = "Stand how you want, then press  S  or  ENTER  to set standing height"
        color = (180, 255, 180)
    else:
        msg = "Step into frame so your body is visible"
        color = (0, 165, 255)

    cv2.putText(frame, msg, (20, h - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    cv2.putText(frame, "[S] Start   [Q] Quit", (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def draw_hud(frame, status: str, detector: JumpDetector):
    h, w = frame.shape[:2]
    bar_h = 130 if detector.active else 90

    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)

    color = (0, 255, 0)
    if "JUMP" in status:
        color = (0, 255, 255)
    elif "Lost" in status or "Waiting" in status:
        color = (0, 165, 255)

    cv2.putText(frame, f"Status: {status}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if detector.active:
        cv2.putText(
            frame,
            f"Jumps sent: {detector.jump_count}  |  Sensitivity: {detector.threshold:.3f}",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "[+] more sensitive  [-] less sensitive  [R] recalibrate  [Q] quit",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 160, 160),
            1,
        )
        cv2.putText(
            frame,
            "Click into the game, then jump (Space) or squat (Down) here to play",
            (12, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 160, 160),
            1,
        )
    else:
        cv2.putText(
            frame,
            "Setup mode - jumps are disabled until you press Start",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )


def press_space(keyboard: Controller) -> None:
    keyboard.press(Key.space)
    keyboard.release(Key.space)


def create_pose_landmarker() -> vision.PoseLandmarker:
    model_path = resource_path(MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Make sure pose_landmarker_lite.task is in the same folder as the program."
        )

    with open(model_path, "rb") as f:
        model_buffer = f.read()

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_buffer),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    keyboard = Controller()
    detector = JumpDetector()
    status = "Get in position, then press S to start"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        input("Press Enter to exit...")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    landmarker = create_pose_landmarker()
    frame_timestamp = 0
    duck_held = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            frame_timestamp = max(frame_timestamp + 1, int(time.monotonic() * 1000))
            results = landmarker.detect_for_video(mp_image, frame_timestamp)

            hip_y = None
            has_body = False
            if results.pose_landmarks:
                has_body = True
                landmarks = results.pose_landmarks[0]
                left_hip = landmarks[vision.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[vision.PoseLandmark.RIGHT_HIP]
                hip_y = (left_hip.y + right_hip.y) / 2.0

                draw_pose(frame, landmarks)

            if not detector.active:
                detector.track_setup_position(hip_y)
                if has_body and hip_y is not None:
                    preview_px = int(hip_y * h)
                    cv2.line(frame, (0, preview_px), (w, preview_px), (100, 100, 255), 1)
                draw_setup_overlay(frame, has_body)
            else:
                draw_baseline_lines(frame, detector)
                jumped, status = detector.process(hip_y)
                if jumped:
                    press_space(keyboard)
                    cv2.circle(frame, (w // 2, h // 2), 60, (0, 255, 255), 4)

            want_duck = detector.active and detector.ducking
            if want_duck != duck_held:
                if want_duck:
                    keyboard.press(Key.down)
                else:
                    keyboard.release(Key.down)
                duck_held = want_duck

            draw_hud(frame, status, detector)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S"), 13):  # S or Enter
                if not detector.active:
                    ok_start, status = detector.start_with_current_position(hip_y)
                    if ok_start:
                        print("Baseline set. Jump tracking is active.")
            if key in (ord("r"), ord("R")):
                detector.reset_to_setup()
                status = "Get in position, then press S to start"
            if detector.active and key in (ord("+"), ord("=")):
                detector.threshold = max(0.03, detector.threshold - 0.005)
            if detector.active and key in (ord("-"), ord("_")):
                detector.threshold = min(0.15, detector.threshold + 0.005)
    finally:
        if duck_held:
            keyboard.release(Key.down)
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        input("Press Enter to exit...")
        raise
