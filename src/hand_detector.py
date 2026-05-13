import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe hand landmark topology (21 keypoints).
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class HandDetector:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self._smoothed_hands = []
        self._hold_frames = 3
        self._hold_remaining = 0
        self._smoothing_alpha = 0.65

    def detect_from_frame(self, frame):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self.timestamp_ms:
            now_ms = self.timestamp_ms + 1
        self.timestamp_ms = now_ms

        results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        if not results.hand_landmarks:
            if self._hold_remaining > 0 and self._smoothed_hands:
                height, width = frame.shape[:2]
                for hand_points in self._smoothed_hands:
                    self._draw_one_hand(frame, hand_points, width, height)
                self._hold_remaining -= 1
                return True
            return False

        current_hands = []
        for hand_landmarks in results.hand_landmarks:
            current_hands.append([(lm.x, lm.y) for lm in hand_landmarks])

        self._smoothed_hands = self._smooth_hands(current_hands)
        self._hold_remaining = self._hold_frames

        height, width = frame.shape[:2]
        for hand_points in self._smoothed_hands:
            self._draw_one_hand(frame, hand_points, width, height)

        return True

    def _smooth_hands(self, current_hands):
        if not self._smoothed_hands:
            return current_hands

        smoothed = []
        for hand_idx, current_points in enumerate(current_hands):
            if hand_idx >= len(self._smoothed_hands):
                smoothed.append(current_points)
                continue

            prev_points = self._smoothed_hands[hand_idx]
            if len(prev_points) != len(current_points):
                smoothed.append(current_points)
                continue

            merged = []
            for (cx, cy), (px, py) in zip(current_points, prev_points):
                x = self._smoothing_alpha * cx + (1.0 - self._smoothing_alpha) * px
                y = self._smoothing_alpha * cy + (1.0 - self._smoothing_alpha) * py
                merged.append((x, y))
            smoothed.append(merged)

        return smoothed

    def _draw_one_hand(self, frame, hand_points, width, height):
        for start_idx, end_idx in _HAND_CONNECTIONS:
            sx, sy = hand_points[start_idx]
            ex, ey = hand_points[end_idx]
            start_pt = (int(sx * width), int(sy * height))
            end_pt = (int(ex * width), int(ey * height))
            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

        for x, y in hand_points:
            point = (int(x * width), int(y * height))
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

    def reset_state(self):
        self._smoothed_hands = []
        self._hold_remaining = 0

    def close(self):
        self.detector.close()
