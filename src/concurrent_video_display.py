from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from src.hand_detector import HandDetector


def _detect_backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return 0


class ConcurrentVideoCaptureDisplay:
    """백그라운드에서 프레임을 캡처하고 일정 간격으로 화면에 출력하는 클래스입니다."""

    def __init__(
        self,
        camera_index: int,
        *,
        window_name: str = "Camera",
        target_fps: float = 30.0,
        backend: int | None = None,
        mirror: bool = False,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        """카메라 인덱스와 출력 설정을 저장하고 내부 상태를 초기화합니다."""
        self.camera_index = camera_index
        self.window_name = window_name
        self.target_fps = target_fps if target_fps > 0 else 30.0
        self.frame_interval = 1.0 / self.target_fps
        self.backend = _detect_backend() if backend is None else backend
        self.mirror = mirror
        self.resolution = resolution

        self._capture: cv2.VideoCapture | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._capture_future: Future[None] | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._actual_width: int = 0
        self._actual_height: int = 0

        self._enable_hand_detector = True
        self.hand_detector = HandDetector()

    def _configure_capture_mode(self) -> None:
        """설정된 해상도로 카메라를 구성하기. 단, None이면 원본 해상도를 유지한 후 실제 적용된 해상도를 저장한다."""
        if self._capture is None:
            return

        if self.resolution is not None:
            target_w, target_h = self.resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def initialize_window(self) -> None:
        """실제 적용된 해상도로 창을 만듭니다."""
        if self._actual_width <= 0 or self._actual_height <= 0:
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(self.window_name, self._actual_width, self._actual_height)

    def start(self) -> None:
        """카메라를 열고 캡처와 출력을 병렬로 처리하는 루프를 시작합니다."""
        if self._capture is not None:
            return

        self._stop_event.clear()
        self._capture = cv2.VideoCapture(self.camera_index, self.backend) if self.backend else cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")

        self._configure_capture_mode()
        self.initialize_window()

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-capture")
        self._capture_future = self._executor.submit(self._capture_loop)

        try:
            self._display_loop()
        finally:
            self.stop()
            self.close()

    def run(self) -> None:
        """start()를 실행하는 편의 메서드입니다."""
        self.start()

    def stop(self) -> None:
        """실행 중인 루프를 종료하도록 중단 신호를 보냅니다."""
        self._stop_event.set()

    def close(self) -> None:
        """카메라 자원을 해제하고 OpenCV 창을 닫습니다."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        try:
            self.hand_detector.close()
        except Exception:
            pass

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._capture_future = None

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def _capture_loop(self) -> None:
        """카메라에서 최신 프레임을 계속 읽어 공유 버퍼에 저장합니다."""
        assert self._capture is not None

        while not self._stop_event.is_set():
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame.copy()
            else:
                time.sleep(min(self.frame_interval, 0.01))

    def _display_loop(self) -> None:
        """고정된 간격으로 프레임을 출력해 화면 지연이 일정하게 유지되도록 합니다."""
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            if self._capture_future is not None and self._capture_future.done():
                error = self._capture_future.exception()
                if error is not None:
                    raise error

            frame = self._get_latest_frame()
            if frame is not None:
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # 손 감지하여 frame에 반영
                if self._enable_hand_detector:
                    self.hand_detector.detect_from_frame(frame)

                cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.stop()
                break
            if key == 13:  # ENTER
                self._enable_hand_detector = not self._enable_hand_detector
                if not self._enable_hand_detector:
                    self.hand_detector.reset_state()

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.stop()
                    break
            except Exception:
                self.stop()
                break

            next_tick += self.frame_interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _get_latest_frame(self) -> np.ndarray | None:
        """가장 최근에 캡처된 프레임의 복사본을 반환합니다."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()
