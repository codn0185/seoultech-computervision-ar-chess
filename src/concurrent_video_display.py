from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from src.hand_detector import HandDetector
from src.object3d import Object3D


def _detect_backend() -> int:
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return 0


class ConcurrentVideoCaptureDisplay:
    """백그라운드에서 프레임을 캡처하고 일정 간격으로 화면에 출력하는 클래스입니다."""

    _DEFAULT_CHESSBOARD_PATTERN_SIZE: tuple[int, int] = (7, 7)  # 교점 개수 (w, h)

    def __init__(
        self,
        camera_index: int,
        *,
        window_name: str = "Camera",
        target_fps: float = 30.0,
        backend: int | None = None,
        mirror: bool = False,
        resolution: tuple[int, int] | None = None,
        chessboard_pattern_size: tuple[int, int] | None = None,
        chessboard_square_size: float = 25.0,
        chessboard_calibration_views: int = 12,
    ) -> None:
        """카메라 인덱스와 출력 설정을 저장하고 내부 상태를 초기화합니다."""
        self.camera_index = camera_index
        self.window_name = window_name
        self.target_fps = target_fps if target_fps > 0 else 30.0
        self.frame_interval = 1.0 / self.target_fps
        self.backend = _detect_backend() if backend is None else backend
        self.mirror = mirror
        self.resolution = resolution

        self._chessboard_pattern_size = chessboard_pattern_size if chessboard_pattern_size is not None else self._DEFAULT_CHESSBOARD_PATTERN_SIZE
        self.chessboard_square_size = chessboard_square_size if chessboard_square_size > 0 else 1.0
        self.chessboard_calibration_views = chessboard_calibration_views if chessboard_calibration_views > 0 else 12

        self._capture: cv2.VideoCapture | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._capture_future: Future[None] | None = None
        self._processing_future: Future[None] | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._processed_frame: np.ndarray | None = None  # 손 감지 + 미러링 적용된 프레임
        self._actual_width: int = 0
        self._actual_height: int = 0

        self._chessboard_object_points: np.ndarray | None = None
        self._chessboard_image_points: list[np.ndarray] = []
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None
        self._pose_ready = False
        self._last_detected_corners: np.ndarray | None = None  # 현재 프레임에서 감지된 체스보드 코너
        self._object_3d_list: list[Object3D] = []  # 렌더링할 3D 객체 리스트
        self._chessboard_detection_interval = 50  # 감지되지 않을 때 몇 프레임마다 탐색할지 (다운스케일과 함께 사용)
        self._frame_count = 0  # 현재 프레임 카운트

        # Hand Detector
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

    def _create_chessboard_object_points(self, pattern_size: tuple[int, int]) -> np.ndarray:
        cols, rows = pattern_size
        object_points = np.zeros((rows * cols, 1, 3), dtype=np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
        object_points[:, 0, :2] = grid * self.chessboard_square_size
        return object_points

    def _detect_chessboard_corners(
        self,
        frame: np.ndarray,
    ) -> tuple[bool, np.ndarray | None]:
        """체스보드 모서리를 감지합니다. 항상 설정된 단일 패턴 크기를 사용합니다.

        다운스케일된 프레임에서 탐지하여 연산 시간을 단축하고 프레임 끊김을 줄입니다.
        """
        # 다운스케일하여 연산 시간 단축 (4배 빨라짐)
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        found, corners = self._find_chessboard_corners(gray, self._chessboard_pattern_size)

        if not found or corners is None:
            return False, None

        # 다운스케일된 좌표를 원본 해상도로 복원
        if corners is not None:
            corners = corners / scale_factor

        if self._chessboard_object_points is None:
            self._chessboard_object_points = self._create_chessboard_object_points(self._chessboard_pattern_size)

        return True, corners

    def _find_chessboard_corners(
        self,
        gray: np.ndarray,
        pattern_size: tuple[int, int],
    ) -> tuple[bool, np.ndarray | None]:
        corners = None
        found = False

        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                found, corners = cv2.findChessboardCornersSB(
                    gray,
                    pattern_size,
                    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                )
            except cv2.error:
                found = False
                corners = None

        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found and corners is not None:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if not found or corners is None:
            return False, None

        corners = np.asarray(corners, dtype=np.float32)
        if corners.ndim == 2:
            corners = corners.reshape(-1, 1, 2)
        return True, corners

    def _maybe_calibrate_from_chessboard(self, image_size: tuple[int, int]) -> bool:
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            return True

        if self._chessboard_object_points is None:
            return False

        if len(self._chessboard_image_points) < 3:
            return False

        calibration_image_points = [points.copy() for points in self._chessboard_image_points]
        calibration_object_points = [self._chessboard_object_points.copy() for _ in calibration_image_points]

        try:
            _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
                calibration_object_points,
                calibration_image_points,
                image_size,
                None,
                None,
            )
        except cv2.error:
            return False

        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        return True

    def _estimate_pose_from_chessboard(self, corners: np.ndarray) -> bool:
        if self._chessboard_object_points is None:
            return False
        if self._camera_matrix is None or self._dist_coeffs is None:
            return False

        try:
            success, rvec, tvec = cv2.solvePnP(
                self._chessboard_object_points,
                corners,
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return False

        if not success:
            return False

        self._rvec = rvec
        self._tvec = tvec
        self._pose_ready = True
        return True

    def _should_detect_chessboard_and_update(self, frame: np.ndarray) -> bool:
        """체스보드 탐지 조건을 판단하고 필요시에만 탐지/업데이트합니다.

        - 캘리브레이션 완료 전:
          * 아직 감지되지 않음: 일정 간격마다만 탐색 (CPU 절감)
          * 이미 감지됨: 매 프레임마다 탐색 (필요한 뷰 모으기)
        - 캘리브레이션 완료 후:
          * 매 프레임마다 포즈 추정 (실시간 추적)
        """
        # 캘리브레이션이 완료됨 → 항상 포즈 업데이트
        if self._camera_matrix is not None and self._dist_coeffs is not None:
            return self._update_chessboard_pose(frame)

        # 캘리브레이션 중:
        # - 이미 감지된 적 있거나 (현재 수집 중)
        # - 현재 프레임이 탐지 간격에 일치
        should_detect = len(self._chessboard_image_points) > 0 or self._frame_count % self._chessboard_detection_interval == 0

        if should_detect:
            return self._update_chessboard_pose(frame)

        self._pose_ready = False
        return False

    def _update_chessboard_pose(self, frame: np.ndarray) -> bool:
        found, corners = self._detect_chessboard_corners(frame)
        if not found or corners is None:
            self._pose_ready = False
            self._last_detected_corners = None
            return False

        # 실시간 감지된 코너 저장 (디스플레이 스레드에서 포즈 추정용)
        self._last_detected_corners = corners.copy()

        if len(self._chessboard_image_points) < self.chessboard_calibration_views:
            should_store = True
            if self._chessboard_image_points:
                last_corners = self._chessboard_image_points[-1]
                current = corners.reshape(-1, 2)
                previous = last_corners.reshape(-1, 2)
                mean_shift = float(np.mean(np.linalg.norm(current - previous, axis=1)))
                should_store = mean_shift >= 3.0

            if should_store:
                self._chessboard_image_points.append(corners.copy())

        image_size = (frame.shape[1], frame.shape[0])
        calibrated = self._maybe_calibrate_from_chessboard(image_size)
        # 포즈 추정은 디스플레이 스레드에서 수행
        # if calibrated:
        #     self._estimate_pose_from_chessboard(corners)

        cv2.drawChessboardCorners(frame, self._chessboard_pattern_size, corners, True)
        # 칼리브레이션 완료 여부만 반환 (포즈 준비 상태는 디스플레이 스레드에서 업데이트)
        return calibrated

    def _estimate_pose_if_ready(self) -> bool:
        """칼리브레이션이 완료되었으면 실시간 감지된 체스보드로 포즈를 추정합니다.

        주의: 실시간제로 감지된 체스보드로 동적 포즈를 추정하여,
        체스보드가 움직일 때 3D 객체도 함께 움직임.
        """
        if self._camera_matrix is None or self._dist_coeffs is None:
            self._pose_ready = False
            return False

        # 실시간 감지된 체스보드 코너로 포즈 추정
        if self._last_detected_corners is not None:
            if self._estimate_pose_from_chessboard(self._last_detected_corners):
                return True

        self._pose_ready = False
        return False

    def add_3d_object(self, obj: Object3D) -> None:
        """렌더링할 3D 객체를 추가합니다.

        Args:
            obj: 추가할 Object3D 객체.
        """
        if obj not in self._object_3d_list:
            self._object_3d_list.append(obj)

    def remove_3d_object(self, obj: Object3D) -> None:
        """렌더링할 3D 객체를 제거합니다.

        Args:
            obj: 제거할 Object3D 객체.
        """
        if obj in self._object_3d_list:
            self._object_3d_list.remove(obj)

    def clear_3d_objects(self) -> None:
        """모든 3D 객체를 제거합니다."""
        self._object_3d_list.clear()

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

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="video-worker")
        self._capture_future = self._executor.submit(self._capture_loop)
        self._processing_future = self._executor.submit(self._processing_loop)

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
            self._processing_future = None

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

    def _processing_loop(self) -> None:
        """손 감지 및 체스보드 탐지를 처리합니다.

        캡처된 프레임을 받아서:
        1. 미러링 처리
        2. 손 감지
        3. 체스보드 탐지/칼리브레이션
        4. 처리 결과를 _processed_frame에 저장
        """
        while not self._stop_event.is_set():
            frame = self._get_latest_frame()
            if frame is None:
                time.sleep(0.001)  # CPU 점유 방지
                continue

            # 미러링 처리
            if self.mirror:
                frame = cv2.flip(frame, 1)

            # 손 감지
            if self._enable_hand_detector:
                self.hand_detector.detect_from_frame(frame)

            # 체스보드 탐지 및 칼리브레이션
            self._should_detect_chessboard_and_update(frame)

            # 처리 결과 저장
            with self._processing_lock:
                self._processed_frame = frame.copy()

            # 프레임 카운트 증가 (체스보드 탐지 간격 제어용)
            self._frame_count += 1

    def _display_loop(self) -> None:
        """고정된 간격으로 프레임을 출력해 화면 지연이 일정하게 유지되도록 합니다."""
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            if self._capture_future is not None and self._capture_future.done():
                error = self._capture_future.exception()
                if error is not None:
                    raise error

            if self._processing_future is not None and self._processing_future.done():
                error = self._processing_future.exception()
                if error is not None:
                    raise error

            # 처리 스레드에서 준비한 프레임 받기
            with self._processing_lock:
                frame = self._processed_frame.copy() if self._processed_frame is not None else None

            if frame is not None:
                # 포즈 추정 (디스플레이 스레드만 수행)
                chessboard_ready = self._estimate_pose_if_ready()

                if chessboard_ready:
                    status_text = "Chessboard pose ready"
                    # 모든 3D 객체 렌더링
                    for obj_3d in self._object_3d_list:
                        self._draw_3d_object_on_frame(frame, obj_3d, color_bgr=(0, 255, 0), thickness=2)
                elif self._camera_matrix is None:
                    status_text = f"Calibrating chessboard {len(self._chessboard_image_points)}/{self.chessboard_calibration_views}"
                else:
                    status_text = "Chessboard detected"

                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.stop()
                break
            if key == 13:  # ENTER
                self._enable_hand_detector = not self._enable_hand_detector
                if not self._enable_hand_detector:
                    self.hand_detector.reset_state()
            if key == 8:  # BACKSPACE
                pass

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

    def _draw_3d_object_on_frame(
        self,
        frame: np.ndarray,
        object3d: Object3D,
        color_bgr: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 1,
    ):
        if self._rvec is None or self._tvec is None or self._camera_matrix is None or self._dist_coeffs is None:
            return

        # position이 있으면 좌표에 오프셋 적용
        vertexes = object3d.vertexes.copy()
        if hasattr(object3d, "position") and object3d.position is not None:
            vertexes = vertexes + object3d.position

        points_2d, _ = cv2.projectPoints(
            vertexes,
            self._rvec,
            self._tvec,
            self._camera_matrix,
            self._dist_coeffs,
        )
        points_2d = points_2d.reshape(-1, 2).astype(int)
        for edge in object3d.edges:
            pt1 = tuple(points_2d[edge[0]])
            pt2 = tuple(points_2d[edge[1]])
            cv2.line(frame, pt1, pt2, color_bgr, thickness)
