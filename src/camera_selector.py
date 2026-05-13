import sys
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# 기본 미리보기 최대 바운딩 (16:9 비율)
DEFAULT_PREVIEW_SIZE = (320, 180)


def _detect_backend():
    """
    운영체제에 맞는 OpenCV 캡처 백엔드를 반환합니다.
    Windows에서는 `CAP_DSHOW`를 사용하고, 그 외 플랫폼은 기본값을 사용합니다.
    """
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return 0


def list_available_cameras(max_index=8, read_attempt=True):
    """
    시스템에서 사용 가능한 카메라 인덱스 목록을 반환합니다.
    `max_index`까지 탐색하며, `read_attempt`가 True이면 프레임을 읽어 실제로 작동하는지 확인합니다.
    """
    backend = _detect_backend()
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
        ok = False
        if cap.isOpened():
            if read_attempt:
                ret, _ = cap.read()
                ok = ret
            else:
                ok = True
        try:
            cap.release()
        except Exception:
            pass
        if ok:
            available.append(i)
    return available


def _fit_size(src_w, src_h, max_w, max_h):
    """
    주어진 원본 크기(src_w, src_h)를 비율을 유지하며
    최대 바운딩(max_w, max_h) 안에 맞춘 크기를 반환합니다.
    """
    if max_w is None or max_h is None:
        return src_w, src_h
    try:
        sw = float(src_w)
        sh = float(src_h)
        mw = float(max_w)
        mh = float(max_h)
        if sw <= 0 or sh <= 0:
            return int(max_w), int(max_h)
        scale = min(mw / sw, mh / sh)
        if scale >= 1.0:
            return int(sw), int(sh)
        return int(sw * scale), int(sh * scale)
    except Exception:
        return int(max_w), int(max_h)


class CameraSelector(tk.Tk):
    def __init__(self, max_index: int = 8, preview_size: tuple[int, int] | None = None):
        """
        윈도우를 초기화하고 UI 구성 요소를 준비합니다.
        `max_index`로 탐색 범위를 지정합니다.
        `preview_size`가 None이면 카메라 캡처 정보 또는 첫 프레임/위젯 크기에서 자동으로 결정합니다.
        """
        super().__init__()
        self.title("카메라 선택")
        self.preview_size = preview_size
        self.backend = _detect_backend()

        self.available = list_available_cameras(max_index)
        self.selected = None
        self._cap = None
        self._preview_job = None

        self._build_ui()

    def _build_ui(self):
        """
        GUI 레이아웃을 구성합니다.
        왼쪽에는 카메라 목록과 확인/취소 버튼을, 오른쪽에는 실시간 미리보기를 표시합니다.
        """
        frame = ttk.Frame(self)
        frame.pack(padx=8, pady=8)

        left = ttk.Frame(frame)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="사용 가능한 카메라:").pack(anchor="w")
        self.listbox = tk.Listbox(left, height=8)
        self.listbox.pack()
        for idx in self.available:
            self.listbox.insert(tk.END, f"Camera {idx}")

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_frame, text="확인", command=self._on_ok).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="취소", command=self._on_cancel).pack(side="left")

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, padx=(10, 0))
        ttk.Label(right, text="미리보기:").pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack()

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        if self.available:
            self.listbox.selection_set(0)
            self._open_preview(self.available[0])

    def _on_select(self, _ev=None):
        """
        리스트 박스에서 항목을 선택했을 때 호출됩니다.
        선택된 인덱스의 카메라 미리보기를 연다.
        """
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = int(self.available[sel[0]])
        self._open_preview(idx)

    def _open_preview(self, index):
        """
        지정한 카메라 인덱스로 VideoCapture를 열고 미리보기 업데이트를 시작합니다.
        실패하면 캡처 객체를 정리합니다.
        """
        self._close_preview()
        try:
            self._cap = cv2.VideoCapture(index, self.backend) if self.backend else cv2.VideoCapture(index)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                return
        except Exception:
            self._cap = None
            return
        # preview_size가 None이면 우선 위젯 크기를 바운딩으로 사용하고,
        # 없으면 기본 바운딩을 사용한다.
        if self.preview_size is None:
            try:
                # 레이아웃이 아직 완성되지 않았으면 최신화
                self.update_idletasks()
                lw = self.preview_label.winfo_width()
                lh = self.preview_label.winfo_height()
                if lw > 10 and lh > 10:
                    self.preview_size = (lw, lh)
            except Exception:
                pass
            if self.preview_size is None:
                # 안전한 기본값 (16:9 비율)
                self.preview_size = DEFAULT_PREVIEW_SIZE

        self._update_preview()

    def _update_preview(self):
        """
        현재 열려 있는 캡처에서 프레임을 읽어 Tkinter 레이블에 표시합니다.
        주기적으로 자신을 다시 예약하여 실시간 미리보기를 유지합니다.
        """
        if not self._cap:
            return
        ret, frame = self._cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            max_w, max_h = (None, None)
            if self.preview_size is not None:
                max_w, max_h = self.preview_size
            else:
                max_w, max_h = DEFAULT_PREVIEW_SIZE
            new_w, new_h = _fit_size(w, h, max_w, max_h)
            img = Image.fromarray(frame)
            try:
                img = img.resize((new_w, new_h), Image.LANCZOS)
            except Exception:
                img.thumbnail((new_w, new_h))
            self._tkimg = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self._tkimg)
        self._preview_job = self.after(30, self._update_preview)

    def _close_preview(self):
        """
        미리보기 타이머를 취소하고 VideoCapture를 안전하게 닫습니다.
        """
        if self._preview_job:
            try:
                self.after_cancel(self._preview_job)
            except Exception:
                pass
            self._preview_job = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _on_ok(self):
        """
        확인 버튼이 눌렸을 때 호출됩니다.
        선택한 카메라 인덱스를 `self.selected`에 저장하고 GUI를 닫습니다.
        """
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("선택 없음", "카메라를 선택하세요.")
            return
        self.selected = self.available[sel[0]]
        self._close_preview()
        self.destroy()

    def _on_cancel(self):
        """
        취소 버튼이 눌렸을 때 호출됩니다. 선택을 취소하고 GUI를 닫습니다.
        """
        self.selected = None
        self._close_preview()
        self.destroy()

    def run(self):
        """
        GUI 루프를 실행하고 종료 후 선택된 카메라 인덱스를 반환합니다.
        """
        self.mainloop()
        return self.selected


def select_camera_gui(max_index=8):
    """
    `CameraSelector`를 생성하고 실행한 뒤 선택된 카메라 인덱스를 반환하는 편의 함수입니다.
    """
    sel = CameraSelector(max_index=max_index)
    return sel.run()


if __name__ == "__main__":
    idx = select_camera_gui(max_index=8)
    print("selected:", idx)
