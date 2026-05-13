from src.camera_selector import select_camera_gui
from src.concurrent_video_display import ConcurrentVideoCaptureDisplay


def main():
    idx = select_camera_gui(max_index=8)
    if idx is None:
        return

    player = ConcurrentVideoCaptureDisplay(idx, mirror=True, resolution=(1280, 720))
    player.run()


if __name__ == "__main__":
    main()
