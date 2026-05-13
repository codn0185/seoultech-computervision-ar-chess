import numpy as np


class Object3D:
    position: np.ndarray  # 시작 지점 (x, y, z) - 체스보드 기준 좌표
    vertexes: np.ndarray  # 정점 리스트 [(x, y, z), ...]
    edges: list[tuple[int, int]]  # 선분 리스트 [(v1, v2), ...]
