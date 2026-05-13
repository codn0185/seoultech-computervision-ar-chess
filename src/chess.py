import numpy as np

import chess
import chess.variant
from chess import Move

from src.object3d import Object3D


class Chess:
    def __init__(self):
        self.board = chess.variant.GiveawayBoard()

    def reset_board(self):
        """체스보드를 초기화한다."""
        self.board = chess.variant.GiveawayBoard()

    def piece_at(self, file_index, rank_index):
        """주어진 위치의 말을 반환한다."""
        return self.board.piece_at(chess.square(file_index, rank_index))

    def move(self, uci):
        """체스 말을 이동시킨다."""
        self.board.push(Move.from_uci(uci))

    def undo(self):
        """체스 말의 직전 이동을 되돌린다."""
        self.board.pop()


class ChessPiece(Object3D):
    pass


# 1x1x1 크기의 정육면체 (임시)
class Cube3D(Object3D):
    size = 50.0

    position = np.array([2 * size, 3 * size, -size / 2])

    vertexes = np.array(
        [
            [-size / 2, -size / 2, -size / 2],
            [size / 2, -size / 2, -size / 2],
            [size / 2, size / 2, -size / 2],
            [-size / 2, size / 2, -size / 2],
            [-size / 2, -size / 2, size / 2],
            [size / 2, -size / 2, size / 2],
            [size / 2, size / 2, size / 2],
            [-size / 2, size / 2, size / 2],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
