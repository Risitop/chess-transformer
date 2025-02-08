from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import NamedTuple


class ColorType(Enum):
    """Represents a chess piece color."""

    WHITE = 0
    BLACK = 1


class PieceType(Enum):
    """Represents a chess piece."""

    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5


class Piece(NamedTuple):
    type: PieceType | None
    color: ColorType | None

    def __repr__(self) -> str:
        if self.empty:
            return ".."
        return "PNBRQK"[self.type.value] + "wb"[self.color.value]  # type: ignore

    @property
    def empty(self) -> bool:
        """Returns True if the piece is empty."""
        return self.type is None


@dataclass
class GameState:
    """A compact representation of a chess game state."""

    # 8 bits = one row, smallest bit is a1, then b1, largest is h8
    w_pawn: np.uint64
    w_rook: np.uint64
    w_knight: np.uint64
    w_bishop: np.uint64
    w_queen: np.uint64
    w_king: np.uint64
    b_pawn: np.uint64
    b_rook: np.uint64
    b_knight: np.uint64
    b_bishop: np.uint64
    b_queen: np.uint64
    b_king: np.uint64

    # Other state information
    has_castled: bool = False
    current_player: ColorType = ColorType.WHITE
    in_check: ColorType | None = None

    @classmethod
    def empty(cls) -> "GameState":
        """Creates an empty game state."""
        return cls(
            w_pawn=np.uint64(0),
            w_rook=np.uint64(0),
            w_knight=np.uint64(0),
            w_bishop=np.uint64(0),
            w_queen=np.uint64(0),
            w_king=np.uint64(0),
            b_pawn=np.uint64(0),
            b_rook=np.uint64(0),
            b_knight=np.uint64(0),
            b_bishop=np.uint64(0),
            b_queen=np.uint64(0),
            b_king=np.uint64(0),
        )

    @classmethod
    def initialize(cls) -> "GameState":
        """Initializes the game state."""
        return cls(
            w_pawn=np.uint64(0x000000000000FF00),
            w_rook=np.uint64(0x0000000000000081),
            w_knight=np.uint64(0x0000000000000042),
            w_bishop=np.uint64(0x0000000000000024),
            w_queen=np.uint64(0x0000000000000008),
            w_king=np.uint64(0x0000000000000010),
            b_pawn=np.uint64(0x00FF000000000000),
            b_rook=np.uint64(0x8100000000000000),
            b_knight=np.uint64(0x4200000000000000),
            b_bishop=np.uint64(0x2400000000000000),
            b_queen=np.uint64(0x0800000000000000),
            b_king=np.uint64(0x1000000000000000),
        )

    def __getitem__(self, pos: tuple[int, int] | str) -> Piece:
        """Returns the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_tuple(pos)
        if not (0 <= pos[0] < 8 and 0 <= pos[1] < 8):
            raise IndexError(f"Position out of bounds: {pos}")
        mask = np.uint64(1) << (pos[0] * 8 + pos[1])
        if self.w_pawn & mask:
            return Piece(PieceType.PAWN, ColorType.WHITE)
        if self.w_rook & mask:
            return Piece(PieceType.ROOK, ColorType.WHITE)
        if self.w_knight & mask:
            return Piece(PieceType.KNIGHT, ColorType.WHITE)
        if self.w_bishop & mask:
            return Piece(PieceType.BISHOP, ColorType.WHITE)
        if self.w_queen & mask:
            return Piece(PieceType.QUEEN, ColorType.WHITE)
        if self.w_king & mask:
            return Piece(PieceType.KING, ColorType.WHITE)
        if self.b_pawn & mask:
            return Piece(PieceType.PAWN, ColorType.BLACK)
        if self.b_rook & mask:
            return Piece(PieceType.ROOK, ColorType.BLACK)
        if self.b_knight & mask:
            return Piece(PieceType.KNIGHT, ColorType.BLACK)
        if self.b_bishop & mask:
            return Piece(PieceType.BISHOP, ColorType.BLACK)
        if self.b_queen & mask:
            return Piece(PieceType.QUEEN, ColorType.BLACK)
        if self.b_king & mask:
            return Piece(PieceType.KING, ColorType.BLACK)
        return Piece(None, None)


def _str_pos_to_tuple(pos: str) -> tuple[int, int]:
    """Converts a string position to a tuple: 'a1' -> (0, 0) etc."""
    return int(pos[1]) - 1, ord(pos[0]) - ord("a")


if __name__ == "__main__":
    board = GameState.initialize()
    for row in range(8, 0, -1):
        for col in "abcdefgh":
            print(board[col + str(row)], end=" ")
        print()
