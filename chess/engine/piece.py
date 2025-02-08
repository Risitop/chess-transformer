"""This module defines the Piece class and related state enums."""

import numpy as np
from enum import Enum
from typing import NamedTuple

_COL_MASK = np.uint8(8)
_TYP_MASK = np.uint8(7)


class ColorType(Enum):
    """Represents a chess piece color."""

    WHITE = np.uint8(0)
    BLACK = np.uint8(1)


class PieceType(Enum):
    """Represents a chess piece type."""

    EMPTY = np.uint8(0)
    PAWN = np.uint8(1)
    ROOK = np.uint8(2)
    KNIGHT = np.uint8(3)
    BISHOP = np.uint8(4)
    QUEEN = np.uint8(5)
    KING = np.uint8(6)


class PieceState(Enum):
    """Represents a chess piece as an 8-bit integer."""

    EMPTY = np.uint8(0)
    PAWN_W = np.uint8(1)
    ROOK_W = np.uint8(2)
    KNIGHT_W = np.uint8(3)
    BISHOP_W = np.uint8(4)
    QUEEN_W = np.uint8(5)
    KING_W = np.uint8(6)
    PAWN_B = np.uint8(9)
    ROOK_B = np.uint8(10)
    KNIGHT_B = np.uint8(11)
    BISHOP_B = np.uint8(12)
    QUEEN_B = np.uint8(13)
    KING_B = np.uint8(14)

    @property
    def color(self) -> ColorType:
        """Returns the color of the piece."""
        return ColorType.BLACK if self.value & _COL_MASK else ColorType.WHITE

    @property
    def type(self) -> PieceType:
        """Returns the type of the piece."""
        return PieceType(int(self.value & _TYP_MASK))


class Piece(NamedTuple):
    state: PieceState
    pos: np.uint64

    def __repr__(self) -> str:
        if self.empty:
            return ".."
        return ".PNBRQK"[self.type.value] + "wb"[self.color.value]  # type: ignore

    @property
    def color(self) -> ColorType:
        """Returns the piece's color."""
        return self.state.color

    @property
    def type(self) -> PieceType:
        """Returns the piece's type."""
        return self.state.type

    @property
    def empty(self) -> bool:
        """Returns True if the piece is empty."""
        return self.type == PieceType.EMPTY
