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
    EMPTY = np.uint8(2)


class PieceType(Enum):
    """Represents a chess piece type."""

    EMPTY = np.uint8(0)
    PAWN = np.uint8(1)
    ROOK = np.uint8(2)
    KNIGHT = np.uint8(3)
    BISHOP = np.uint8(4)
    QUEEN = np.uint8(5)
    KING = np.uint8(6)

    def __str__(self) -> str:
        return ".PRNBQK"[self.value]  # type: ignore


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
        return (
            ColorType.EMPTY
            if self == PieceState.EMPTY
            else ColorType.BLACK
            if self.value & _COL_MASK
            else ColorType.WHITE
        )

    @property
    def type(self) -> PieceType:
        """Returns the type of the piece."""
        return PieceType(int(self.value & _TYP_MASK))

    @classmethod
    def get(cls, piece: PieceType, color: ColorType) -> "PieceState":
        """Returns the piece state for a given type and color."""
        return cls(piece.value + color.value * 8)


class Piece(NamedTuple):
    """Represents a chess piece in a given state."""

    state: PieceState
    pos: str

    def __str__(self) -> str:
        if self.empty:
            return ".."
        return str(self.type) + "wb"[self.color.value]  # type: ignore

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


class Move(NamedTuple):
    """Represents a chess move."""

    player: ColorType
    piece: Piece
    start: str
    end: str
    is_valid: bool
    repr: str
    is_capture: bool = False
    is_capture_type_: PieceType | None = None
    is_castle: bool = False
    is_double_pawn_push: bool = False
    is_long_castle: bool = False
    is_promotion: bool = False
    is_promotion_to: PieceType | None = None

    def __str__(self) -> str:
        """Returns the move as a standardized string."""
        return self.repr

    @classmethod
    def invalid(cls) -> "Move":
        """Returns an invalid move."""
        return cls(
            player=ColorType.EMPTY,
            piece=Piece(PieceState.EMPTY, ""),
            start="",
            end="",
            repr="",
            is_valid=False,
            is_capture=False,
            is_castle=False,
            is_long_castle=False,
            is_promotion=False,
            is_promotion_to=None,
        )

    @property
    def is_capture_type(self) -> PieceType:
        """Returns the type of the captured piece."""
        if self.is_capture_type_ is None:
            raise ValueError("No capture type for this move.")
        return self.is_capture_type_
