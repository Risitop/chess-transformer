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

    piece: Piece
    end: str
    captures: Piece | None = None
    is_castle: bool = False
    is_double_pawn_push: bool = False
    is_long_castle: bool = False
    is_promotion_to: PieceType = PieceType.EMPTY
    checks: ColorType = ColorType.EMPTY
    checkmate: bool = False

    def __str__(self) -> str:
        """Returns the move as a standardized string."""
        if self.is_castle:
            return "O-O-O" if self.is_long_castle else "O-O"
        repr = str(self.piece.type) + self.start
        if self.is_capture:
            repr += "x"
        repr += self.end
        if self.is_promotion:
            repr += "=" + str(self.is_promotion_to)
        if self.checks != ColorType.EMPTY:
            repr += "#" if self.checkmate else "+"
        return repr

    @property
    def player(self) -> ColorType:
        """Returns the player who made the move."""
        return self.piece.color

    @property
    def start(self) -> str:
        """Returns the starting position of the move."""
        return self.piece.pos

    @property
    def is_promotion(self) -> bool:
        """Returns the type of the promoted piece."""
        return self.is_promotion_to != PieceType.EMPTY

    @property
    def is_capture(self) -> bool:
        """Returns the type of the captured piece."""
        return self.captures is not None

    @classmethod
    def to_checkmate(cls, move: "Move") -> "Move":
        """Returns the move as a checkmate."""
        return Move(
            move.piece,
            move.end,
            move.captures,
            move.is_castle,
            move.is_double_pawn_push,
            move.is_long_castle,
            move.is_promotion_to,
            move.checks,
            True,
        )

    @classmethod
    def from_str(
        cls, move_str: str, board: dict[str, Piece], current: ColorType
    ) -> "Move":
        if move_str.startswith("O-O"):  # TODO: add check while castling
            checks, checkmate = _parse_checks(move_str, current)
            is_long_castle = move_str.startswith("O-O-O")
            piece_state = PieceState.get(PieceType.KING, current)
            return Move(
                piece=Piece(state=piece_state, pos=""),
                end="",
                is_castle=True,
                is_long_castle=is_long_castle,
                checks=checks,
                checkmate=checkmate,
            )
        return Move._parse(move_str, board)

    @classmethod
    def _parse(cls, move_str: str, board: dict[str, Piece]) -> "Move":
        """Parses a move string."""
        piece_type = move_str[0]  # e.g. Q
        src_pos = move_str[1:3]
        src_piece = board[src_pos]
        if src_piece.empty:
            raise ValueError(f"No piece found at position: {src_pos}")
        if str(src_piece.type) != piece_type:
            raise ValueError(
                f"Invalid piece type on {src_pos}: {piece_type} (expected {src_piece.type})"
            )
        capture = "x" in move_str
        if capture:
            dest_pos = move_str[4:6]
            if board[dest_pos].empty:
                raise ValueError(f"No captured piece found at position: {dest_pos}")
        else:
            dest_pos = move_str[3:5]
        checks, checkmate = _parse_checks(move_str, src_piece.color)
        return Move(
            piece=src_piece,
            end=dest_pos,
            captures=board[dest_pos] if capture else None,
            checks=checks,
            checkmate=checkmate,
        )


def _parse_checks(move_str: str, color: ColorType) -> tuple[ColorType, bool]:
    """Parses the checks and checkmate from a move string."""
    checks = ColorType.EMPTY
    checkmate = False
    if "+" in move_str:
        checks = color
    if "#" in move_str:
        checks = color
        checkmate = True
    return checks, checkmate


def get_opp_color(color: ColorType) -> ColorType:
    """Returns the opposite color."""
    return ColorType.WHITE if color == ColorType.BLACK else ColorType.BLACK
