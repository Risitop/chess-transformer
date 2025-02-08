from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import NamedTuple

T_POS = tuple[int, int] | str


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
    pos: tuple[int, int]

    def __repr__(self) -> str:
        if self.empty:
            return ".."
        return "PNBRQK"[self.type.value] + "wb"[self.color.value]  # type: ignore

    @property
    def empty(self) -> bool:
        """Returns True if the piece is empty."""
        return self.type is None

    @property
    def coords(self) -> str:
        """Returns the piece's coordinates."""
        return _tuple_pos_to_str(self.pos)


class Move(NamedTuple):
    """Represents a chess move."""

    start: tuple[int, int]
    end: tuple[int, int]
    piece: Piece
    capture: Piece | None = None
    promotion: PieceType | None = None
    check: bool = False
    checkmate: bool = False
    en_passant: bool = False
    castling: bool = False

    def __repr__(self) -> str:
        return f"{self.piece} {self.start} -> {self.end}"


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

    def __getitem__(self, pos: T_POS) -> Piece:
        """Returns the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_tuple(pos)
        if not (0 <= pos[0] < 8 and 0 <= pos[1] < 8):
            raise IndexError(f"Position out of bounds: {pos}")
        mask = np.uint64(1) << (pos[0] * 8 + pos[1])
        if self.w_pawn & mask:
            return Piece(PieceType.PAWN, ColorType.WHITE, pos)
        if self.w_rook & mask:
            return Piece(PieceType.ROOK, ColorType.WHITE, pos)
        if self.w_knight & mask:
            return Piece(PieceType.KNIGHT, ColorType.WHITE, pos)
        if self.w_bishop & mask:
            return Piece(PieceType.BISHOP, ColorType.WHITE, pos)
        if self.w_queen & mask:
            return Piece(PieceType.QUEEN, ColorType.WHITE, pos)
        if self.w_king & mask:
            return Piece(PieceType.KING, ColorType.WHITE, pos)
        if self.b_pawn & mask:
            return Piece(PieceType.PAWN, ColorType.BLACK, pos)
        if self.b_rook & mask:
            return Piece(PieceType.ROOK, ColorType.BLACK, pos)
        if self.b_knight & mask:
            return Piece(PieceType.KNIGHT, ColorType.BLACK, pos)
        if self.b_bishop & mask:
            return Piece(PieceType.BISHOP, ColorType.BLACK, pos)
        if self.b_queen & mask:
            return Piece(PieceType.QUEEN, ColorType.BLACK, pos)
        if self.b_king & mask:
            return Piece(PieceType.KING, ColorType.BLACK, pos)
        return Piece(None, None, pos)

    def __setitem__(self, pos: T_POS, piece: Piece) -> None:
        """Sets the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_tuple(pos)
        mask = np.uint64(1) << (pos[0] * 8 + pos[1])
        if piece.color == ColorType.WHITE:
            if piece.type == PieceType.PAWN:
                self.w_pawn |= mask
            elif piece.type == PieceType.ROOK:
                self.w_rook |= mask
            elif piece.type == PieceType.KNIGHT:
                self.w_knight |= mask
            elif piece.type == PieceType.BISHOP:
                self.w_bishop |= mask
            elif piece.type == PieceType.QUEEN:
                self.w_queen |= mask
            elif piece.type == PieceType.KING:
                self.w_king |= mask
        elif piece.color == ColorType.BLACK:
            if piece.type == PieceType.PAWN:
                self.b_pawn |= mask
            elif piece.type == PieceType.ROOK:
                self.b_rook |= mask
            elif piece.type == PieceType.KNIGHT:
                self.b_knight |= mask
            elif piece.type == PieceType.BISHOP:
                self.b_bishop |= mask
            elif piece.type == PieceType.QUEEN:
                self.b_queen |= mask
            elif piece.type == PieceType.KING:
                self.b_king |= mask
        elif piece.color is None:  # Empty square
            current = self[pos]
            if current.color == ColorType.WHITE:
                if current.type == PieceType.PAWN:
                    self.w_pawn &= ~mask
                elif current.type == PieceType.ROOK:
                    self.w_rook &= ~mask
                elif current.type == PieceType.KNIGHT:
                    self.w_knight &= ~mask
                elif current.type == PieceType.BISHOP:
                    self.w_bishop &= ~mask
                elif current.type == PieceType.QUEEN:
                    self.w_queen &= ~mask
                elif current.type == PieceType.KING:
                    self.w_king &= ~mask
            elif current.color == ColorType.BLACK:
                if current.type == PieceType.PAWN:
                    self.b_pawn &= ~mask
                elif current.type == PieceType.ROOK:
                    self.b_rook &= ~mask
                elif current.type == PieceType.KNIGHT:
                    self.b_knight &= ~mask
                elif current.type == PieceType.BISHOP:
                    self.b_bishop &= ~mask
                elif current.type == PieceType.QUEEN:
                    self.b_queen &= ~mask
                elif current.type == PieceType.KING:
                    self.b_king &= ~mask
        else:
            raise ValueError(f"Invalid piece: {piece}")

    def copy(self) -> "GameState":
        """Copies the game state."""
        return GameState(
            w_pawn=self.w_pawn,
            w_rook=self.w_rook,
            w_knight=self.w_knight,
            w_bishop=self.w_bishop,
            w_queen=self.w_queen,
            w_king=self.w_king,
            b_pawn=self.b_pawn,
            b_rook=self.b_rook,
            b_knight=self.b_knight,
            b_bishop=self.b_bishop,
            b_queen=self.b_queen,
            b_king=self.b_king,
            has_castled=self.has_castled,
            current_player=self.current_player,
            in_check=self.in_check,
        )

    def apply_move(self, start: T_POS, end: T_POS) -> "GameState":
        """Applies a move to the game state."""
        new_state = self.copy()
        new_state[end] = new_state[start]
        if isinstance(start, str):
            start = _str_pos_to_tuple(start)
        new_state[start] = Piece(None, None, start)
        return new_state

    def print(self) -> None:
        """Prints the board state."""
        for row in range(8, 0, -1):
            for col in "abcdefgh":
                print(self[col + str(row)], end=" ")
            print()


def _str_pos_to_tuple(pos: str) -> tuple[int, int]:
    """Converts a string position to a tuple: 'a1' -> (0, 0) etc."""
    return int(pos[1]) - 1, ord(pos[0]) - ord("a")


def _tuple_pos_to_str(pos: tuple[int, int]) -> str:
    """Converts a tuple position to a string: (0, 0) -> 'a1' etc."""
    return chr(pos[1] + ord("a")) + str(pos[0] + 1)


if __name__ == "__main__":
    board = GameState.initialize()
    # Play a few classic moves
    board = board.apply_move("e2", "e4")
    board = board.apply_move("e7", "e5")
    board = board.apply_move("g1", "f3")
    board = board.apply_move("b8", "c6")
    board.print()
