from dataclasses import dataclass
import numpy as np
from chess.engine.piece import ColorType, Piece, PieceState, PieceType
from typing import Iterator

_T_POS = str | np.uint64


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

    @property
    def _bitboards(self) -> set[tuple[PieceState, np.uint64]]:
        """Iterates over all bitboards."""
        return {
            (PieceState.PAWN_W, self.w_pawn),
            (PieceState.ROOK_W, self.w_rook),
            (PieceState.KNIGHT_W, self.w_knight),
            (PieceState.BISHOP_W, self.w_bishop),
            (PieceState.QUEEN_W, self.w_queen),
            (PieceState.KING_W, self.w_king),
            (PieceState.PAWN_B, self.b_pawn),
            (PieceState.ROOK_B, self.b_rook),
            (PieceState.KNIGHT_B, self.b_knight),
            (PieceState.BISHOP_B, self.b_bishop),
            (PieceState.QUEEN_B, self.b_queen),
            (PieceState.KING_B, self.b_king),
        }

    def _set_state(self, state: PieceState, bitboard: np.uint64) -> None:
        """Sets a bitboard for a piece state."""
        if state == PieceState.PAWN_W:
            self.w_pawn = bitboard
        elif state == PieceState.ROOK_W:
            self.w_rook = bitboard
        elif state == PieceState.KNIGHT_W:
            self.w_knight = bitboard
        elif state == PieceState.BISHOP_W:
            self.w_bishop = bitboard
        elif state == PieceState.QUEEN_W:
            self.w_queen = bitboard
        elif state == PieceState.KING_W:
            self.w_king = bitboard
        elif state == PieceState.PAWN_B:
            self.b_pawn = bitboard
        elif state == PieceState.ROOK_B:
            self.b_rook = bitboard
        elif state == PieceState.KNIGHT_B:
            self.b_knight = bitboard
        elif state == PieceState.BISHOP_B:
            self.b_bishop = bitboard
        elif state == PieceState.QUEEN_B:
            self.b_queen = bitboard
        elif state == PieceState.KING_B:
            self.b_king = bitboard

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

    def __getitem__(self, pos: _T_POS) -> Piece:
        """Returns the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_uint64(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for piece, bitboard in self._bitboards:
            if pos & bitboard:
                return Piece(piece, pos)
        return Piece(PieceState.EMPTY, pos)

    def __setitem__(self, pos: _T_POS, piece: Piece) -> None:
        """Sets the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_uint64(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for state, bitboard in self._bitboards:
            if piece.type == PieceType.EMPTY and pos & bitboard:
                self._set_state(state, bitboard & ~pos)
                return
            if piece.state == state:
                self._set_state(state, bitboard | pos)

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

    def apply_move(self, start: _T_POS, end: _T_POS) -> "GameState":
        """Applies a move to the game state."""
        if isinstance(start, str):
            start = _str_pos_to_uint64(start)
        if isinstance(end, str):
            end = _str_pos_to_uint64(end)
        if not self._is_valid_move(start, end):
            return self
        new_state = self.copy()
        new_state[end] = new_state[start]
        new_state[start] = Piece(PieceState.EMPTY, start)
        return new_state

    def print(self) -> None:
        """Prints the board state."""
        for row in range(8, 0, -1):
            for col in "abcdefgh":
                print(self[col + str(row)], end=" ")
            print()

    def _is_valid_move(self, start: np.uint64, end: np.uint64) -> bool:
        """Checks if a move is valid."""
        return True


def _str_pos_to_uint64(pos: str) -> np.uint64:
    """Converts a string position to a uint64 bitboard."""
    return np.uint64(1) << ((np.uint64(pos[1]) - 1) * 8 + (ord(pos[0]) - ord("a")))  # type: ignore


if __name__ == "__main__":
    board = GameState.initialize()
    # play a few moves
    board = board.apply_move("e2", "e4")
    board = board.apply_move("e7", "e5")
    board.print()
