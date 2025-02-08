from dataclasses import dataclass
import numpy as np
from chess.engine.piece import ColorType, Piece, PieceState, PieceType

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
            raise ValueError(f"Invalid position: {pos}")
        if self.w_pawn & pos:
            return Piece(PieceState.PAWN_W, pos)
        if self.w_rook & pos:
            return Piece(PieceState.ROOK_W, pos)
        if self.w_knight & pos:
            return Piece(PieceState.KNIGHT_W, pos)
        if self.w_bishop & pos:
            return Piece(PieceState.BISHOP_W, pos)
        if self.w_queen & pos:
            return Piece(PieceState.QUEEN_W, pos)
        if self.w_king & pos:
            return Piece(PieceState.KING_W, pos)
        if self.b_pawn & pos:
            return Piece(PieceState.PAWN_B, pos)
        if self.b_rook & pos:
            return Piece(PieceState.ROOK_B, pos)
        if self.b_knight & pos:
            return Piece(PieceState.KNIGHT_B, pos)
        if self.b_bishop & pos:
            return Piece(PieceState.BISHOP_B, pos)
        if self.b_queen & pos:
            return Piece(PieceState.QUEEN_B, pos)
        if self.b_king & pos:
            return Piece(PieceState.KING_B, pos)
        return Piece(PieceState.EMPTY, pos)

    def __setitem__(self, pos: _T_POS, piece: Piece) -> None:
        """Sets the piece at a position."""
        if isinstance(pos, str):
            pos = _str_pos_to_uint64(pos)
        if piece.color == ColorType.WHITE:
            if piece.type == PieceType.PAWN:
                self.w_pawn |= pos
            elif piece.type == PieceType.ROOK:
                self.w_rook |= pos
            elif piece.type == PieceType.KNIGHT:
                self.w_knight |= pos
            elif piece.type == PieceType.BISHOP:
                self.w_bishop |= pos
            elif piece.type == PieceType.QUEEN:
                self.w_queen |= pos
            elif piece.type == PieceType.KING:
                self.w_king |= pos
        elif piece.color == ColorType.BLACK:
            if piece.type == PieceType.PAWN:
                self.b_pawn |= pos
            elif piece.type == PieceType.ROOK:
                self.b_rook |= pos
            elif piece.type == PieceType.KNIGHT:
                self.b_knight |= pos
            elif piece.type == PieceType.BISHOP:
                self.b_bishop |= pos
            elif piece.type == PieceType.QUEEN:
                self.b_queen |= pos
            elif piece.type == PieceType.KING:
                self.b_king |= pos
        elif piece.color is None:
            current = self[pos]
            if current.color == ColorType.WHITE:
                if current.type == PieceType.PAWN:
                    self.w_pawn &= ~pos
                elif current.type == PieceType.ROOK:
                    self.w_rook &= ~pos
                elif current.type == PieceType.KNIGHT:
                    self.w_knight &= ~pos
                elif current.type == PieceType.BISHOP:
                    self.w_bishop &= ~pos
                elif current.type == PieceType.QUEEN:
                    self.w_queen &= ~pos
                elif current.type == PieceType.KING:
                    self.w_king &= ~pos
            elif current.color == ColorType.BLACK:
                if current.type == PieceType.PAWN:
                    self.b_pawn &= ~pos
                elif current.type == PieceType.ROOK:
                    self.b_rook &= ~pos
                elif current.type == PieceType.KNIGHT:
                    self.b_knight &= ~pos
                elif current.type == PieceType.BISHOP:
                    self.b_bishop &= ~pos
                elif current.type == PieceType.QUEEN:
                    self.b_queen &= ~pos
                elif current.type == PieceType.KING:
                    self.b_king &= ~pos
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

    def apply_move(self, start: _T_POS, end: _T_POS) -> "GameState":
        """Applies a move to the game state."""
        new_state = self.copy()
        if isinstance(start, str):
            start = _str_pos_to_uint64(start)
        if isinstance(end, str):
            end = _str_pos_to_uint64(end)
        new_state[end] = new_state[start]
        new_state[start] = Piece(PieceState.EMPTY, start)
        return new_state

    def print(self) -> None:
        """Prints the board state."""
        for row in range(8, 0, -1):
            for col in "abcdefgh":
                print(self[col + str(row)], end=" ")
            print()


def _str_pos_to_uint64(pos: str) -> np.uint64:
    """Converts a string position to a uint64 bitboard."""
    return np.uint64(1) << ((np.uint64(pos[1]) - 1) * 8 + (ord(pos[0]) - ord("a")))  # type: ignore


if __name__ == "__main__":
    board = GameState.initialize()
    # play a few moves
    board = board.apply_move("e2", "e4")
    board = board.apply_move("e7", "e5")
    board.print()
