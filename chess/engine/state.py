from dataclasses import dataclass
import numpy as np
from chess.engine.piece import ColorType, Piece, PieceState, PieceType, Move
import math
import functools
import random

_T_POS = str | np.uint64 | tuple[int, int]


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

    # Move history
    moves: list[Move]
    turn: int = 1

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

    @property
    def last_move(self) -> Move | None:
        """Returns the last move played."""
        if not self.moves:
            return None
        return self.moves[-1]

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
            moves=[],
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
            moves=[],
        )

    def __getitem__(self, pos: _T_POS) -> Piece:
        """Returns the piece at a position."""
        pos = _any_to_uint64_pos(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for piece, bitboard in self._bitboards:
            if pos & bitboard:
                return Piece(piece, _uint64_to_str_pos(pos))
        return Piece(PieceState.EMPTY, _uint64_to_str_pos(pos))

    def __setitem__(self, pos: _T_POS, piece: Piece) -> None:
        """Sets the piece at a position."""
        pos = _any_to_uint64_pos(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for state, bitboard in self._bitboards:
            if pos & bitboard:
                self._set_state(state, bitboard & ~pos)
                return
            if piece.state == state:
                self._set_state(state, bitboard | pos)

    def apply_move(self, move: Move) -> "GameState":
        """Applies a move to the game state."""
        if not move.is_valid:
            return self
        self.moves.append(move)
        self[move.end] = self[move.start]
        self[move.start] = Piece(PieceState.EMPTY, move.start)
        self.current_player = self._switch_turn()
        return self

    def print(self) -> None:
        """Prints the board state."""
        print(f"Player: {self.current_player.name} / Last: {self.last_move}")
        # Move history
        print("> ", end="")
        for i, move in enumerate(self.moves):
            if i % 2 == 0:
                print(f"{i // 2 + 1}.", end=" ")
            print(move, end=" ")
        print()
        for row in range(8, 0, -1):
            print(row, end=" - ")
            for col in "abcdefgh":
                print(self[col + str(row)], end=" ")
            print()
        print("    A  B  C  D  E  F  G  H")
        print()

    def get_legal_moves(self) -> dict[str, Move]:
        """Returns all legal moves for the current player."""
        moves = {}
        for row in range(8):
            for col in range(8):
                piece = self[row, col]
                if piece.color != self.current_player:
                    continue
                if piece.type == PieceType.PAWN:
                    moves.update(Rules.get_pawn_moves(piece, (row, col), self))
        return moves

    def _get_move(self, uint_start: np.uint64, uint_end: np.uint64) -> Move:
        """Returns a move from a start and end position."""
        piece_src = self[uint_start]
        piece_dest = self[uint_end]
        start = _uint64_to_cartesian_pos(uint_start)
        end = _uint64_to_str_pos(uint_end)
        if piece_src.type == PieceType.EMPTY:
            return Move.invalid()
        if piece_src.color != self.current_player:
            return Move.invalid()
        if piece_dest.color == self.current_player:
            return Move.invalid()
        if piece_src.type == PieceType.PAWN:
            pawn_moves = Rules.get_pawn_moves(piece_src, start, self)
            if end not in pawn_moves:
                return Move.invalid()
            return pawn_moves[end]
        return Move.invalid()

    def _switch_turn(self) -> ColorType:
        """Switches the current player."""
        Rules.clear_cache()
        self.turn += self.current_player == ColorType.BLACK
        return (
            ColorType.WHITE
            if self.current_player == ColorType.BLACK
            else ColorType.BLACK
        )


class Rules:
    """Dynamic rules for chess moves."""

    _cache = {}

    @classmethod
    def clear_cache(cls):
        cls._cache = {}

    @staticmethod
    def get_pawn_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid, non-capture pawn moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        moves = {}
        direction = 1 if piece.color == ColorType.WHITE else -1

        # Move forward
        for step_size in (1, 2):
            front_square = (start[0], start[1] + step_size * direction)
            if step_size == 1 and front_square[1] < 0 or front_square[1] >= 8:
                return moves
            if step_size == 2 and (start[1] - direction) % 7:
                continue
            if state[front_square].empty:
                dest = _cartesian_to_str_pos(front_square)
                moves[dest] = Move(
                    player=piece.color,
                    piece=piece,
                    start=_cartesian_to_str_pos(start),
                    end=dest,
                    is_valid=True,
                    repr=dest,
                    is_double_pawn_push=step_size == 2,
                )

        # Diagonal capture
        for diagonal in (-1, 1):
            diag_square = (start[0] + diagonal, start[1] + direction)
            if diag_square[0] < 0 or diag_square[0] >= 8:
                continue
            if not state[diag_square].empty and state[diag_square].color != piece.color:
                dest = f"{_cartesian_to_str_pos(diag_square)}+"
                moves[dest] = Move(
                    player=piece.color,
                    piece=piece,
                    start=_cartesian_to_str_pos(start),
                    end=dest[:-1],
                    is_valid=True,
                    repr=dest,
                    is_capture=True,
                )

        # En passant
        last_move = state.last_move
        if (
            last_move is not None
            and last_move.is_double_pawn_push
            and last_move.end[1] == start[1]
            and abs(ord(last_move.end[0]) - start[0]) == 1
        ):
            dest = f"{last_move.end[0] + str(start[1] + direction)}+"
            moves[dest] = Move(
                player=piece.color,
                piece=piece,
                start=_cartesian_to_str_pos(start),
                end=dest[:-1],
                is_valid=True,
                repr=dest,
                is_capture=True,
            )

        # Promotion
        promotions = []
        for dest, move in moves.items():
            if move.end[1] in ("1", "8"):
                promotions.append(dest)

        for dest in promotions:
            del moves[dest]
            for promotion in (
                PieceType.QUEEN,
                PieceType.ROOK,
                PieceType.BISHOP,
                PieceType.KNIGHT,
            ):
                dest_ = f"{dest}={promotion.name[0]}"
                moves[dest_] = Move(
                    player=piece.color,
                    piece=piece,
                    start=_cartesian_to_str_pos(start),
                    end=dest,
                    is_valid=True,
                    repr=dest_,
                    is_promotion=True,
                    is_promotion_to=promotion,
                )

        Rules._cache[start] = moves
        return moves


# Position formalism conversion functions
@functools.cache
def _cartesian_to_str_pos(pos: tuple[int, int]) -> str:
    """Converts a cartesian position to a string position."""
    return chr(pos[0] + ord("a")) + str(pos[1] + 1)


@functools.cache
def _cartesian_to_uint64_pos(pos: tuple[int, int]) -> np.uint64:
    """Converts a cartesian position to a uint64 bitboard."""
    return np.uint64(1) << np.uint64(pos[1] * 8 + pos[0])


@functools.cache
def _str_to_cartesian_pos(pos: str) -> tuple[int, int]:
    """Converts a string position to a cartesian position."""
    return ord(pos[0]) - ord("a"), int(pos[1]) - 1


@functools.cache
def _str_to_uint64_pos(pos: str) -> np.uint64:
    """Converts a string position to a uint64 bitboard."""
    x, y = _str_to_cartesian_pos(pos)
    return np.uint64(1) << np.uint64(y * 8 + x)


@functools.cache
def _uint64_to_str_pos(pos: np.uint64) -> str:
    """Converts a uint64 bitboard to a string position."""
    return _cartesian_to_str_pos(_uint64_to_cartesian_pos(pos))


@functools.cache
def _uint64_to_cartesian_pos(pos: np.uint64) -> tuple[int, int]:
    """Converts a uint64 bitboard to a cartesian position."""
    bit_pos = int(math.log2(pos))
    return bit_pos % 8, bit_pos // 8


@functools.cache
def _any_to_uint64_pos(pos: _T_POS) -> np.uint64:
    """Converts any position to a uint64 bitboard."""
    if isinstance(pos, str):
        return _str_to_uint64_pos(pos)
    if isinstance(pos, tuple):
        return _cartesian_to_uint64_pos(pos)
    return pos


if __name__ == "__main__":
    board = GameState.initialize()
    board.print()
    # play a few moves
    for _ in range(100):
        moves = board.get_legal_moves()
        if not moves:
            print("No more moves!")
            break
        move = random.choice(list(moves.values()))
        board = board.apply_move(move)
        board.print()
