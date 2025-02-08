from dataclasses import dataclass
import numpy as np
from chess.engine.piece import ColorType, Piece, PieceState, PieceType, Move
from chess.engine import utils
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
    winner: ColorType = ColorType.EMPTY

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
            w_knight=np.uint64(0x0000000000000024),
            w_bishop=np.uint64(0x0000000000000042),
            w_queen=np.uint64(0x0000000000000008),
            w_king=np.uint64(0x0000000000000010),
            b_pawn=np.uint64(0x00FF000000000000),
            b_rook=np.uint64(0x8100000000000000),
            b_knight=np.uint64(0x2400000000000000),
            b_bishop=np.uint64(0x4200000000000000),
            b_queen=np.uint64(0x0800000000000000),
            b_king=np.uint64(0x1000000000000000),
            moves=[],
        )

    def __getitem__(self, pos: _T_POS) -> Piece:
        """Returns the piece at a position."""
        pos = utils.any_to_uint64_pos(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for piece, bitboard in self._bitboards:
            if pos & bitboard:
                return Piece(piece, utils.uint64_to_str_pos(pos))
        return Piece(PieceState.EMPTY, utils.uint64_to_str_pos(pos))

    def __setitem__(self, pos: _T_POS, piece: Piece) -> None:
        """Sets the piece at a position."""
        pos = utils.any_to_uint64_pos(pos)
        if pos.bit_count() != 1:
            raise IndexError(f"Invalid position: {pos}")
        for state, bitboard in self._bitboards:
            if pos & bitboard:
                self._set_state(state, bitboard & ~pos)
            if piece.state == state:
                self._set_state(state, bitboard | pos)

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

    def apply_move(self, move: Move) -> "GameState":
        """Applies a move to the game state."""
        if not move.is_valid:
            return self
        self.moves.append(move)
        if move.is_capture and move.is_capture_type == PieceType.KING:
            self.winner = move.player
        if not move.is_promotion:
            self[move.end] = self[move.start]
        else:
            if move.is_promotion_to is None:
                raise RuntimeError("Promotion move without promotion type.")
            self[move.end] = Piece(
                PieceState.get(move.is_promotion_to, move.player),
                move.end,
            )
        self[move.start] = Piece(PieceState.EMPTY, move.start)
        self.current_player = self._switch_turn()
        return self

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
                if piece.type == PieceType.ROOK:
                    moves.update(Rules.get_rook_moves(piece, (row, col), self))
                if piece.type == PieceType.BISHOP:
                    moves.update(Rules.get_bishop_moves(piece, (row, col), self))
                if piece.type == PieceType.QUEEN:
                    moves.update(Rules.get_rook_moves(piece, (row, col), self))
                    moves.update(Rules.get_bishop_moves(piece, (row, col), self))
                if piece.type == PieceType.KNIGHT:
                    moves.update(Rules.get_knight_moves(piece, (row, col), self))
                if piece.type == PieceType.KING:
                    moves.update(Rules.get_king_moves(piece, (row, col), self))
        return moves

    @property
    def ended(self) -> bool:
        """Returns True if the game has ended."""
        return self.winner != ColorType.EMPTY

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
            if step_size == 1 and (front_square[1] < 0 or front_square[1] >= 8):
                return moves
            if step_size == 2 and (start[1] - direction) % 7:
                continue
            if state[front_square].empty:
                dest = utils.cartesian_to_str_pos(front_square)
                moves[dest] = Move(
                    player=piece.color,
                    piece=piece,
                    start=utils.cartesian_to_str_pos(start),
                    end=dest,
                    is_valid=True,
                    is_double_pawn_push=step_size == 2,
                )

        # Diagonal capture
        for diagonal in (-1, 1):
            diag_square = (start[0] + diagonal, start[1] + direction)
            if diag_square[0] < 0 or diag_square[0] >= 8:
                continue
            if not state[diag_square].empty and state[diag_square].color != piece.color:
                dest = f"{utils.cartesian_to_str_pos(diag_square)}+"
                moves[dest] = Move(
                    player=piece.color,
                    piece=piece,
                    start=utils.cartesian_to_str_pos(start),
                    end=dest[:-1],
                    is_valid=True,
                    is_capture=True,
                    is_capture_type_=state[diag_square].type,
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
                start=utils.cartesian_to_str_pos(start),
                end=dest[:-1],
                is_valid=True,
                is_capture=True,
                is_capture_type_=PieceType.PAWN,
            )

        # Promotion
        promotions = []
        for dest, move in moves.items():
            if move.end[1] in ("1", "8"):
                promotions.append((dest, move))

        for dest, move in promotions:
            del moves[dest]
            for promotion in (
                PieceType.QUEEN,
                PieceType.ROOK,
                PieceType.BISHOP,
                PieceType.KNIGHT,
            ):
                dest_ = f"{dest}={str(promotion)[0]}"
                moves[dest_] = Move(
                    player=move.player,
                    piece=move.piece,
                    start=move.start,
                    end=move.end,
                    is_valid=True,
                    is_promotion=True,
                    is_promotion_to=promotion,
                    is_capture=move.is_capture,
                    is_capture_type_=move.is_capture_type_,
                )

        Rules._cache[start] = moves
        return moves

    @staticmethod
    def get_rook_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid, non-capture rook moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        strart_str = utils.cartesian_to_str_pos(start)
        moves = {}
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, 8):
                dest = (start[0] + direction[0] * step, start[1] + direction[1] * step)
                if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                    break
                dest_str = utils.cartesian_to_str_pos(dest)
                if not state[dest].empty:
                    if state[dest].color != piece.color:
                        dest_ = f"R{dest_str}+"
                        moves[dest_] = Move(
                            player=piece.color,
                            piece=piece,
                            start=strart_str,
                            end=dest_str,
                            is_valid=True,
                            is_capture=True,
                            is_capture_type_=state[dest].type,
                        )
                    break
                moves[dest_str] = Move(
                    player=piece.color,
                    piece=piece,
                    start=strart_str,
                    end=dest_str,
                    is_valid=True,
                )

        Rules._cache[start] = moves
        return moves

    @staticmethod
    def get_bishop_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid, non-capture bishop moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        strart_str = utils.cartesian_to_str_pos(start)
        moves = {}
        for direction in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            for step in range(1, 8):
                dest = (start[0] + direction[0] * step, start[1] + direction[1] * step)
                if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                    break
                dest_str = utils.cartesian_to_str_pos(dest)
                if not state[dest].empty:
                    if state[dest].color != piece.color:
                        dest_ = f"{dest_str}+"
                        moves[dest_] = Move(
                            player=piece.color,
                            piece=piece,
                            start=strart_str,
                            end=dest_str,
                            is_valid=True,
                            is_capture=True,
                            is_capture_type_=state[dest].type,
                        )
                    break
                moves[dest_str] = Move(
                    player=piece.color,
                    piece=piece,
                    start=strart_str,
                    end=dest_str,
                    is_valid=True,
                )

        Rules._cache[start] = moves
        return moves

    @staticmethod
    def get_knight_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid, non-capture knight moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        strart_str = utils.cartesian_to_str_pos(start)
        moves = {}
        for direction in (
            (2, 1),
            (-2, 1),
            (2, -1),
            (-2, -1),
            (1, 2),
            (-1, 2),
            (1, -2),
            (-1, -2),
        ):
            dest = (start[0] + direction[0], start[1] + direction[1])
            if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                continue
            dest_str = utils.cartesian_to_str_pos(dest)
            if not state[dest].empty:
                if state[dest].color != piece.color:
                    dest_ = f"{dest_str}+"
                    moves[dest_] = Move(
                        player=piece.color,
                        piece=piece,
                        start=strart_str,
                        end=dest_str,
                        is_valid=True,
                        is_capture=True,
                        is_capture_type_=state[dest].type,
                    )
                continue
            moves[dest_str] = Move(
                player=piece.color,
                piece=piece,
                start=strart_str,
                end=dest_str,
                is_valid=True,
            )

        Rules._cache[start] = moves
        return moves

    @staticmethod
    def get_king_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid, non-capture king moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        strart_str = utils.cartesian_to_str_pos(start)
        moves = {}
        for direction in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ):
            dest = (start[0] + direction[0], start[1] + direction[1])
            if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                continue
            dest_str = utils.cartesian_to_str_pos(dest)
            if not state[dest].empty:
                if state[dest].color != piece.color:
                    dest_ = f"{dest_str}+"
                    moves[dest_] = Move(
                        player=piece.color,
                        piece=piece,
                        start=strart_str,
                        end=dest_str,
                        is_valid=True,
                        is_capture=True,
                        is_capture_type_=state[dest].type,
                    )
                continue
            moves[dest_str] = Move(
                player=piece.color,
                piece=piece,
                start=strart_str,
                end=dest_str,
                is_valid=True,
            )

        Rules._cache[start] = moves
        return moves


if __name__ == "__main__":
    board = GameState.initialize()
    board.print()
    # play a few moves
    for _ in range(999):
        moves = board.get_legal_moves()
        if not moves:
            break
        move = random.choice(list(moves.values()))
        board = board.apply_move(move)
        board.print()
        if board.ended:
            break
    if not moves:
        print("No more moves!")
    if board.winner != ColorType.EMPTY:
        print(f"King has fallen! Winner: {board.winner.name}")
