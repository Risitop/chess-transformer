from dataclasses import dataclass
import numpy as np
from chess.engine.piece import ColorType, Piece, PieceState, PieceType, Move
from chess.engine import utils
import random
from typing import Literal

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
    w_has_castled: bool = False
    b_has_castled: bool = False
    current_player: ColorType = ColorType.WHITE

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

    def copy(self) -> "GameState":
        """Returns a copy of the game state."""
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
            moves=self.moves.copy(),
            turn=self.turn,
            winner=self.winner,
            w_has_castled=self.w_has_castled,
            b_has_castled=self.b_has_castled,
            current_player=self.current_player,
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
            if piece.state == state:
                self._set_state(state, bitboard | pos)
            else:
                self._set_state(state, bitboard & ~pos)

    def print(self) -> None:
        """Prints the board state."""
        print(f"Player: {self.current_player.name} / Last: {self.last_move}")
        print("> ", end="")
        for i, move in enumerate(self.moves):
            if i > 0 and i % 8 == 0:
                print()
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

    @property
    def is_ended(self) -> bool:
        """Returns True if the game has ended."""
        return self.winner != ColorType.EMPTY

    def apply_move(self, move: Move) -> "GameState":
        """Applies a move to the game state (assumed valid!)."""
        self.moves.append(move)

        # Special cases
        if move.is_castle:
            _castle(self, move)
        elif move.is_promotion:
            self[move.end] = Piece(
                PieceState.get(move.is_promotion_to, move.player), move.end
            )
        else:
            self[move.end] = self[move.start]
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
                if piece.type == PieceType.ROOK or piece.type == PieceType.QUEEN:
                    moves.update(
                        Rules.get_rook_bishop_moves(piece, (row, col), self, "rook")
                    )
                if piece.type == PieceType.BISHOP or piece.type == PieceType.QUEEN:
                    moves.update(
                        Rules.get_rook_bishop_moves(piece, (row, col), self, "bishop")
                    )
                if piece.type == PieceType.KNIGHT:
                    moves.update(Rules.get_knight_moves(piece, (row, col), self))
                if piece.type == PieceType.KING:
                    moves.update(Rules.get_king_moves(piece, (row, col), self))

        valid_moves = self._filter_check_moves(moves)
        if not valid_moves:
            self.winner = _get_opp_color(self.current_player)
        return valid_moves

    def _switch_turn(self) -> ColorType:
        """Switches the current player."""
        Rules.clear_cache()
        self.turn += self.current_player == ColorType.BLACK
        return _get_opp_color(self.current_player)

    def _filter_check_moves(self, moves: dict[str, Move]) -> dict[str, Move]:
        """Filters out moves that put current player's king in check."""
        current_player = self.current_player
        new_moves = {}
        for move in moves.values():
            new_state = self.copy().apply_move(move)
            if _king_is_checked(new_state, new_state._get_king(current_player)):
                continue
            if _king_is_checked(
                new_state, new_state._get_king(_get_opp_color(current_player))
            ):
                new_move = Move(
                    piece=move.piece,
                    end=move.end,
                    captures=move.captures,
                    is_castle=move.is_castle,
                    is_double_pawn_push=move.is_double_pawn_push,
                    is_long_castle=move.is_long_castle,
                    is_promotion_to=move.is_promotion_to,
                    checks=_get_opp_color(current_player),
                )
                new_moves[str(new_move)] = new_move
            else:
                new_moves[str(move)] = move
        return new_moves

    def _get_king(self, color: ColorType) -> Piece:
        """Returns the king for a given color."""
        for row in range(8):
            for col in range(8):
                piece = self[row, col]
                if piece.color == color and piece.type == PieceType.KING:
                    return piece
        raise ValueError(f"King not found for color: {color}")


class Rules:
    """Dynamic rules for chess moves."""

    _cache = {}
    _ROOK_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
    _BISHOP_MOVES = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    _KNIGHT_MOVES = (
        (2, 1),
        (-2, 1),
        (2, -1),
        (-2, -1),
        (1, 2),
        (-1, 2),
        (1, -2),
        (-1, -2),
    )
    _KING_MOVES = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    )

    @classmethod
    def clear_cache(cls):
        """Clears the move cache."""
        cls._cache = {}

    @staticmethod
    def get_pawn_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid pawn moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        moves = []
        direction = 1 if piece.color == ColorType.WHITE else -1

        # Move forward
        for step_size in (1, 2):
            front_square = (start[0], start[1] + step_size * direction)
            if step_size == 1 and (front_square[1] < 0 or front_square[1] >= 8):
                return {}
            if step_size == 2 and (start[1] - direction) % 7:
                continue
            if state[front_square].empty:
                dest = utils.cartesian_to_str_pos(front_square)
                is_double = step_size == 2
                moves.append(Move(piece=piece, end=dest, is_double_pawn_push=is_double))

        # Diagonal capture
        for diagonal in (-1, 1):
            diag_square = (start[0] + diagonal, start[1] + direction)
            if diag_square[0] < 0 or diag_square[0] >= 8:
                continue
            if not state[diag_square].empty and state[diag_square].color != piece.color:
                dest = utils.cartesian_to_str_pos(diag_square)
                moves.append(Move(piece=piece, end=dest, captures=state[diag_square]))

        # En passant
        last_move = state.last_move
        if (
            last_move is not None
            and last_move.is_double_pawn_push
            and last_move.end[1] == start[1]
            and abs(ord(last_move.end[0]) - start[0]) == 1
        ):
            dest = last_move.end[0] + str(start[1] + direction)
            moves.append(Move(piece=piece, end=dest, captures=state[last_move.end]))

        # Promotion
        promotions_idx = []
        for idx in range(len(moves)):
            if moves[idx].end[1] in ("1", "8"):
                promotions_idx.append(idx)

        for idx in promotions_idx:
            move = moves[idx]
            for promotion in (
                PieceType.QUEEN,
                PieceType.ROOK,
                PieceType.BISHOP,
                PieceType.KNIGHT,
            ):
                moves.append(
                    Move(piece=move.piece, end=move.end, is_promotion_to=promotion)
                )

        result = {
            str(move): move
            for idx, move in enumerate(moves)
            if idx not in promotions_idx
        }
        Rules._cache[start] = result
        return result

    @staticmethod
    def get_rook_bishop_moves(
        piece: Piece,
        start: tuple[int, int],
        state: GameState,
        moveset: Literal["rook", "bishop"],
    ) -> dict[str, Move]:
        """Returns all valid rook and bishop moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        match moveset:
            case "rook":
                directions = Rules._ROOK_MOVES
            case "bishop":
                directions = Rules._BISHOP_MOVES
            case _:
                raise ValueError(f"Invalid moveset: {moveset}")

        moves = []
        for direction in directions:
            for step in range(1, 8):
                dest = (start[0] + direction[0] * step, start[1] + direction[1] * step)
                if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                    break
                dest_str = utils.cartesian_to_str_pos(dest)
                if not state[dest].empty:
                    if state[dest].color != piece.color:
                        moves.append(
                            Move(piece=piece, end=dest_str, captures=state[dest])
                        )
                    break
                moves.append(Move(piece=piece, end=dest_str))

        result = {str(move): move for move in moves}
        Rules._cache[start] = result
        return result

    @staticmethod
    def get_knight_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid knight moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        moves = []
        for direction in Rules._KNIGHT_MOVES:
            dest = (start[0] + direction[0], start[1] + direction[1])
            if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                continue
            dest_str = utils.cartesian_to_str_pos(dest)
            if not state[dest].empty:
                if state[dest].color != piece.color:
                    moves.append(Move(piece=piece, end=dest_str, captures=state[dest]))
                continue
            moves.append(Move(piece=piece, end=dest_str))

        result = {str(move): move for move in moves}
        Rules._cache[start] = result
        return result

    @staticmethod
    def get_king_moves(
        piece: Piece, start: tuple[int, int], state: GameState
    ) -> dict[str, Move]:
        """Returns all valid king moves from a position."""
        if start in Rules._cache:
            return Rules._cache[start]

        moves = []
        for direction in Rules._KING_MOVES:
            dest = (start[0] + direction[0], start[1] + direction[1])
            if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                continue
            dest_str = utils.cartesian_to_str_pos(dest)
            if not state[dest].empty:
                if state[dest].color != piece.color:
                    moves.append(Move(piece=piece, end=dest_str, captures=state[dest]))
                continue
            moves.append(Move(piece=piece, end=dest_str))

        # Castling
        short, long = _check_castling(state, piece.color)
        if short:
            moves.append(Move(piece=piece, end="", is_castle=True))
        if long:
            moves.append(Move(piece=piece, end="", is_castle=True, is_long_castle=True))

        result = {str(move): move for move in moves}
        Rules._cache[start] = result
        return result


def _check_castling(state: GameState, color: ColorType) -> tuple[bool, bool]:
    """Returns True if castling is possible for a given color."""
    short, long = False, False
    if state.current_player == ColorType.WHITE and not state.w_has_castled:
        # Short castle
        king = state["e1"].type == PieceType.KING
        rook = state["h1"].type == PieceType.ROOK
        empty1 = state["f1"].empty
        empty2 = state["g1"].empty
        short = king and rook and empty1 and empty2

        # Long castle
        king = state["e1"].type == PieceType.KING
        rook = state["a1"].type == PieceType.ROOK
        empty1 = state["b1"].empty
        empty2 = state["c1"].empty
        empty3 = state["d1"].empty
        long = king and rook and empty1 and empty2 and empty3

    if state.current_player == ColorType.BLACK and not state.b_has_castled:
        # Short castle
        king = state["e8"].type == PieceType.KING
        rook = state["h8"].type == PieceType.ROOK
        empty1 = state["f8"].empty
        empty2 = state["g8"].empty
        short = king and rook and empty1 and empty2

        # Long castle
        king = state["e8"].type == PieceType.KING
        rook = state["a8"].type == PieceType.ROOK
        empty1 = state["b8"].empty
        empty2 = state["c8"].empty
        empty3 = state["d8"].empty
        long = king and rook and empty1 and empty2 and empty3

    return short, long


def _castle(state: GameState, move: Move) -> None:
    """Performs a castling move."""
    if move.player == ColorType.WHITE:
        if move.is_long_castle:
            state["c1"] = state["e1"]
            state["d1"] = state["a1"]
            state["a1"] = Piece(PieceState.EMPTY, "e1")
            state["b1"] = Piece(PieceState.EMPTY, "a1")
            state["e1"] = Piece(PieceState.EMPTY, "e1")
        else:
            state["g1"] = state["e1"]
            state["f1"] = state["h1"]
            state["h1"] = Piece(PieceState.EMPTY, "e1")
            state["e1"] = Piece(PieceState.EMPTY, "e1")
        state.w_has_castled = True
    else:
        if move.is_long_castle:
            state["c8"] = state["e8"]
            state["d8"] = state["a8"]
            state["a8"] = Piece(PieceState.EMPTY, "e8")
            state["b8"] = Piece(PieceState.EMPTY, "a8")
            state["e8"] = Piece(PieceState.EMPTY, "e8")
        else:
            state["g8"] = state["e8"]
            state["f8"] = state["h8"]
            state["h8"] = Piece(PieceState.EMPTY, "e8")
            state["e8"] = Piece(PieceState.EMPTY, "e8")
        state.b_has_castled = True


def _king_is_checked(state: GameState, king: Piece) -> bool:
    """Check all directions around the king for direct checks."""
    king_pos = utils.str_to_cartesian_pos(king.pos)
    target_player = king.color
    # Look for queens, rooks, bishops
    for direction in Rules._ROOK_MOVES + Rules._BISHOP_MOVES:
        for step in range(1, 8):
            dest = (
                king_pos[0] + direction[0] * step,
                king_pos[1] + direction[1] * step,
            )
            if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
                break
            piece = state[dest]
            if not piece.empty:
                if piece.color == target_player:
                    break
                if piece.type == PieceType.QUEEN:
                    return True
                if piece.type == PieceType.ROOK and abs(sum(direction)) == 1:
                    return True
                if piece.type == PieceType.BISHOP and not sum(direction) % 2:
                    return True

    # Look for knights
    for direction in Rules._KNIGHT_MOVES:
        dest = (king_pos[0] + direction[0], king_pos[1] + direction[1])
        if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
            continue
        piece = state[dest]
        if (
            not piece.empty
            and piece.color != target_player
            and piece.type == PieceType.KNIGHT
        ):
            return True

    # Look for pawns
    direction = 1 if target_player == ColorType.WHITE else -1
    for diagonal in (-1, 1):
        dest = (king_pos[0] + diagonal, king_pos[1] + direction)
        if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
            continue
        piece = state[dest]
        if (
            not piece.empty
            and piece.color != target_player
            and piece.type == PieceType.PAWN
        ):
            return True

    # Look for kings
    for direction in Rules._KING_MOVES:
        dest = (king_pos[0] + direction[0], king_pos[1] + direction[1])
        if dest[0] < 0 or dest[0] >= 8 or dest[1] < 0 or dest[1] >= 8:
            continue
        piece = state[dest]
        if (
            not piece.empty
            and piece.color != target_player
            and piece.type == PieceType.KING
        ):
            return True

    return False


def _get_opp_color(color: ColorType) -> ColorType:
    """Returns the opposite color."""
    return ColorType.WHITE if color == ColorType.BLACK else ColorType.BLACK


if __name__ == "__main__":
    board = GameState.initialize()
    # play a few moves
    for _ in range(999):
        moves = board.get_legal_moves()
        if not moves:
            break
        move = random.choice(list(moves.values()))
        board = board.apply_move(move)
        if board.is_ended:
            break
    board.print()
    if not moves:
        print("No more moves!")
        if board.winner == ColorType.EMPTY:
            print("Stalemate!")
        else:
            print(f"King has fallen! Winner: {board.winner.name}")
