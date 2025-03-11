import dataclasses
import random

import chess
import numpy as np
import torch

chess.Move

_PIECE2IDX = {piece: index for index, piece in enumerate("PNBRQKpnbrqk")}
_MAX_MOVES = 256

# State encoding indices
ST_IDX_PIECE = 0
ST_IDX_SQUARE = 1
MT_IDX_CASTLE_B = 0
MT_IDX_CASTLE_W = 1
MT_IDX_TURN = 2
MT_IDX_MOVE = 3


@dataclasses.dataclass
class ChessState:
    """A preprocessed chess state.

    Attributes
    ----------
    board : chess.Board
        The chess board, for reference. May be turned off in the future.

    board_state : torch.Tensor
        The vectorized board state, of shape (T, 2) where T is the number of pieces
        on the board and 2 is the piece type and square index.

    board_metadata : torch.Tensor
        Metadata about the board, including castling rights and recent moves.

    legal_moves : list[str]
        A list of legal moves in UCI format.

    is_checkmate : bool
        Whether the current board state is a checkmate.
    """

    board: chess.Board
    board_state: torch.Tensor
    board_metadata: torch.Tensor
    legal_moves: list[str]
    is_checkmate: bool


class ChessDataloader:
    def __init__(self):
        pass

    def get_boards(self, batch_size: int) -> list[ChessState]:
        """Get a batch of chess states.

        Parameters
        ----------
        batch_size : int
            The number of chess states to return.
        """
        states = []
        for _ in range(batch_size):
            board = _generate_random_board()
            states.append(
                ChessState(
                    board=board,
                    board_state=_vectorize_board(board),
                    board_metadata=self._vectorize_metadata(board),
                    legal_moves=[move.uci() for move in board.legal_moves],
                    is_checkmate=board.is_checkmate(),
                )
            )
        return states

    def collate_inputs(
        self, states: list[ChessState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate a list of chess states into a batch."""
        batch_size = len(states)
        max_pieces = max(state.board_state.size(0) for state in states)
        board_state = torch.zeros(batch_size, max_pieces, 2, dtype=torch.long)
        board_state[:, :, ST_IDX_PIECE] = 12  # pad tokens
        board_state[:, :, ST_IDX_SQUARE] = 64
        metadata = torch.full((batch_size, 8), 64, dtype=torch.long)
        for idx, state in enumerate(states):
            board_state[idx, : state.board_state.size(0)] = state.board_state
            metadata[idx] = state.board_metadata
        return board_state, metadata

    def _vectorize_metadata(self, board: chess.Board) -> torch.Tensor:
        """Vectorize metadata about the chess board."""
        metadata = torch.full((8,), 64, dtype=torch.long)
        metadata[MT_IDX_CASTLE_B] = board.has_castling_rights(chess.BLACK)
        metadata[MT_IDX_CASTLE_W] = board.has_castling_rights(chess.WHITE)
        metadata[MT_IDX_TURN] = board.turn
        for idx, move in enumerate(reversed(board.move_stack)):
            metadata[MT_IDX_MOVE + idx] = move.to_square
            if idx >= 4:
                break
        return metadata


def _generate_random_board() -> chess.Board:
    """Generate a random chess board."""
    board = chess.Board()
    n_moves = np.random.randint(0, _MAX_MOVES)
    for _ in range(n_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
    return board


def _vectorize_board(board: chess.Board) -> torch.Tensor:
    """Vectorize a chess board state for input to the model."""
    board_map = board.piece_map()
    board_state = torch.zeros(len(board_map), 2, dtype=torch.long)
    for idx, (square, piece) in enumerate(board_map.items()):
        board_state[idx, ST_IDX_PIECE] = _PIECE2IDX[piece.symbol()]
        board_state[idx, ST_IDX_SQUARE] = square
    return board_state
