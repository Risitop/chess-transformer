import dataclasses
import multiprocessing as mp
import random

import chess
import numpy as np
import torch

_PIECE2IDX = {piece: index for index, piece in enumerate("PNBRQKpnbrqk")}
_MAX_MOVES = 256
_GEN_CHUNK_SIZE = 256
_MIN_CHUNKS_GEN = 8

# State encoding indices
ST_IDX_PIECE = 0
ST_IDX_SQUARE = 1
MT_IDX_CASTLE_B = 0
MT_IDX_CASTLE_W = 1
MT_IDX_TURN = 2
MT_IDX_MOVE = 3
MV_PROM = 65


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

    board_state: torch.Tensor
    board_metadata: torch.Tensor
    legal_moves: list[list[int]]
    is_checkmate: bool


class ChessDataloader:
    """A dataloader for chess data.

    This dataloader generates random chess boards and vectorizes them for input to the
    model. It also maintains a buffer of generated boards to reduce the overhead of
    generating new boards.

    Attributes
    ----------
    buffer_target_size : int
        The target size of the buffer. The dataloader will generate new boards in the
        background until the buffer reaches this size.

    n_jobs : int
        The number of worker processes to use for generating boards in the background.
        If set to -1, the number of worker processes will be set to the number of CPU
        cores.
    """

    def __init__(self, buffer_target_size: int = 2_000, n_jobs: int = 16):
        self._board_buffer = []
        self._buffer_target_size = buffer_target_size
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        self._generation_pool = mp.Pool(n_jobs)
        self._virtual_samples = 0

    def get_boards(self, batch_size: int) -> list[ChessState]:
        """Get a batch of chess states.

        Parameters
        ----------
        batch_size : int
            The number of chess states to return.
        """
        n_from_buffer = min(len(self._board_buffer), batch_size)
        states = self._board_buffer[:n_from_buffer]
        self._board_buffer = self._board_buffer[n_from_buffer:]
        if len(states) < batch_size:
            states.extend(generate_boards(batch_size - len(states)))
        if len(self._board_buffer) + self._virtual_samples < self._buffer_target_size:
            missing = self._buffer_target_size - len(self._board_buffer)
            chunks_to_gen = max(missing // _GEN_CHUNK_SIZE, _MIN_CHUNKS_GEN)
            self._background_refill(chunks_to_gen)
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

    def _background_refill(self, n_chunks: int):
        """Refill the buffer in the background."""
        for _ in range(n_chunks):
            self._generation_pool.apply_async(
                generate_boards, (_GEN_CHUNK_SIZE,), callback=self._add_to_buffer
            )
        self._virtual_samples += n_chunks * _GEN_CHUNK_SIZE

    def _add_to_buffer(self, chunk: list[ChessState]):
        """Add a chunk of generated boards to the buffer."""
        self._virtual_samples -= len(chunk)
        self._board_buffer.extend(chunk)


def generate_boards(n: int) -> list[ChessState]:
    states = []
    for _ in range(n):
        board = _generate_random_board()
        vboard, leg_moves = _vectorize_board(board)
        states.append(
            ChessState(
                board_state=vboard,
                board_metadata=_vectorize_metadata(board),
                legal_moves=leg_moves,
                is_checkmate=board.is_checkmate(),
            )
        )
    return states


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


def _vectorize_board(board: chess.Board) -> tuple[torch.Tensor, list[list[int]]]:
    """Vectorize a chess board state for input to the model."""
    board_map = board.piece_map()
    board_state = torch.zeros(len(board_map), 2, dtype=torch.long)
    legal_moves = {square: set() for square in range(65)}  # 64 + promotion
    for move in board.legal_moves:
        legal_moves[move.from_square].add(move.to_square)
    moves = []
    for idx, (square, piece) in enumerate(board_map.items()):
        board_state[idx, ST_IDX_PIECE] = _PIECE2IDX[piece.symbol()]
        board_state[idx, ST_IDX_SQUARE] = square
        moves.append(list(legal_moves[square]))
    return board_state, moves


def _vectorize_metadata(board: chess.Board) -> torch.Tensor:
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
