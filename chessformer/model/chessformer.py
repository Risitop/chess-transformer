import dataclasses
import logging
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer.model.mlp import MLP

_MOVES_PTH = Path(__file__).parent.parent.parent / "assets" / "all_moves.txt"


@dataclasses.dataclass
class BoardState:
    """Dataclass for storing the state of a chess board."""

    board: chess.Board
    state: torch.Tensor
    player: torch.Tensor


class Chessformer(nn.Module):
    """A transformer-based model for chess move prediction.

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers.
    dim_hidden : int
        Hidden layer size.
    n_layers : int
        Number of transformer layers.
    n_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate.
    """

    def __init__(
        self,
        n_hidden: int,
        dim_hidden: int,
        n_layers: int,
        n_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Constants
        self.all_moves = _load_moves()
        self.move2idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx2move = {idx: move for idx, move in enumerate(self.all_moves)}

        # Model
        self.emb_piece = nn.Embedding(13, dim_hidden)  # (64,) -> (64, C)
        self.emb_pos = nn.Embedding(64, dim_hidden)  # (64,) -> (64, C)
        self.emb_player = nn.Embedding(2, dim_hidden)  # (1,) -> (1, C)
        self.combiner = MLP(3 * dim_hidden, dim_hidden, 2, dim_hidden, dropout_rate)
        self.mha = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim_hidden,
                n_heads,
                dim_feedforward=dim_hidden,
                dropout=dropout_rate,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.decoder = MLP(dim_hidden, 1, 2, dim_hidden, dropout_rate)
        self.mlp = MLP(64, len(self.all_moves), n_hidden, dim_hidden, dropout_rate)

        nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Chessformer initialized with {nparams} trainable parameters.")

    def forward(self, board_state: BoardState) -> torch.Tensor:
        state, player = board_state.state, board_state.player
        pieces = self.emb_piece(state)  # (64, C)
        pos = self.emb_pos(torch.arange(64))  # (64, C)
        player = torch.tile(self.emb_player(player), (64, 1))  # (64, C)
        stacked = torch.stack([pieces, pos, player], dim=1).reshape(64, -1)
        x = self.combiner(stacked)  # (64, C)
        x = self.mha(x)
        x = self.decoder(x).squeeze()
        return self.mlp(x)

    def step(self, board: chess.Board) -> torch.Tensor:
        board_state = _vectorize_board(board)
        logits = self.forward(board_state)
        logits = F.softmax(logits, dim=-1)
        legal_moves = [move.uci() for move in board.legal_moves]
        target = torch.zeros(len(self.all_moves))
        for move in legal_moves:
            target[self.move2idx[move]] = 1.0 / len(legal_moves)
        loss = F.cross_entropy(logits, target)

        top_idx = torch.argmax(logits).item()
        top_move = self.idx2move[top_idx]  # type: ignore
        if top_move in legal_moves:
            board.push_uci(top_move)

        return loss


def _load_moves() -> list[str]:
    """Load all possible chess moves from file."""
    with open(_MOVES_PTH, "r") as f:
        return f.read().splitlines()


def _vectorize_board(board: chess.Board) -> BoardState:
    """Vectorize a chess board state for input to the model."""
    pieces = "PNBRQKpnbrqk"
    piece_map = {piece: 1 + index for index, piece in enumerate(pieces)}

    board_state = torch.zeros(8, 8, dtype=torch.long)
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, row))
            if piece is not None:
                board_state[row, col] = piece_map[piece.symbol()]
    return BoardState(
        board=board,
        state=board_state.flatten(),
        player=torch.tensor(board.turn, dtype=torch.long),
    )
