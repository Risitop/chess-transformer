import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer import dataloader as dl
from chessformer import logging, utils
from chessformer.model.mlp import MLP


@dataclasses.dataclass
class StateAction:
    """Dataclass for storing the state and action of a chess board."""

    action_prob: torch.Tensor
    player: bool


@dataclasses.dataclass
class TrainingMetrics:
    """Dataclass for storing the training metrics of a chessformer model."""

    illegal_prob: float


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
    reward_discount : float, optional
        Discount factor for rewards that are used to calculate the policy loss.
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
        # Internals
        self.all_moves = utils.load_moves()
        self.move2idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx2move = {idx: move for idx, move in enumerate(self.all_moves)}
        self.dataloader = dl.ChessDataloader()
        self.device = torch.device("cpu")

        # Model state embeddings
        self.emb_piece = nn.Embedding(13, dim_hidden, padding_idx=12)
        self.emb_pos = nn.Embedding(65, dim_hidden, padding_idx=64)
        self.emb_castle_b = nn.Embedding(2, dim_hidden)
        self.emb_castle_w = nn.Embedding(2, dim_hidden)
        self.emb_turn = nn.Embedding(2, dim_hidden)
        self.cls_token = nn.Parameter(torch.randn(dim_hidden))
        self.prv_move = nn.Parameter(torch.randn(dim_hidden))

        # Transformer architecture
        self.mha = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim_hidden,
                n_heads,
                dim_feedforward=4 * dim_hidden,
                dropout=dropout_rate,
                batch_first=True,
                bias=False,
            ),
            num_layers=n_layers,
        )
        self.move_decoder = MLP(
            dim_hidden,
            len(self.all_moves),
            n_hidden,
            4 * dim_hidden,
            dropout_rate,
        )
        self.apply(self._init_weights)

        nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Chessformer initialized with {nparams} trainable parameters.")

    def to(self, device: str | torch.device) -> "Chessformer":
        self.device = torch.device(device)
        return super().to(device)

    def forward(self, state: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model, state (B, T, 2) and metadata (B, M)."""
        B, T, _ = state.shape
        state = state.to(self.device)
        metadata = metadata.to(self.device)

        # Base embeddings
        state_emb = self.emb_piece(state[:, :, dl.ST_IDX_PIECE])  # (B, T, C)
        pos_emb = self.emb_pos(state[:, :, dl.ST_IDX_SQUARE])
        base_emb = state_emb + pos_emb
        cb_emb = self.emb_castle_b(metadata[:, dl.MT_IDX_CASTLE_B]).unsqueeze(1)
        cw_emb = self.emb_castle_w(metadata[:, dl.MT_IDX_CASTLE_W]).unsqueeze(1)
        trn_emb = self.emb_turn(metadata[:, dl.MT_IDX_TURN]).unsqueeze(1)
        mvs_emb = torch.stack(
            [
                self.emb_pos(metadata[:, idx]) + self.prv_move
                for idx in range(dl.MT_IDX_MOVE, len(metadata[0]))
            ],
            dim=1,
        )
        cls_emb = self.cls_token.repeat(B, 1).unsqueeze(1)
        full_emb = torch.cat(
            [cls_emb, base_emb, cb_emb, cw_emb, trn_emb, mvs_emb], dim=1
        )

        # Padding mask
        state_mask = state[:, :, 0] == 12
        metadata_mask = metadata == 64
        mask = torch.cat(
            [
                torch.zeros(B, 1, dtype=torch.bool, device=self.device),
                state_mask,
                metadata_mask,
            ],
            dim=1,
        )

        # Transformer
        trans_emb = self.mha(full_emb, src_key_padding_mask=mask)  # (B, T, C)
        moves_logits = self.move_decoder(trans_emb[:, 0])
        return moves_logits

    def step(self, states: list[dl.ChessState]) -> tuple[torch.Tensor, TrainingMetrics]:
        """Take a step in the games, return the legal loss and monitoring metrics."""
        batch_size = len(states)
        state, metadata = self.dataloader.collate_inputs(states)
        move_logits = self.forward(state, metadata)
        if len(move_logits.shape) == 1:
            move_logits = move_logits.unsqueeze(0)
        action_probs = F.softmax(move_logits, dim=-1)

        # Calculate move legality loss
        n_moves = len(self.all_moves)
        illegal_target = torch.zeros(
            (batch_size, n_moves), device=self.device, dtype=torch.float32
        )
        for idx, state in enumerate(states):
            legal_moves = state.legal_moves
            if not legal_moves:
                continue
            legal_idx = [self.move2idx[move] for move in legal_moves]
            illegal_target[idx, legal_idx] = 1.0 / len(legal_moves)

        prob_illegal = (action_probs * (illegal_target == 0)).sum(dim=1).mean()
        loss_legal = F.cross_entropy(action_probs, illegal_target)

        return (
            loss_legal,
            TrainingMetrics(illegal_prob=prob_illegal.item()),
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
