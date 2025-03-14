import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer import dataloader as dl
from chessformer import logging, utils


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
    n_jobs : int, optional
        Number of parallel jobs to use for data generation.
    """

    def __init__(
        self,
        n_hidden: int,
        dim_hidden: int,
        n_layers: int,
        n_heads: int,
        dropout_rate: float,
        n_jobs: int,
    ):
        super().__init__()
        # Internals
        self.all_moves = utils.load_moves()
        self.move2idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx2move = {idx: move for idx, move in enumerate(self.all_moves)}
        self.dataloader = dl.ChessDataloader(n_jobs=n_jobs)
        self.device = torch.device("cpu")

        # Model state embeddings
        self.emb_piece = nn.Embedding(13, dim_hidden, padding_idx=12)
        self.emb_pos = nn.Embedding(64, dim_hidden)
        self.emb_castle_b = nn.Embedding(2, dim_hidden)
        self.emb_castle_w = nn.Embedding(2, dim_hidden)
        self.emb_turn = nn.Embedding(2, dim_hidden)
        self.prv_move = nn.Parameter(torch.zeros(dim_hidden))

        # Transformer architecture
        self.mha = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim_hidden,
                n_heads,
                dim_feedforward=4 * dim_hidden,
                dropout=dropout_rate,
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=n_layers,
        )
        self.move_decoder = nn.Linear(dim_hidden, 64, bias=False)
        self.move_decoder.weight = self.emb_pos.weight

        self.apply(self._init_weights)

        nparams = utils.bigint_to_str(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )
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

        full_emb = torch.cat([base_emb, cb_emb, cw_emb, trn_emb, mvs_emb], dim=1)

        # Padding mask
        state_mask = state[:, :, 0] == 12
        metadata_mask = torch.zeros(
            B, metadata.shape[1], device=self.device, dtype=torch.bool
        )
        mask = torch.cat([state_mask, metadata_mask], dim=1)

        # Transformer
        trans_emb = self.mha(full_emb, src_key_padding_mask=mask)  # (B, T, C)
        trans_emb = trans_emb[:, : -metadata.shape[1], :]  # Keep only pieces
        return self.move_decoder(trans_emb)

    def step(self, states: list[dl.ChessState]) -> tuple[torch.Tensor, TrainingMetrics]:
        """Take a step in the games, return the legal loss and monitoring metrics."""
        state, metadata = self.dataloader.collate_inputs(states)
        B, T, _ = state.shape
        move_logits = self.forward(state, metadata)
        if len(move_logits.shape) == 1:
            move_logits = move_logits.unsqueeze(0)

        # Calculate move legality loss
        illegal_target = torch.zeros(
            (B, T, 64),
            device=self.device,
            dtype=torch.float32,
        )
        for idx, state in enumerate(states):
            for pidx, moveset in enumerate(state.legal_moves):
                if not moveset:
                    continue
                illegal_target[idx, pidx, moveset] = 1 / len(moveset)

        loss_legal = F.cross_entropy(
            move_logits.transpose(1, 2), illegal_target.transpose(1, 2)
        )
        action_probs = F.softmax(move_logits, dim=-1)
        prob_illegal = (action_probs * (illegal_target == 0)).sum(dim=-1).mean()

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
