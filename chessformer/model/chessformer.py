import dataclasses
import random
import time
from pathlib import Path
from typing import Literal

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer import logging, utils
from chessformer.model.mlp import MLP

_MOVES_PTH = Path(__file__).parent.parent.parent / "assets" / "all_moves.txt"
_GAMES_PTH = Path(__file__).parent.parent.parent / "out"
_PIECE2IDX = {piece: 1 + index for index, piece in enumerate("PNBRQKpnbrqk")}
_INPUT_SIZE = 67  # 64 squares + player + castling rights


@dataclasses.dataclass
class StateAction:
    """Dataclass for storing the state and action of a chess board."""

    action_prob: torch.Tensor
    player: bool


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
        self.all_moves = _load_moves()
        self.move2idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx2move = {idx: move for idx, move in enumerate(self.all_moves)}
        self.games_path = _GAMES_PTH / time.strftime("%Y-%m-%d_%H-%M-%S.games")
        self.device = torch.device("cpu")

        # Model
        self.emb_piece = nn.Embedding(13, dim_hidden)
        self.emb_pos = nn.Embedding(_INPUT_SIZE, dim_hidden)
        self.combiner = MLP(
            2 * dim_hidden, dim_hidden, n_hidden, dim_hidden, dropout_rate
        )
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
        self.decoder = MLP(
            _INPUT_SIZE * dim_hidden,
            len(self.all_moves),
            n_hidden,
            dim_hidden,
            dropout_rate,
        )

        nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Chessformer initialized with {nparams} trainable parameters.")

    def to(self, device: str | torch.device) -> "Chessformer":
        self.device = torch.device(device)
        return super().to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, T = state.shape
        pos = torch.tile(torch.arange(_INPUT_SIZE, device=self.device), (B, 1))
        state_emb = self.emb_piece(state)
        pos_emb = self.emb_pos(pos)
        stacked = torch.cat([state_emb, pos_emb], dim=-1)
        x = self.combiner(stacked)  # (130, C) -> (_INPUT_SIZE, C)
        x = self.mha(x)  # (B, _INPUT_SIZE, C)
        return self.decoder(x.reshape(B, -1))  # Decode the player token

    def step(self, boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
        """Take a step in the games, return the legal loss and illegal proba."""
        batch_size = len(boards)
        state = _vectorize_boards(boards, self.device)
        logits = self.forward(state)  # (B, _INPUT_SIZE) -> (B, O)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        action_probs = F.softmax(logits, dim=-1)

        # Calculate move legality loss
        n_moves = len(self.all_moves)
        illegal_target = torch.zeros(
            (batch_size, n_moves), device=self.device, dtype=torch.float32
        )
        for idx, board in enumerate(boards):
            legal_moves = [move.uci() for move in board.legal_moves]
            if len(legal_moves) == 0:
                raise ValueError("No legal moves available.")
            legal_idx = [self.move2idx[move] for move in legal_moves]
            illegal_target[idx, legal_idx] = 1.0 / len(legal_moves)
            chosen_move = random.choice(legal_moves)
            board.push_uci(chosen_move)

        prob_illegal = (action_probs * (illegal_target == 0)).sum(dim=1).mean()
        loss_legal = F.cross_entropy(action_probs, illegal_target)

        return loss_legal, prob_illegal

    def train(
        self,
        mode: Literal["pretrain", "policy"],
        n_games: int,
        batch_size: int = 4,
        moves_per_game: int = 256,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.999,
        learning_rate_min: float = 1e-6,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        checkmate_reward: float = 100.0,
        reward_discount: float = 0.99,
    ) -> "Chessformer":
        """Train the model using self-play.

        Parameters
        ----------
        mode : Literal["pretrain", "policy"]
            Training mode.
            - "pretrain": Train the model to predict legal moves.
            - "policy": Train the model to predict the best move.
        n_games : int
            Number of games to play.
        batch_size : int, optional
            Number of games to play in parallel.
        moves_per_game : int, optional
            Number of moves per game.
        learning_rate : float, optional
            Initial learning rate.
        learning_rate_decay : float, optional
            Learning rate decay factor.
        learning_rate_min : float, optional
            Minimum learning rate.
        weight_decay : float, optional
            L2 regularization strength.
        gradient_clip : float, optional
            Gradient clipping threshold.
        log_every_n : int, optional
            Log a full game every n games.
        checkmate_reward : float, optional
            Reward for a checkmate.
        reward_discount : float, optional
            Discount factor for rewards that are used to calculate the policy loss.
        """
        if not _GAMES_PTH.exists():
            _GAMES_PTH.mkdir()
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        logging.info(f"Training Chessformer for {n_games} games. Mode: `{mode}`")

        max_moves_total = batch_size * moves_per_game
        for game_n in range(0, n_games, batch_size):
            tstart = time.time()

            batch_size = min(batch_size, n_games - game_n)
            boards = [chess.Board() for _ in range(batch_size)]
            is_active = [True] * batch_size
            total_loss = 0.0
            illegal_prob = 0.0
            finished_games = 0
            moves_left = max_moves_total

            # Play the games in parallel until all are over
            for move_k in range(moves_per_game):
                active_idx = [idx for idx, active in enumerate(is_active) if active]
                active_boards = [boards[idx] for idx in active_idx]
                with torch.amp.autocast(  # type: ignore
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda",
                ):
                    loss, probl = self.step(active_boards)
                    total_loss += loss
                    illegal_prob += probl

                if mode == "pretrain":
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                    optimizer.step()
                else:
                    raise NotImplementedError

                for idx, board in zip(active_idx, active_boards):
                    if board.is_game_over():
                        finished_games += 1
                        is_active[idx] = False
                        moves_left -= moves_per_game - move_k
                    else:
                        moves_left -= 1

                pct_finished = 100 * (1 - moves_left / max_moves_total)
                print(
                    f"[ Games {game_n}-{game_n + batch_size}/{n_games}\t"
                    f"{pct_finished:5.1f}% ]",
                    end="\r",
                )

                if not any(is_active):
                    break

            # Learning rate decay
            learning_rate = max(learning_rate * learning_rate_decay, learning_rate_min)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            total_loss /= moves_per_game
            illegal_prob /= moves_per_game

            # Monitoring
            with open(self.games_path, "a") as f:
                f.write(utils.board_to_pgn(boards[0]) + "\n")

            tend = time.time()

            message = (
                f"[ Games {game_n}-{game_n + batch_size}/{n_games}\t100.0% ] "
                f"Finished: {100 * finished_games / batch_size:5.1f}% / "
                f"Loss: {total_loss.item():.3f} / "  # type: ignore
                f"P(illegal): {illegal_prob.item():.3f} / "  # type: ignore
                f"LR: {learning_rate:.2e} / "
                f"{(tend - tstart) / batch_size:.2f}s/game / "
                f"ETA: {((n_games - game_n) / batch_size) * (tend - tstart):.2f}s"
            )
            print(message, end=" " * 20 + "\r")
            logging.debug(message)

        return self


def _load_moves() -> list[str]:
    """Load all possible chess moves from file."""
    with open(_MOVES_PTH, "r") as f:
        return f.read().splitlines()


def _vectorize_boards(boards: list[chess.Board], device: torch.device) -> torch.Tensor:
    """Vectorize a chess board state for input to the model."""
    # 64 + player and if each player has castled
    board_state = torch.zeros(len(boards), _INPUT_SIZE, dtype=torch.long)
    for idx, board in enumerate(boards):
        board_map = board.piece_map()
        for square, piece in board_map.items():
            board_state[idx, square] = _PIECE2IDX[piece.symbol()]
        board_state[idx, -3] = board.has_castling_rights(chess.BLACK)
        board_state[idx, -2] = board.has_castling_rights(chess.WHITE)
    board_state[:, -1] = board.turn
    return board_state.to(device)


def _get_total_policy_loss(
    history: list[StateAction], reward: float, reward_discount: float
) -> torch.Tensor:
    """Calculate the policy loss for the game.

    Parameters
    ----------
    reward : float
        The reward for the game, if >0 then white won, if <0 then black won.
    """
    policy_loss = torch.tensor(0.0, device=history[0].action_prob.device)
    for state_action in history:
        policy_loss += _action_loss(state_action, reward)
        reward *= reward_discount
    return policy_loss


def _action_loss(state_action: StateAction, reward: float) -> torch.Tensor:
    """Calculate the policy loss for a single action."""
    if state_action.player == chess.BLACK:
        # Neg reward for black player
        return state_action.action_prob * reward
    return -state_action.action_prob * reward
