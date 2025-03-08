import dataclasses
import logging
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer.model.mlp import MLP
from chessformer import utils
import time

_MOVES_PTH = Path(__file__).parent.parent.parent / "assets" / "all_moves.txt"
_GAMES_PTH = Path(__file__).parent.parent.parent / "out"
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BoardState:
    """Dataclass for storing the state of a chess board."""

    board: chess.Board
    state: torch.Tensor
    player: torch.Tensor


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
        reward_discount: float,
    ):
        super().__init__()
        # Internals
        self.all_moves = _load_moves()
        self.move2idx = {move: idx for idx, move in enumerate(self.all_moves)}
        self.idx2move = {idx: move for idx, move in enumerate(self.all_moves)}
        self.reward_discount = reward_discount
        self.history = []
        self.games_path = _GAMES_PTH / time.strftime("%Y-%m-%d_%H-%M-%S.games")

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
        logger.info(f"Chessformer initialized with {nparams} trainable parameters.")

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
        action_probs = F.softmax(logits, dim=-1)

        # Calculate move legality loss
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_target = torch.zeros(len(self.all_moves))
        legal_idx = [self.move2idx[move] for move in legal_moves]
        legal_target[legal_idx] = 1.0 / len(legal_moves)
        loss = F.cross_entropy(action_probs, legal_target)

        # Keep track of the chosen policy
        action_probs = F.softmax(action_probs[legal_idx], dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        board.push_uci(legal_moves[action.item()])  # type: ignore
        self.history.append(StateAction(action_probs[action], board.turn))

        return loss

    def get_total_policy_loss(self, reward: float) -> torch.Tensor:
        """Calculate the policy loss for the game.

        Parameters
        ----------
        reward : float
            The reward for the game, if >0 then white won, if <0 then black won.
        """
        policy_loss = 0
        for state_action in self.history:
            policy_loss += _action_loss(state_action, reward)
            reward *= self.reward_discount
        return policy_loss / len(self.history)  # type: ignore

    def reset(self):
        self.history = []

    def train(
        self,
        n_games: int,
        moves_per_game: int = 256,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.999,
        learning_rate_min: float = 1e-6,
        weight_decay: float = 1e-4,
        gradient_clip: float = 1.0,
        log_every_n: int = 100,
        checkmate_reward: float = 100.0,
    ) -> "Chessformer":
        """Train the model using self-play.

        Parameters
        ----------
        n_games : int
            Number of games to play.
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
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        for game_n in range(n_games):
            board = chess.Board()
            self.reset()
            losses = []

            # Play a game
            for _ in range(moves_per_game):
                losses.append(self.step(board))
                if board.is_game_over():
                    break

            loss: torch.Tensor = sum(losses) / len(losses)  # type: ignore
            if board.is_checkmate():
                if board.result() == "1-0":
                    reward = checkmate_reward
                else:
                    reward = -checkmate_reward
                loss += self.get_total_policy_loss(reward)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
            optimizer.step()

            print(
                f"[ Game {game_n} ] Moves: {len(board.move_stack)} / "
                f"Loss: {loss.item():.4f} / "
                f"LR: {learning_rate:.2e}",
                end=" " * 20 + "\r",
            )
            if not (game_n % log_every_n):
                with open(self.games_path, "a") as f:
                    f.write(utils.board_to_pgn(board) + "\n")

            # Learning rate decay
            learning_rate = max(learning_rate * learning_rate_decay, learning_rate_min)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        return self


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


def _action_loss(state_action: StateAction, reward: float) -> torch.Tensor:
    """Calculate the policy loss for a single action."""
    if state_action.player == chess.BLACK:
        # Neg reward for black player
        return state_action.action_prob * reward
    return -state_action.action_prob * reward
