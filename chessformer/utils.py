import math
from pathlib import Path

import chess
import chess.pgn

_MOVES_PTH = Path(__file__).parent.parent / "assets" / "all_moves.txt"


def load_moves() -> list[str]:
    """Load all possible chess moves from file."""
    with open(_MOVES_PTH, "r") as f:
        return f.read().splitlines()


def board_to_pgn(board: chess.Board):
    """Convert a chess.Board object to a PGN string."""
    game = chess.pgn.Game()
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter).replace("\n", " ")
    if pgn_string.endswith("*"):
        pgn_string = pgn_string[:-1]
    return pgn_string


def cosine_lr_with_warmup(
    lr: float, iteration: int, warmup: int, decay_until: int, min_lr: float
) -> float:
    """Cosine learning rate schedule with warmup and decay."""
    if iteration < warmup:
        return lr * (iteration + 1) / (warmup + 1)
    if iteration > decay_until:
        return min_lr
    decay_ratio = (iteration - warmup) / (decay_until - warmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


def bigint_to_str(x: int) -> str:
    """Convert a big integer to a string."""
    if x > 1e12:
        return f"{x / 1e12:.1f}T"
    if x > 1e9:
        return f"{x / 1e9:.1f}B"
    if x > 1e6:
        return f"{x / 1e6:.1f}M"
    if x > 1e3:
        return f"{x / 1e3:.1f}K"
    return str(x)
