import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chessformer import logging
from chessformer.model import Chessformer

_GAMES_PTH = Path(__file__).parent.parent.parent / "out"
_CKPT_PTH = Path(__file__).parent.parent.parent / "checkpoints"

MODEL_KWARGS = dict(
    n_hidden=4,
    dim_hidden=768,
    n_layers=24,
    n_heads=12,
    dropout_rate=0.1,
)

mode = "pretrain"
n_positions = 1_000_000
batch_size = 256
learning_rate = 1e-3
learning_rate_decay = 0.98
learning_rate_min = 1e-6
weight_decay = 1e-2
gradient_clip = 1.0
decay_every = 5_000
print_every = 100
checkpoint_every = 100_000

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)  # type: ignore

    if torch.cuda.is_available():
        model.to("cuda")
    if not _GAMES_PTH.exists():
        _GAMES_PTH.mkdir()
    if not _CKPT_PTH.exists():
        _CKPT_PTH.mkdir()

    # Train the model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    logging.info(f"Pre-training Chessformer for {n_positions} positions.")

    losses_mem = []
    illegal_prob_mem = []
    checkmate_f1_mem = []
    decay_in = decay_every
    print_in = print_every
    checkpoint_in = checkpoint_every
    checkpoint_n = 0
    tstart = time.time()
    for position_k in range(0, n_positions, batch_size):
        batch_size = min(batch_size, n_positions - position_k)

        states = model.dataloader.get_boards(batch_size)
        with torch.amp.autocast(  # type: ignore
            device_type=model.device.type,
            enabled=model.device.type == "cuda",
        ):
            loss, metrics = model.step(states)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        # Learning rate decay
        decay_in -= batch_size
        if decay_in <= 0:
            learning_rate = max(learning_rate * learning_rate_decay, learning_rate_min)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
            decay_in = decay_every

        # Checkpoint
        checkpoint_in -= batch_size
        if checkpoint_in <= 0:
            checkpoint_in = checkpoint_every
            checkpoint_n += 1
            torch.save(
                model.state_dict(),
                _CKPT_PTH / f"chessformer_pretrain_{checkpoint_n}.ckpt",
            )

        # Monitoring
        losses_mem.append(loss.item())
        illegal_prob_mem.append(metrics.illegal_prob)
        checkmate_f1_mem.append(metrics.checkmate_f1)
        losses_mem = losses_mem[-100:]
        illegal_prob_mem = illegal_prob_mem[-100:]
        checkmate_f1_mem = checkmate_f1_mem[-100:]

        print_in -= batch_size
        if print_in <= 0:
            print_in = print_every
            pct = 100 * (position_k + batch_size) / n_positions
            elapsed = time.time() - tstart
            pos_per_s = (position_k + batch_size) / elapsed
            message = (
                f"[ Positions {position_k}-{position_k + batch_size}/{n_positions} / {pct:5.1f}% ] "
                f"Loss: {np.mean(losses_mem):.3f} / "  # type: ignore
                f"P(illegal): {np.mean(illegal_prob_mem):.3f} / "  # type: ignore
                f"Checkmate F1: {np.mean(checkmate_f1_mem):.3f} / "  # type: ignore
                f"LR: {learning_rate:.2e} / "
                f"{pos_per_s:.2f} positions/s / "
                f"ETA: {(n_positions - position_k) / pos_per_s / 60:.2f} min"
            )
            print(message, end=" " * 20 + "\r")
            logging.debug(message)

    torch.save(model.state_dict(), "chessformer_pretrain_final.ckpt")
