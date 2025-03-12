import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chessformer import logging, utils
from chessformer.model import Chessformer

_GAMES_PTH = Path(__file__).parent.parent / "out"
_CKPT_PTH = Path(__file__).parent.parent / "checkpoints"

MODEL_KWARGS = dict(
    n_hidden=1,
    dim_hidden=768,
    n_layers=24,
    n_heads=12,
    dropout_rate=0.0,
    n_jobs=16,
)

mode = "pretrain"
n_positions = current_position = 1_000_000
batch_size = 64
accumulate_grad = 32
lr_init = 6e-5
lr_min = 6e-6
lr_warmup = 15
lr_decay_until = 450
weight_decay = 1e-1
gradient_clip = 1.0
decay_every = 5_000
checkpoint_every = 100
compile_model = True
beta1, beta2 = 0.9, 0.95

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)  # type: ignore

    if torch.cuda.is_available():
        logging.info("Sending model to CUDA.")
        model.to("cuda")

    if compile_model:
        logging.info("Compiling model...")
        model = torch.compile(model)

    if not _GAMES_PTH.exists():
        _GAMES_PTH.mkdir()
    if not _CKPT_PTH.exists():
        _CKPT_PTH.mkdir()

    amp_ctx = torch.amp.autocast(  # type: ignore
        device_type=model.device.type,
        enabled=model.device.type == "cuda",
    )

    # Train the model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        fused=True,
    )

    logging.info(f"Pre-training Chessformer for {n_positions} positions.")

    losses_mem = []
    illegal_prob_mem = []
    decay_in = decay_every
    checkpoint_in = checkpoint_every
    checkpoint_n = 0
    step = 0
    tstart = time.time()
    while True:
        # Accumulate gradients
        total_loss = 0.0
        for gstep in range(accumulate_grad):
            batch_size = min(batch_size, current_position)
            current_position -= batch_size
            states = model.dataloader.get_boards(batch_size)
            with amp_ctx:
                loss, metrics = model.step(states)
                loss = loss / accumulate_grad
            loss.backward()
            total_loss += loss.item()

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        learning_rate = utils.cosine_lr_with_warmup(
            lr_init, step, lr_warmup, lr_decay_until, lr_min
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
            decay_in = decay_every
        step += 1

        # Checkpoint
        checkpoint_in -= 1
        if checkpoint_in <= 0:
            checkpoint_in = checkpoint_every
            checkpoint_n += 1
            torch.save(
                model.state_dict(),
                _CKPT_PTH / f"chessformer_pretrain_{checkpoint_n}.ckpt",
            )

        # Monitoring
        losses_mem.append(total_loss)
        illegal_prob_mem.append(metrics.illegal_prob)
        losses_mem = losses_mem[-100:]
        illegal_prob_mem = illegal_prob_mem[-100:]

        position_k = n_positions - current_position
        seen = utils.bigint_to_str(position_k)
        pct = 100 * position_k / n_positions
        elapsed = time.time() - tstart
        pos_per_s = position_k / elapsed
        message = (
            f"[ Step {step} / {pct:5.1f}% / #boards: {seen} ] "
            f"Loss: {np.mean(losses_mem):.3f} / "  # type: ignore
            f"|Grad|: {gnorm:.4f} / "
            f"LR: {learning_rate:.2e} / "
            f"p(illegal): {np.mean(illegal_prob_mem):.3f} / "  # type: ignore
            f"{pos_per_s:.2f} pos/s / "
            f"DL buffer: {len(model.dataloader._board_buffer)} / "
            f"ETA: {(n_positions - position_k) / pos_per_s / 60:.2f} min"
        )
        logging.info(message)

        if current_position <= 0:
            break

    torch.save(model.state_dict(), "chessformer_pretrain_final.ckpt")
