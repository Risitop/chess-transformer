import torch

from chessformer.model import Chessformer

SUFFIX = 2

MODEL_KWARGS = dict(
    n_hidden=10,
    dim_hidden=512,
    n_layers=24,
    n_heads=8,
    dropout_rate=0.2,
)
TRAIN_KWARGS = dict(
    mode="pretrain",
    n_games=20000,
    moves_per_game=192,
    batch_size=256,
    learning_rate=1e-3,
    learning_rate_decay=0.999,
    learning_rate_min=1e-6,
    weight_decay=1e-2,
    gradient_clip=1.0,
    checkmate_reward=0.0,
    repeat_penalty=0.01,
)

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)  # type: ignore
    if torch.cuda.is_available():
        model.to("cuda")
    model.train(**TRAIN_KWARGS)  # type: ignore
    torch.save(model.state_dict(), "chessformer_pretrain.pth")
