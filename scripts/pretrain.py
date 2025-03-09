import torch

from chessformer.model import Chessformer

SUFFIX = 2

MODEL_KWARGS = dict(
    n_hidden=5,
    dim_hidden=768,
    n_layers=24,
    n_heads=12,
    dropout_rate=0.1,
)
TRAIN_KWARGS = dict(
    mode="pretrain",
    n_games=20000,
    moves_per_game=192,
    batch_size=256,
    learning_rate=1e-3,
    learning_rate_decay=1.0,
    learning_rate_min=1e-6,
    weight_decay=1e-2,
    gradient_clip=1.0,
    checkmate_reward=0.0,
)

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)  # type: ignore
    if torch.cuda.is_available():
        model.to("cuda")
    model.train(**TRAIN_KWARGS)  # type: ignore
    torch.save(model.state_dict(), "chessformer_pretrain.pth")
