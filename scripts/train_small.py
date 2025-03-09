import torch

from chessformer.model import Chessformer

MODEL_KWARGS = dict(
    n_hidden=3,
    dim_hidden=128,
    n_layers=8,
    n_heads=4,
    dropout_rate=0.2,
)
TRAIN_KWARGS = dict(
    n_games=500,
    batch_size=32,
    learning_rate=1e-3,
    learning_rate_decay=0.99,
    learning_rate_min=1e-6,
    weight_decay=1e-4,
    gradient_clip=1.0,
    checkmate_reward=200.0,
    reward_discount=0.99,
)

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)
    if torch.cuda.is_available():
        model.to("cuda")
    model.train(**TRAIN_KWARGS)
