import torch

from chessformer.model import Chessformer

MODEL_KWARGS = dict(
    n_hidden=3,
    dim_hidden=256,
    n_layers=8,
    n_heads=4,
    dropout_rate=0.2,
)
TRAIN_KWARGS = dict(
    n_games=10000,
    batch_size=8,
    learning_rate=1e-4,
    learning_rate_decay=1.0,
    learning_rate_min=1e-6,
    weight_decay=0.0,
    gradient_clip=1.0,
    checkmate_reward=1.0,
    reward_discount=0.99,
)

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)
    if torch.cuda.is_available():
        model.to("cuda")
    model.train(**TRAIN_KWARGS)
    torch.save(model.state_dict(), "chessformer_v1_large_10k.pth")
