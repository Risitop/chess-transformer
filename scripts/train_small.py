import torch

from chessformer.model import Chessformer

SUFFIX = 2

MODEL_KWARGS = dict(
    n_hidden=3,
    dim_hidden=256,
    n_layers=8,
    n_heads=4,
    dropout_rate=0.2,
)
TRAIN_KWARGS = dict(
    n_games=10000,
    batch_size=16,
    learning_rate=5e-4,
    learning_rate_decay=1.0,
    learning_rate_min=1e-6,
    weight_decay=0.0,
    gradient_clip=1.0,
    checkmate_reward=0.1,
    reward_discount=0.99,
    warmup_games=2000,
)

if __name__ == "__main__":
    model = Chessformer(**MODEL_KWARGS)
    weights = torch.load("chessformer_v1_large_10k.pth")
    model.load_state_dict(weights)
    if torch.cuda.is_available():
        model.to("cuda")
    model.train(**TRAIN_KWARGS)
    torch.save(model.state_dict(), f"chessformer_v1_large_10k_{SUFFIX}.pth")
