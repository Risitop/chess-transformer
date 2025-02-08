import torch.nn as nn
import torch
from chessgpt.data.vocab import ChessVocab, ChessTokenizer
from torch.nn import functional as F
from chessgpt.data.dataset import ChessDataset
from torch.utils.data import DataLoader


class ChessBigram(nn.Module):
    """A basic bigram model for chess games."""

    def __init__(self, vocab: ChessVocab):
        super(ChessBigram, self).__init__()
        self._vocab = vocab
        self._embedding = nn.Embedding(len(vocab), len(vocab))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model, returns logits."""
        return self._embedding(x)  # [B, T, len(vocab)]

    def step(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """One training step with a batch of data."""
        # x: [B, T], target: [B]
        logits = self(x)  # [B, T, len(vocab)]
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        return F.cross_entropy(logits, target)

    def _generate_one(self, x: torch.Tensor) -> torch.Tensor:
        """Generates one token."""
        squeezed = False
        if x.dim() == 1:
            squeezed = True
            x = x.unsqueeze(0)
        logits = self(x)[:, -1, :]
        probas = F.softmax(logits, dim=-1)
        predicted = torch.multinomial(probas, num_samples=1)
        if squeezed:
            predicted = predicted.squeeze(0)
        return predicted

    def generate(self, start: str, max_length: int = 20) -> str:
        """Generates a sequence of tokens."""
        self._tokenizer = ChessTokenizer(self._vocab)
        with torch.no_grad():
            xl = self._tokenizer.tokenize(start, add_eos=False, add_sos=True)
            x = torch.tensor(xl).type(torch.long)
            for _ in range(max_length):
                next_token = self._generate_one(x)
                x = torch.cat([x, next_token], dim=-1)
                if next_token == self._vocab.eos_token_id:
                    break
        return self._tokenizer.detokenize(x)

    def fit(
        self,
        train: list[str],
        val: list[str],
        block_size: int = 16,
        batch_size: int = 16,
        epochs: int = 30,
        lr: float = 1e-2,
    ) -> "ChessBigram":
        """Train the model on a dataset."""
        train_dataset = ChessDataset.parse_training_data(
            train, ChessTokenizer(self._vocab), block_size
        )
        val_dataset = ChessDataset.parse_training_data(
            val, ChessTokenizer(self._vocab), block_size
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        print("Launching training...")
        for epoch in range(epochs):
            self.train()
            total_loss_t = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = self.step(x, y)
                loss.backward()
                optimizer.step()
                total_loss_t += loss.item()
            self.eval()
            total_loss_v = 0
            for x, y in val_loader:
                with torch.no_grad():
                    loss = self.step(x, y)
                    total_loss_v += loss.item()
            print(
                f"Epoch {epoch} | Train loss: {total_loss_t / len(train_loader):.2f} "
                f"| Val loss: {total_loss_v / len(val_loader):.2f}"
            )

        return self
