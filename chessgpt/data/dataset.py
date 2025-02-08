from torch.utils.data import Dataset
from chessgpt.data.vocab import ChessTokenizer
import torch


class ChessDataset(Dataset):
    """Training dataset for chess games."""

    def __init__(self, contexts: list[str], targets: list[str]):
        self._contexts = contexts
        self._targets = targets

    def __len__(self):
        return len(self._contexts)

    def __getitem__(self, idx: int):
        return self._contexts[idx], self._targets[idx]

    @classmethod
    def parse_training_data(
        cls, dataset: list[str], tokenizer: ChessTokenizer, block_size: int
    ) -> "ChessDataset":
        """Returns a tensor of token IDs from a list of strings."""
        print(f"Loading {len(dataset)} games...")
        contexts, targets = [], []
        for game in dataset:
            try:
                tokens = tokenizer.tokenize(game, add_sos=True, add_eos=True)
                tokens_t = torch.tensor(tokens).type(torch.long)
            except ValueError:
                print(f"Skipping invalid game: {game}")
                continue
            for i in range(block_size - 1):
                context = torch.zeros(block_size, dtype=torch.long).fill_(
                    tokenizer._vocab.pad_token_id
                )
                target = torch.zeros(block_size, dtype=torch.long).fill_(
                    tokenizer._vocab.pad_token_id
                )
                context[:i] = tokens_t[:i]
                target[:i] = tokens_t[1 : i + 1]
                contexts.append(context)
                targets.append(target)
                if i >= len(tokens) - 1:
                    break
        return ChessDataset(contexts=contexts, targets=targets)
