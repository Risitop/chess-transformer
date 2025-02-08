import json
from chessgpt.constants import DATA_DIR
import torch

_PAD = 20


class ChessTokenizer:
    def __init__(self, vocab: "ChessVocab"):
        self._vocab = vocab

    def tokenize(self, string: str, add_sos: bool, add_eos: bool) -> list[int]:
        """Tokenizes a string into a tensor of token IDs."""
        if add_sos:
            string = self._vocab.inv(self._vocab.sos_token_id) + string
        if add_eos:
            string += self._vocab.inv(self._vocab.eos_token_id)
        values = []
        start = 0
        tokens = self._vocab.tokens
        while start < len(string):
            found = False
            for tk in tokens:
                if len(tk) + start > len(string):
                    continue
                if string.startswith(tk, start):
                    values.append(self._vocab[tk])
                    start += len(tk)
                    found = True
                    break
            if not found:
                raise ValueError(f"Invalid token: {string[start:]}")
        return values

    def detokenize(self, tensor: torch.Tensor) -> str:
        """Detokenizes a tensor of token IDs into a string."""
        return (
            "".join([self._vocab.inv(int(token.item())) for token in tensor[1:]])
            .strip()
            .replace("  ", " ")
        )


class ChessVocab:
    """A vocabulary for chess games tokenization."""

    def __init__(self, mapper: dict[str, int]):
        self._mapper = mapper
        self._inv_mapper = {v: k for k, v in mapper.items()}

    @classmethod
    def get_base_vocab(cls) -> "ChessVocab":
        """Generates a basic vocabulary for chess games."""
        pieces = ["P", "N", "B", "R", "Q", "K"]
        positions = [f"{col}{row}" for col in "abcdefgh" for row in range(1, 9)]
        specials = ["O-O", "O-O-O", "+", "#", "=", "x", " "]
        internals = ["<sos>", "<eos>", "<pad>"]
        vocab = pieces + positions + specials + internals
        mapper = {token: i for i, token in enumerate(vocab)}
        for frag in vocab:  # Add smallest fragments
            for c in frag:
                if c not in mapper:
                    mapper[c] = len(mapper)
        return cls(mapper)

    def __len__(self):
        return len(self._mapper)

    def __getitem__(self, token: str) -> int:
        return self._mapper[token]

    def __contains__(self, token: str) -> bool:
        return token in self._mapper

    def __iter__(self):
        return iter(self._mapper)

    @property
    def pad_token_id(self) -> int:
        """Returns the padding token ID."""
        return self["<pad>"]

    @property
    def sos_token_id(self) -> int:
        """Returns the start of sequence token ID."""
        return self["<sos>"]

    @property
    def eos_token_id(self) -> int:
        """Returns the end of sequence token ID."""
        return self["<eos>"]

    def inv(self, token_id: int) -> str:
        """Returns the token from the token ID."""
        return self._inv_mapper[token_id]

    def enrich_vocab(self, data: list[str], target_size: int):
        """Enriches the vocabulary with new tokens using basic BPE."""
        tokenizer = ChessTokenizer(self)
        tk_data = [
            tokenizer.tokenize(game, add_sos=False, add_eos=False) for game in data
        ]
        while len(self) < target_size:
            pair_counts = {}
            for game in tk_data:
                for i in range(len(game) - 1):
                    pair = (game[i], game[i + 1])
                    pair_counts.setdefault(pair, 0)
                    pair_counts[pair] += 1
            if len(pair_counts) == 0:
                break
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore
            new_token = self.inv(best_pair[0]) + self.inv(best_pair[1])
            new_token_id = self.extend(new_token)
            tk_data = [_merge_pair(best_pair, new_token_id, game) for game in tk_data]
            print(
                f"Added token: `{new_token}` (ID: {new_token_id})" + " " * _PAD,
                end="\r",
            )

    def extend(self, token: str) -> int:
        """Extends the vocabulary with new tokens."""
        if token in self:
            print(f"Token {token} already in vocabulary.")
            return -1
        new_id = len(self)
        self._mapper[token] = new_id
        self._inv_mapper[new_id] = token
        return new_id

    @property
    def tokens(self) -> list[str]:
        """Returns the sorted list of tokens, longest tokens first."""
        return sorted(self, key=len, reverse=True)

    def save(self, name: str) -> None:
        """Saves the vocabulary to a JSON file."""
        with open(DATA_DIR / "vocab" / f"{name}.json", "w") as f:
            json.dump(self._mapper, f)

    @classmethod
    def load(cls, name: str) -> "ChessVocab":
        """Loads the vocabulary from a JSON file."""
        with open(DATA_DIR / "vocab" / f"{name}.json", "r") as f:
            mapper = json.load(f)
        return cls(mapper)


def _merge_pair(pair: tuple, new_token: int, data: list[int]) -> list[int]:
    """Merges all pairs of consecutive tokens into a new one."""
    new_data = []
    i = 0
    while i < len(data):
        if i < len(data) - 1 and data[i] == pair[0] and data[i + 1] == pair[1]:
            new_data.append(new_token)
            i += 2
        else:
            new_data.append(data[i])
            i += 1
    return new_data
