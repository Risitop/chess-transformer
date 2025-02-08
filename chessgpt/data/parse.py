from chessgpt.constants import DATA_DIR
from pathlib import Path
import chess.pgn as pgn
import chess
from typing import NamedTuple, Literal


class GameData(NamedTuple):
    """Simple data class for storing game data."""

    elo: int
    moves: list[str]
    result: Literal["1-0", "0-1", "1/2-1/2"]


def extract_games(
    file_path: Path, n_games: int, min_elo: int, out: Path, chunk_size: int = 1000
) -> list[GameData]:
    """Extracts n games from a PGN file."""
    games = []
    stream = file_path.open("r")
    total = 0

    while total < n_games:
        game = pgn.read_game(stream)
        if game is None:
            break
        elo = min(int(game.headers["WhiteElo"]), int(game.headers["BlackElo"]))
        if elo < min_elo:
            continue
        uci_moves = [move.uci() for move in game.mainline_moves()]
        san_moves = (
            game.board()
            .variation_san([chess.Move.from_uci(move) for move in uci_moves])
            .split(" ")
        )
        san_moves = [move for i, move in enumerate(san_moves) if i % 3]
        try:
            enriched_moves = _enrich_moves(uci_moves, san_moves)
        except ValueError as _:
            continue
        result = game.headers["Result"]
        if result not in ["1-0", "0-1", "1/2-1/2"]:
            continue
        games.append(GameData(elo, enriched_moves, result))  # type: ignore
        if len(games) >= chunk_size:
            total += len(games)
            print(f"Writing chunk ({total} total games saved)...")
            out_stream = out.open("a")
            for game in games:
                out_stream.write(f"{game.elo},")
                out_stream.write(" ".join(game.moves) + ",")
                out_stream.write(f"{game.result}\n")
            games = []
            out_stream.close()

    stream.close()
    return games


def _enrich_moves(uci_moves: list, san_moves: list) -> list:
    """Enriches UCI moves with SAN moves."""
    enriched_moves = []
    if len(uci_moves) != len(san_moves):
        raise ValueError("UCI and SAN moves must have the same length.")
    if len(uci_moves) == 0:
        raise ValueError("No moves to enrich.")
    for uci, san in zip(uci_moves, san_moves):
        piece_id = san[0]
        if piece_id == "O":
            enriched_moves.append(san)
            continue
        if piece_id in "abcdefgh":
            piece_id = "P"
        if piece_id not in "KQRBNP":
            raise ValueError(f"Invalid piece identifier: {piece_id}")
        src, target = uci[:2], uci[2:4]
        capture = "x" if "x" in san else ""
        base_move = f"{piece_id}{src}{capture}{target}"
        suffix_idx = san.find(target) + len(target)
        enriched_moves.append(f"{base_move}{san[suffix_idx:]}")
    return enriched_moves


if __name__ == "__main__":
    filename = "2025-01-big.pgn"
    FILE_DIR = DATA_DIR / "raw" / filename
    OUT_DIR = DATA_DIR / "processed" / (filename[:-4] + ".gptd")
    if OUT_DIR.exists():
        raise FileExistsError(f"Output file already exists: {OUT_DIR}")
    f = FILE_DIR.open("r")
    games = extract_games(FILE_DIR, 20000, 1500, OUT_DIR)
