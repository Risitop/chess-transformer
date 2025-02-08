from chess.constants import DATA_DIR, WHITE, BLACK

from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Literal

@dataclass
class Game:
    """Represents a single game of chess."""
    
    min_elo: int
    moves: list[str]
    
    def __len__(self) -> int:
        return len(self.moves)
    
    @property
    def winner(self) -> int | None:
        """Returns the winner of the game."""
        if self.moves[-1] == "1-0":
            return WHITE
        elif self.moves[-1] == "0-1":
            return BLACK
        return None
    
def _is_move(move: str) -> bool:
    """Checks if a string is a valid move."""
    return all(char in "abcdefgh12345678NBRQKx-+#O=01" for char in move)
    
def _extract_moves(line: str) -> list[str]:
    """Extracts the moves from a line of PGN."""
    
    moves = line.split(" ")[1:]
    return [move for move in moves if _is_move(move)]
    
def extract_game(file: TextIOWrapper) -> Game:
    """Extracts the next game from a PGN file."""
    
    min_elo = float("inf")
    moves = []
    started = False
    read_lines = 0
    
    while (line := file.readline()):
        line = line.strip()
        if "Event" in line:
            started = True
        elif started and "Elo" in line:
            try:
                elo = int(line.split(" ")[1][1:-2])
            except ValueError:
                elo = float("inf")
            min_elo = min(min_elo, elo)
        elif started and line.startswith("1."):
            moves = _extract_moves(line)
            break
        if read_lines > 20:
            break
        read_lines += started
    
    if not len(line):
        raise EOFError("No more games to read.")
        
    if min_elo == float("inf") or not moves:
        raise ValueError("Invalid game.")
    
    return Game(min_elo, moves)

def extract_games(file_path: Path, n_games: int, min_elo: int) -> list[Game]:
    """Extracts n games from a PGN file.
    
    Parameters
    ----------
    file_path : Path
        The path to the PGN file.
        
    n_games : int
        The number of games to extract.
        
    min_elo : int
        The minimum Elo rating for the games.
        
    Returns
    -------
    list[Game]
        The extracted games.
    """
    
    games = []
    stream = file_path.open("r")
    
    while n_games == -1 or len(games) < n_games:
        try:
            game = extract_game(stream)
        except EOFError:
            break
        except ValueError:
            continue
        if game.min_elo >= min_elo:
            games.append(game)
            if len(games) % 100 == 0:
                print(f"Extracted {len(games)} games.")
    
    stream.close()
    return games

def write_games(games: list[Game], file_path: Path) -> None:
    """Writes game moves to a file.
    
    Parameters
    ----------
    games : list[Game]
        The games to write.
        
    file_path : Path
        The path to the file.
    """
    stream = file_path.open("w")
    for game in games:
        stream.write(" ".join(game.moves) + "\n")
    stream.close()

if __name__ == "__main__":

    filename = "2025-01-big.pgn"
    FILE_DIR = DATA_DIR / "raw" / filename
    games = extract_games(FILE_DIR, 50000, 2000)
    print("Extracted games:", len(games))
    write_games(games, DATA_DIR / "processed" / filename.replace(".pgn", ".txt"))