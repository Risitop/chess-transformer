import chess
import chess.pgn


def board_to_pgn(board: chess.Board):
    """Convert a chess.Board object to a PGN string."""
    game = chess.pgn.Game()
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_string = game.accept(exporter).replace("\n", " ")
    if pgn_string.endswith("*"):
        pgn_string = pgn_string[:-1]
    return pgn_string
