import random
from chessgpt.engine.state import GameState


def test_game():
    for _ in range(30):
        board = GameState.initialize()
        for _ in range(50):
            moves = board.get_legal_moves()
            if not moves:
                break
            move = random.choice(list(moves.values()))
            board = board.apply_move(move)
            if board.is_ended:
                break
