import chess
import chess.engine


def generate_all_moves():
    board = chess.Board()
    board.clear()
    all_moves = set()

    # Generate moves for each piece type from each square
    for square in chess.SQUARES:
        piece_types = [
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ]
        for piece_type in piece_types:
            for color in [chess.WHITE, chess.BLACK]:
                piece = chess.Piece(piece_type, color)
                board.set_piece_at(square, piece)

                # Generate legal moves for the current piece
                for move in board.legal_moves:
                    all_moves.add(move.uci())

                # Pawn captures
                if piece_type == chess.PAWN:
                    for move in board.generate_legal_captures():
                        all_moves.add(move.uci())

                # Clear the board for the next iteration
                board.remove_piece_at(square)

    # Add castling moves
    castling_moves = ["e1g1", "e1c1", "e8g8", "e8c8"]
    for move in castling_moves:
        all_moves.add(move)

    # Add promotions
    promotions, columns = "qrbn", "abcdefgh"
    for i, column in enumerate(columns):
        for promotion in promotions:
            all_moves.add(f"{column}7{column}8{promotion}")
            all_moves.add(f"{column}2{column}1{promotion}")
            if i >= 1:
                all_moves.add(f"{column}7{columns[i - 1]}8{promotion}")
                all_moves.add(f"{column}2{columns[i - 1]}1{promotion}")
            if i < len(columns) - 1:
                all_moves.add(f"{column}7{columns[i + 1]}8{promotion}")
                all_moves.add(f"{column}2{columns[i + 1]}1{promotion}")

    return sorted(all_moves)


if __name__ == "__main__":
    moves = generate_all_moves()
    with open("all_moves.txt", "w") as f:
        for move in moves:
            f.write(f"{move}\n")
