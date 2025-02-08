from chessgpt.engine import utils


def test_pos_conversions():
    for col in "abcdefgh":
        for row in range(1, 9):
            initial_pos = f"{col}{row}"
            cartesian_pos = utils.str_to_cartesian_pos(initial_pos)
            uint64_pos = utils.cartesian_to_uint64_pos(cartesian_pos)
            cartesian_pos_2 = utils.uint64_to_cartesian_pos(uint64_pos)
            assert cartesian_pos == cartesian_pos_2
            assert initial_pos == utils.cartesian_to_str_pos(cartesian_pos)
