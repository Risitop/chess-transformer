import functools
import math
import numpy as np


# Position formalism conversion functions
@functools.cache
def cartesian_to_str_pos(pos: tuple[int, int]) -> str:
    """Converts a cartesian position to a string position."""
    return chr(pos[0] + ord("a")) + str(pos[1] + 1)


@functools.cache
def cartesian_to_uint64_pos(pos: tuple[int, int]) -> np.uint64:
    """Converts a cartesian position to a uint64 bitboard."""
    return np.uint64(1) << np.uint64(pos[1] * 8 + pos[0])


@functools.cache
def str_to_cartesian_pos(pos: str) -> tuple[int, int]:
    """Converts a string position to a cartesian position."""
    return ord(pos[0]) - ord("a"), int(pos[1]) - 1


@functools.cache
def str_to_uint64_pos(pos: str) -> np.uint64:
    """Converts a string position to a uint64 bitboard."""
    x, y = str_to_cartesian_pos(pos)
    return np.uint64(1) << np.uint64(y * 8 + x)


@functools.cache
def uint64_to_str_pos(pos: np.uint64) -> str:
    """Converts a uint64 bitboard to a string position."""
    return cartesian_to_str_pos(uint64_to_cartesian_pos(pos))


@functools.cache
def uint64_to_cartesian_pos(pos: np.uint64) -> tuple[int, int]:
    """Converts a uint64 bitboard to a cartesian position."""
    bit_pos = int(math.log2(pos))
    return bit_pos % 8, bit_pos // 8


@functools.cache
def any_to_uint64_pos(pos: tuple[int, int] | str) -> np.uint64:
    """Converts any position to a uint64 bitboard."""
    if isinstance(pos, str):
        return str_to_uint64_pos(pos)
    if isinstance(pos, tuple):
        return cartesian_to_uint64_pos(pos)
    return pos
