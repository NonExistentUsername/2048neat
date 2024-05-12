import enum


class FitnessTypes(enum.IntEnum):
    EMPTY_CELLS = 0
    GAME_SCORE = 1
    ILLEGAL_MOVES = 2
    LONG_FALLING = 3
    MAX_TILE = 4
    MOVES_COUNT = 5
    SMOOTHNESS = 6
