import random
from math import log1p, log2
from typing import Callable

import neat

from neat2048.game import Game2048

INFINITY = 10000000000000000000

### START SETTINGS ###
GAMES_COUNT = 4
COUNT_OF_MINIMAL_SCORES_AS_FITNESS = 4

BOARD_SIZE_X = 4
BOARD_SIZE_Y = 4
### END SETTINGS ###

### SOME CONSTS ###
IDEAL_MONOTONICITY = BOARD_SIZE_X * (BOARD_SIZE_Y - 1) + BOARD_SIZE_Y * (
    BOARD_SIZE_X - 1
)

### END CONSTS ###


### START AWARDS ###
GAME_SCORE_AWARD = 0.2 * 1
MOVES_COUNT_AWARD = 0.0011 * 0
EMPTY_CELL_COUNT_AWARD = 1 * 0
MAX_TITLE_AWARD = 2 * 0  # Off for now
MONOTONICITY_AWARD = 2.5 * 0  # Off for now
### END AWARDS ###

### START PENALTIES ###
ILLIGAL_MOVE_PENALTY = 0.03 * 0.1
### END PENALTIES ###


encoded_tiles = {}  # type: dict[int, list[int]]


def encode_tiles(steps: int = 20) -> None:
    vector_size = int(log2(steps)) + 1  # 5 for 20 steps
    tile = 2

    encoded_tiles[0] = [0 for _ in range(vector_size)]

    for i in range(1, steps + 1):
        # use i as representation of tile

        binary_tile = bin(i)[2:]  # Remove 0b prefix

        binary_vector = [int(x) for x in reversed(binary_tile)]

        while len(binary_vector) < vector_size:
            binary_vector.insert(0, 0)

        encoded_tiles[tile] = binary_vector
        tile *= 2


def board_to_input(board: list[list[int]]) -> list[float]:
    inputs = []
    for row in board:
        for tile in row:
            inputs += encoded_tiles[tile]

    return inputs


def board_to_input_v1(board: list[list[int]]) -> list[float]:
    inputs = []
    for row in board:
        for tile in row:
            inputs.append(tile)

    # normalize
    max_tile = max(inputs)
    inputs = [x / max_tile for x in inputs]

    return inputs


# def calc_board_monotonicity(game: Game2048) -> int:
#     monotonicity = 0

#     for x in range(0, BOARD_SIZE_X):
#         right_to_left_count = 0
#         left_to_right_count = 0

#         for y in range(0, BOARD_SIZE_Y - 1):
#             if game[x][y] > game[x][y + 1]:
#                 right_to_left_count += 1
#             elif game[x][y] < game[x][y + 1]:
#                 left_to_right_count += 1

#         monotonicity += max(right_to_left_count, left_to_right_count)

#     for y in range(0, BOARD_SIZE_Y):
#         top_to_bottom_count = 0
#         bottom_to_top_count = 0

#         for x in range(0, BOARD_SIZE_X - 1):
#             if game[x][y] > game[x + 1][y]:
#                 top_to_bottom_count += 1
#             elif game[x][y] < game[x + 1][y]:
#                 bottom_to_top_count += 1

#         monotonicity += max(top_to_bottom_count, bottom_to_top_count)

#     return monotonicity


def get_max_possible_tile(
    board_size_x: int = BOARD_SIZE_X, board_size_y: int = BOARD_SIZE_Y
) -> int:
    return 2 ** (board_size_x * board_size_y)


def play_game(
    get_net_moves: Callable,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    game = Game2048(board_size_x, board_size_y)
    game.add_random_tile()  # Add first tile

    moves_count = 0
    illegal_moves_count = 0
    empty_cells_count = 0

    while not game.game_end:
        moves = get_net_moves(game)

        for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
            changed = game.move(move)
            if changed:
                break
            else:
                illegal_moves_count += 1

        empty_cells_count += game.empty_cells
        moves_count += 1

    fitness = 0

    fitness += log2(game.score + 1) * GAME_SCORE_AWARD
    fitness += moves_count * MOVES_COUNT_AWARD
    fitness += (
        empty_cells_count / (moves_count * board_size_x * board_size_y)
    ) * EMPTY_CELL_COUNT_AWARD  # normalize
    fitness += (
        log2(max([max(row) for row in game.board]))
        / log2(get_max_possible_tile(board_size_x, board_size_y))
    ) * MAX_TITLE_AWARD  # Bonus for max tile

    fitness -= illegal_moves_count * ILLIGAL_MOVE_PENALTY

    return fitness  # 100 is a magic number


def get_fitness(
    get_net_moves: Callable,
    games_count: int = GAMES_COUNT,
    count_of_minimal_scores_as_fitness: int = COUNT_OF_MINIMAL_SCORES_AS_FITNESS,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    fitnesses = []

    for _ in range(games_count):
        fitnesses.append(play_game(get_net_moves, board_size_x, board_size_y))

    fitnesses = sorted(fitnesses, reverse=True)[:count_of_minimal_scores_as_fitness]

    fitness = sum(fitnesses) / len(fitnesses)

    return fitness


def calculate_fitness(
    genome: neat.DefaultGenome, config: neat.Config, global_seed: int
) -> None:
    random.seed(global_seed)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_net_moves(game: Game2048) -> list[tuple[int, float]]:
        inputs = board_to_input(game.board)
        outputs = net.activate(inputs)
        moves = [(i, output) for i, output in enumerate(outputs)]

        return moves

    return get_fitness(get_net_moves)


encode_tiles(20)
