import itertools
import random
from math import fabs, log2
from typing import Callable, List

import neat  # type: ignore
import numpy as np

from neat2048.game import Game2048

INFINITY = 10000000000000000000

### START SETTINGS ###
GAMES_COUNT = 10
COUNT_OF_MINIMAL_SCORES_AS_FITNESS = 10

BOARD_SIZE_X = 4
BOARD_SIZE_Y = 4
### END SETTINGS ###

### SOME CONSTS ###
IDEAL_MONOTONICITY = BOARD_SIZE_X * (BOARD_SIZE_Y - 1) + BOARD_SIZE_Y * (
    BOARD_SIZE_X - 1
)

### END CONSTS ###


### START AWARDS ###
GAME_SCORE_AWARD = 0.03 * 1
MOVES_COUNT_AWARD = 0.002 * 1
EMPTY_CELL_COUNT_AWARD = 0.008 * 0
MAX_TITLE_AWARD = 0.4 * 1
MONOTONICITY_AWARD = 2.5 * 0  # Off for now
### END AWARDS ###

### START PENALTIES ###
ILLIGAL_MOVE_PENALTY = 0.0003 * 0.2
### END PENALTIES ###

###
score_w = 1.0
smoothness_w = 1.0
###


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
    inputs = [log2(float(x) + 1) for x in inputs]
    max_tile = float(max(inputs))
    inputs = [x / max_tile for x in inputs]

    return inputs


def get_max_possible_tile(
    board_size_x: int = BOARD_SIZE_X, board_size_y: int = BOARD_SIZE_Y
) -> int:
    return 2 ** (board_size_x * board_size_y)


def process_long_falling_from_tile(game: Game2048, x: int, y: int) -> int:
    # Min value is 2, max is 4*4 = 16

    best_neighbour_positions = []
    best_neighbour_value = -1

    is_valid = lambda x, y: 0 <= x < game.size_x and 0 <= y < game.size_y

    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        new_x, new_y = x + dx, y + dy

        if not is_valid(new_x, new_y):
            continue

        value = game.board[new_x][new_y]
        if value > best_neighbour_value and value < game.board[x][y]:
            best_neighbour_value = value
            best_neighbour_positions = [(new_x, new_y)]
        elif value == best_neighbour_value:
            best_neighbour_positions.append((new_x, new_y))

    best_val = 1
    for x, y in best_neighbour_positions:
        best_val = max(best_val, 1 + process_long_falling_from_tile(game, x, y))

    return best_val


def find_max_path_from_tiles(game: Game2048) -> int:
    # Min value is 2, max is 4*4 = 16
    max_tile_value = 0
    for x, y in itertools.product(range(game.size_x), range(game.size_y)):
        if game.board[x][y] > max_tile_value:
            max_tile_value = game.board[x][y]

    longest_path = 0
    for x, y in itertools.product(range(game.size_x), range(game.size_y)):
        if game.board[x][y] == max_tile_value:
            longest_path = max(longest_path, process_long_falling_from_tile(game, x, y))

    return longest_path


# Smoothness is the sum of all: differences between two non-zero tiles / the smaller tile
def calc_smoothness(game: Game2048, board_size_x=4, board_size_y=4):
    smoothness = 0.0
    # Only rotate twice to avoid double counting
    # Ignore 0 tiles
    board_size_x = len(game.board)
    board_size_y = len(game.board[0])

    for i in range(board_size_x):
        for j in range(board_size_y):
            if (
                game.board[i][j] != 0
                and j + 1 < board_size_y
                and game.board[i][j + 1] != 0
            ):
                current_smoothness = fabs(
                    log2(game.board[i][j]) - log2(game.board[i][j + 1])
                )
                smoothness = smoothness - current_smoothness

    for j in range(board_size_y):
        for i in range(board_size_x):
            if (
                game.board[i][j] != 0
                and i + 1 < board_size_x
                and game.board[i + 1][j] != 0
            ):
                current_smoothness = fabs(
                    log2(game.board[i][j]) - log2(game.board[i + 1][j])
                )
                smoothness = smoothness - current_smoothness

    return smoothness


def generate_patterns():
    patterns = [
        [
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [8, 9, 10, 11],
            [15, 14, 13, 12],
        ],
        [
            [3, 2, 1, 0],
            [4, 5, 6, 7],
            [11, 10, 9, 8],
            [12, 13, 14, 15],
        ],
        [
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3],
        ],
        [
            [12, 13, 14, 15],
            [11, 10, 9, 8],
            [4, 5, 6, 7],
            [3, 2, 1, 0],
        ],
        [
            [3, 4, 11, 12],
            [2, 5, 10, 13],
            [1, 6, 9, 14],
            [0, 7, 8, 15],
        ],
        [
            [12, 11, 4, 3],
            [13, 10, 5, 2],
            [14, 9, 6, 1],
            [15, 8, 7, 0],
        ],
        [
            [15, 8, 7, 0],
            [14, 9, 6, 1],
            [13, 10, 5, 2],
            [12, 11, 4, 3],
        ],
        [
            [0, 7, 8, 15],
            [1, 6, 9, 14],
            [2, 5, 10, 13],
            [3, 4, 11, 12],
        ],
    ]
    r = 0.25

    patterns2 = []
    for pattern in patterns:
        new_pattern = []
        for row in pattern:
            new_row = [r**tile for tile in row]
            new_pattern.append(new_row)
        patterns2.append(new_pattern)

    return [np.array(pattern) for pattern in patterns2]


NUMPY_PATTERNS = generate_patterns()


def calc_smoothness_v2(game: Game2048, board_size_x=4, board_size_y=4):
    numpy_board = np.array(game.board)

    patterns = NUMPY_PATTERNS

    smoothness = 0.0
    for pattern in patterns:
        current_smoothness = np.sum(numpy_board * pattern)
        smoothness = max(smoothness, current_smoothness)
    return smoothness


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

    smoothness_sum = 0

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
        # smoothness_sum = calc_smoothness(game) + smoothness_sum * 0.9

    fitness: float = 0

    # Approach using smoothness and score

    # return smoothness_sum

    fitness += log2(game.score + 1) * GAME_SCORE_AWARD
    # fitness += moves_count * MOVES_COUNT_AWARD
    # fitness += (
    #     empty_cells_count / (board_size_x * board_size_y * moves_count)
    # )
    # fitness += find_max_path_from_tiles(game) / board_size_x * board_size_y

    # fitness -= illegal_moves_count * ILLIGAL_MOVE_PENALTY

    return fitness  # 100 is a magic number

    # Use magic path for now

    # return path_val


def play_game_parallel(
    get_net_moves: Callable,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
    games_count: int = GAMES_COUNT,
) -> float:
    games = [Game2048(board_size_x, board_size_y) for _ in range(games_count)]
    for game in games:
        game.add_random_tile()  # Add first tile

    fitnesses = []
    illegal_moves_count = 0

    while games:
        games_moves = get_net_moves(games)

        for moves, game in zip(games_moves, games):
            for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
                changed = game.move(move)
                if changed:
                    break
                else:
                    illegal_moves_count += 1

        for game in games:
            if game.game_end:
                games.remove(game)
                tmp_fitness = log2(game.score + 1) * GAME_SCORE_AWARD
                tmp_fitness += (
                    log2(max(max(row) for row in game.board))
                    / log2(get_max_possible_tile(board_size_x, board_size_y))
                ) * MAX_TITLE_AWARD

                fitnesses.append(tmp_fitness)

    fitness = sum(fitnesses) / len(fitnesses)
    fitness -= illegal_moves_count * ILLIGAL_MOVE_PENALTY

    return max(0, fitness)  # to be sure that we don't have negative


def get_fitness(
    get_net_moves: Callable,
    games_count: int = GAMES_COUNT,
    count_of_minimal_scores_as_fitness: int = COUNT_OF_MINIMAL_SCORES_AS_FITNESS,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    fitnesses = [
        play_game(get_net_moves, board_size_x, board_size_y) for _ in range(games_count)
    ]
    return sum(fitnesses) / len(fitnesses)


def get_fitness_parallel(
    get_net_moves_parallel: Callable,
    games_count: int = GAMES_COUNT,
    count_of_minimal_scores_as_fitness: int = COUNT_OF_MINIMAL_SCORES_AS_FITNESS,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    return play_game_parallel(
        get_net_moves_parallel, board_size_x, board_size_y, games_count
    )


def calculate_fitness(
    genome: neat.DefaultGenome, config: neat.Config, global_seed: int
):
    random.seed(global_seed)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_net_moves(game: Game2048) -> list[tuple[int, float]]:
        inputs = board_to_input_v1(game.board)
        outputs = net.activate(inputs)
        moves = list(enumerate(outputs))

        return moves

    return get_fitness(get_net_moves)


encode_tiles(20)
