import itertools
import random
from math import log2
from typing import Callable

import neat

from neat2048.game import Game2048

INFINITY = 10000000000000000000

### START SETTINGS ###
GAMES_COUNT = 5
COUNT_OF_MINIMAL_SCORES_AS_FITNESS = 5

BOARD_SIZE_X = 4
BOARD_SIZE_Y = 4
### END SETTINGS ###

### SOME CONSTS ###
IDEAL_MONOTONICITY = BOARD_SIZE_X * (BOARD_SIZE_Y - 1) + BOARD_SIZE_Y * (
    BOARD_SIZE_X - 1
)

### END CONSTS ###


### START AWARDS ###
GAME_SCORE_AWARD = 0.03 * 1 * 0  # Off for now
MOVES_COUNT_AWARD = 0.002 * 1 * 0  # Off for now
EMPTY_CELL_COUNT_AWARD = 1
MAX_TITLE_AWARD = 0.4 * 0  # Off for now
MONOTONICITY_AWARD = 2.5 * 0  # Off for now
### END AWARDS ###

### START PENALTIES ###
ILLIGAL_MOVE_PENALTY = 0.03 * 0
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


def play_game(
    get_net_moves: Callable,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    game = Game2048(board_size_x, board_size_y)
    game.add_random_tile()  # Add first tile

    # moves_count = 0
    # illegal_moves_count = 0
    # empty_cells_count = 0

    while not game.game_end:
        moves = get_net_moves(game)

        for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
            changed = game.move(move)
            if changed:
                break
            # else:
            #     illegal_moves_count += 1

        # empty_cells_count += game.empty_cells
        # moves_count += 1

    # fitness: float = 0

    # fitness += log2(game.score + 1) * GAME_SCORE_AWARD
    # fitness += moves_count * MOVES_COUNT_AWARD
    # fitness += (
    #     empty_cells_count / (board_size_x * board_size_y)
    # ) * EMPTY_CELL_COUNT_AWARD  # normalize
    # fitness += (
    #     log2(max([max(row) for row in game.board]))
    #     / log2(get_max_possible_tile(board_size_x, board_size_y))
    # ) * MAX_TITLE_AWARD  # Bonus for max tile

    # fitness -= illegal_moves_count * ILLIGAL_MOVE_PENALTY

    # return fitness  # 100 is a magic number

    # Use magic path for now
    path_val: int = find_max_path_from_tiles(game)

    return path_val


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
