import random
from typing import Callable

import neat  # type: ignore

from neat2048.fitness import (
    BOARD_SIZE_X,
    BOARD_SIZE_Y,
    COUNT_OF_MINIMAL_SCORES_AS_FITNESS,
    GAMES_COUNT,
    Game2048,
    board_to_input_v1,
    calc_smoothness_v2,
    find_max_path_from_tiles,
    get_fitness,
    play_game,
)


def random_filling(game: Game2048):
    max_tile_square = 14
    min_tile_square = 1

    for x in range(game.size_x):
        for y in range(game.size_y):
            game[x][y] = 2 ** random.randint(min_tile_square, max_tile_square)

    return game


def play_game_optimized(
    get_net_moves: Callable, board_size_x: int, board_size_y: int
) -> float:
    game = Game2048(board_size_x, board_size_y)
    game = random_filling(game)

    moves = get_net_moves(game)

    for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
        changed = game.move(move, add_random_tile=False)
        if changed:
            break

    fitness: float = 0

    return calc_smoothness_v2(game)

    fitness += find_max_path_from_tiles(game) / (board_size_x * board_size_y)

    return (1.2**fitness) * 10


def get_fitness_optimized(
    get_net_moves: Callable,
    games_count: int = 512,
    board_size_x: int = BOARD_SIZE_X,
    board_size_y: int = BOARD_SIZE_Y,
) -> float:
    fitnesses = [
        play_game_optimized(get_net_moves, board_size_x, board_size_y)
        for _ in range(games_count)
    ]

    return sum(fitnesses) / len(fitnesses)


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

    return get_fitness_optimized(get_net_moves)
