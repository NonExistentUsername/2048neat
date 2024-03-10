import random
from math import log2, sin
from typing import Callable, Tuple

import neat

from neat2048.fitness import board_to_input, board_to_input_v1, get_fitness
from neat2048.game import Game2048
from neat2048.hyper import HyperNetwork


def get_layers_descriptors(
    board_size_x, board_size_y
) -> Tuple[list[int], list[int], list[int]]:
    first_l = [board_size_x, board_size_y]
    hidden_l = [12, 12]
    output_l = [4]

    return first_l, hidden_l, output_l


def calculate_fitness(
    genome: neat.DefaultGenome, config: neat.Config, global_seed: int
) -> None:
    random.seed(global_seed)
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_weights(inputs: list[float]) -> list[float]:
        return cppn.activate(inputs)

    net = None

    def get_net_moves(game: Game2048) -> list[tuple[int, float]]:
        nonlocal net
        inputs = board_to_input_v1(game.board)
        outputs = net.forward(inputs)
        moves = [(i, output) for i, output in enumerate(outputs)]

        return moves

    board_sizes = [
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
    ]

    total_fitness = 1

    for board_size in board_sizes:
        board_size_x, board_size_y = board_size
        descriptors = get_layers_descriptors(board_size_x, board_size_y)
        net = HyperNetwork(*descriptors, get_weights)

        fitness = get_fitness(
            get_net_moves,
            games_count=4,
            count_of_minimal_scores_as_fitness=2,
            board_size_x=board_size_x,
            board_size_y=board_size_y,
        )
        total_fitness *= fitness + 1

    return log2(total_fitness)
