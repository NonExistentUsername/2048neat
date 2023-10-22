import random
from math import log2, sin
from typing import Callable, Tuple

import neat
import torch

from neat2048.fitness import board_to_input, board_to_input_v1, get_fitness
from neat2048.game import Game2048
from neat2048.hyper import Game2048Network, LayerDescriptor


def get_layers_descriptors(
    board_size_x, board_size_y
) -> Tuple[LayerDescriptor, LayerDescriptor, LayerDescriptor]:
    first_l = LayerDescriptor(board_size_x, board_size_y, 1)
    hidden_l = LayerDescriptor(4, 4, 2)
    output_l = LayerDescriptor(4, 1, 1)

    return first_l, hidden_l, output_l


def calculate_fitness(
    genome: neat.DefaultGenome, config: neat.Config, global_seed: int
) -> None:
    random.seed(global_seed)
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    net = None

    def get_net_moves(game: Game2048) -> list[tuple[int, float]]:
        nonlocal net
        inputs = board_to_input_v1(game.board)
        outputs = net.forward(torch.tensor(inputs, dtype=torch.float32)).tolist()
        moves = [(i, output) for i, output in enumerate(outputs)]

        return moves

    board_sizes = [
        [2, 2],
        [3, 3],
        [4, 4],
    ]

    total_fitness = 1

    for board_size in board_sizes:
        board_size_x, board_size_y = board_size
        descriptors = get_layers_descriptors(board_size_x, board_size_y)
        net = Game2048Network(*descriptors, cppn)
        net.init_weights()

        fitness = get_fitness(
            get_net_moves,
            games_count=4,
            count_of_minimal_scores_as_fitness=1,
            board_size_x=board_size_x,
            board_size_y=board_size_y,
        )
        total_fitness *= fitness + 1

    return total_fitness
