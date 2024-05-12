import random
from math import log2, sin
from typing import Callable, List, Optional, Tuple

import neat  # type: ignore

from neat2048.fitness import (
    board_to_input,
    board_to_input_v1,
    get_fitness,
    get_fitness_parallel,
)
from neat2048.game import Game2048
from neat2048.hyper import HyperNetwork


def generate_boards_min_max_algo(game: Game2048, depth: int) -> list[Game2048]:
    if depth == 0:
        return [game]

    boards = []
    for move in range(4):
        new_game = game.copy()
        new_game.move(move, add_random_tile=False)
        boards.extend(generate_boards_min_max_algo(new_game, depth - 1))

    return boards


def get_layers_descriptors(board_size_x, board_size_y) -> List[Tuple[int, int]]:
    return [
        (board_size_x, board_size_y),
        (6, 6),
        (1, 1),  # for min-max
        # (2, 2),  # for move prediction
    ]


def calculate_fitness(
    genome: neat.DefaultGenome, config: neat.Config, global_seed: int
) -> float:
    random.seed(global_seed)
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_weights(inputs: list[float]) -> list[float]:
        return cppn.activate(inputs)

    net: Optional[HyperNetwork] = None

    def get_net_moves(games: list[Game2048]) -> list[list[tuple[int, float]]]:
        nonlocal net
        if not net:
            raise ValueError("net is None")

        all_inputs: List[List[float]] = []

        for game in games:
            inputs = board_to_input_v1(game.board)
            all_inputs.append(inputs)

        outputs = net.forward(all_inputs)
        outputs_enumerated = []
        for output in outputs:
            outputs_enumerated.append(list(enumerate(output)))

        return outputs_enumerated

    def get_net_moves_using_min_max(
        games: list[Game2048],
    ) -> list[list[tuple[int, float]]]:
        nonlocal net
        if not net:
            raise ValueError("net is None")

        all_inputs: List[List[float]] = []

        for game in games:
            generate_more_games = generate_boards_min_max_algo(
                game, 2
            )  # depth=1, so we get 4 moves
            for generated_game in generate_more_games:
                inputs = board_to_input_v1(generated_game.board)
                all_inputs.append(inputs)

        outputs = net.forward(all_inputs)
        outputs = outputs.flatten()
        # print(f"Outputs shape: {outputs.shape}")
        # print(f"Outputs: {outputs}")

        outputs_reformatted: List[List[float]] = []
        for i in range(0, len(outputs), 4 * 4):
            moves = []
            for j in range(4):
                moves.append(max(outputs[i + j * 4 : i + (j + 1) * 4]))

            outputs_reformatted.append(moves)
            # outputs_reformatted.append(outputs[i : i + 4])

        outputs_enumerated: list[list[tuple[int, float]]] = []
        for output in outputs_reformatted:
            outputs_enumerated.append(list(enumerate(output)))

        # print(f"Outputs enumerated shape: {outputs_enumerated}")

        return outputs_enumerated

    board_sizes = [
        [2, 2],
        [3, 3],
        # [4, 4],
    ]

    total_fitness = 1.0

    for board_size in board_sizes:
        board_size_x, board_size_y = board_size
        descriptors = get_layers_descriptors(board_size_x, board_size_y)
        net = HyperNetwork(layers=descriptors, cppn=get_weights)

        fitness = get_fitness_parallel(
            get_net_moves_using_min_max,
            games_count=16,
            count_of_minimal_scores_as_fitness=2,
            board_size_x=board_size_x,
            board_size_y=board_size_y,
        )
        total_fitness *= fitness + 1

    return log2(total_fitness) * 100
