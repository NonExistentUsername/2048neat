from typing import Callable, List, Tuple

import neat  # type: ignore

from neat2048.refactor.hyper import HyperNetwork
from neat2048.refactor.interfaces import INetCreator


def get_layers_descriptors(board_size_x, board_size_y) -> List[Tuple[int, int]]:
    return [
        (board_size_x, board_size_y),
        (12, 12),
        (1, 1),  # for min-max
        # (2, 2),  # for move prediction
    ]


class NEATNetCreator(INetCreator):
    def create_net(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        board_size_x: int,
        board_size_y: int,
    ) -> Callable:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        def get_weights(inputs: list[float]) -> list[float]:
            return net.activate(inputs)

        return get_weights


class HyperNetCreator(INetCreator):
    def create_net(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        board_size_x: int,
        board_size_y: int,
    ) -> Callable:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)

        def get_weights(inputs: list[float]) -> list[float]:
            return cppn.activate(inputs)

        return HyperNetwork(
            get_layers_descriptors(board_size_x, board_size_y), get_weights
        )
