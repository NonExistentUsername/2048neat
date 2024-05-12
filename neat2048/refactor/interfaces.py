import abc
from typing import Callable

import neat  # type: ignore


class IFitnessFunction(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self, genome: neat.DefaultGenome, config: neat.Config, global_seed: int
    ) -> float:
        pass

    def __call__(
        self, genome: neat.DefaultGenome, config: neat.Config, global_seed: int
    ) -> float:
        return self.evaluate(genome, config, global_seed)


class INetCreator(abc.ABC):
    @abc.abstractmethod
    def create_net(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        board_size_x: int,
        board_size_y: int,
    ) -> Callable:
        pass
