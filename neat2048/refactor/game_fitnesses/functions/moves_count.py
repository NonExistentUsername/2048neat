from math import log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class MovesCountFitness(IGameFitness):
    def __init__(self, scale: bool = True, scale_factor: float = 0.002) -> None:
        self.scale = scale
        self.scale_factor = scale_factor

    def fitness(self, game_stats: GameStats) -> float:
        result = log2(game_stats.moves + 1)

        return result * self.scale_factor if self.scale else result
