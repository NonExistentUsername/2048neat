from math import log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class IllegalMovesPenalty(IGameFitness):
    def __init__(self, scale: bool = True, scale_factor: float = 0.0003) -> None:
        self.scale = scale
        self.scale_factor = scale_factor

    def fitness(self, game_stats: GameStats) -> float:
        return (
            game_stats.illegal_moves * self.scale_factor
            if self.scale
            else game_stats.illegal_moves
        )
