from math import log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class GameScoreFitness(IGameFitness):
    def __init__(self, scale: bool = True, scale_factor: float = 0.03) -> None:
        self.scale = scale
        self.scale_factor = scale_factor

    def fitness(self, game_stats: GameStats) -> float:
        game = game_stats.game

        result = log2(game.score + 1)

        return result * self.scale_factor if self.scale else result
