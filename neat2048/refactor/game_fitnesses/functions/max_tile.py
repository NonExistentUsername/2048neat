from math import log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class MaxTileFitness(IGameFitness):
    def __init__(self, scale: bool = True, scale_factor: float = 0.4) -> None:
        self.scale = scale
        self.scale_factor = scale_factor

    def get_log2_max_possible_tile(self, size_x: int, size_y: int) -> int:
        return size_x * size_y

    def fitness(self, game_stats: GameStats) -> float:
        game = game_stats.game

        result = log2(
            max(max(row) for row in game.board)
        ) / self.get_log2_max_possible_tile(game.size_x, game.size_y)

        return result * self.scale_factor if self.scale else result
