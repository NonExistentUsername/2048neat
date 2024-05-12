from math import log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class EmptyCellsFitness(IGameFitness):
    def fitness(self, game_stats: GameStats) -> float:
        return game_stats.total_empty_cells / (
            game_stats.game.size_x * game_stats.game.size_y * (game_stats.moves + 1)
        )
