import itertools
from math import fabs, log2

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


# Smoothness is the sum of all: differences between two non-zero tiles / the smaller tile
def calc_smoothness(game: Game2048, board_size_x=4, board_size_y=4):
    smoothness = 0.0
    # Only rotate twice to avoid double counting
    # Ignore 0 tiles
    board_size_x = len(game.board)
    board_size_y = len(game.board[0])

    for i, j in itertools.product(range(board_size_x), range(board_size_y)):
        if game.board[i][j] != 0 and j + 1 < board_size_y and game.board[i][j + 1] != 0:
            current_smoothness = fabs(
                log2(game.board[i][j]) - log2(game.board[i][j + 1])
            )
            smoothness = smoothness - current_smoothness

    for j, i in itertools.product(range(board_size_y), range(board_size_x)):
        if game.board[i][j] != 0 and i + 1 < board_size_x and game.board[i + 1][j] != 0:
            current_smoothness = fabs(
                log2(game.board[i][j]) - log2(game.board[i + 1][j])
            )
            smoothness = smoothness - current_smoothness

    return smoothness


class SmoothnessFitness(IGameFitness):
    def fitness(self, game_stats: GameStats) -> float:
        game = game_stats.game

        return calc_smoothness(game)
