import itertools

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


def process_long_falling_from_tile(game: Game2048, x: int, y: int) -> int:
    # Min value is 2, max is 4*4 = 16 or board_size_x * board_size_y

    best_neighbour_positions = []
    best_neighbour_value = -1

    is_valid = lambda x, y: 0 <= x < game.size_x and 0 <= y < game.size_y

    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        new_x, new_y = x + dx, y + dy

        if not is_valid(new_x, new_y):
            continue

        value = game.board[new_x][new_y]
        if value > best_neighbour_value and value < game.board[x][y]:
            best_neighbour_value = value
            best_neighbour_positions = [(new_x, new_y)]
        elif value == best_neighbour_value:
            best_neighbour_positions.append((new_x, new_y))

    best_val = 1
    for x, y in best_neighbour_positions:
        best_val = max(best_val, 1 + process_long_falling_from_tile(game, x, y))

    return best_val


def find_max_path_from_tiles(game: Game2048) -> int:
    # Min value is 2, max is 4*4 = 16 or board_size_x * board_size_y
    max_tile_value = 0
    for x, y in itertools.product(range(game.size_x), range(game.size_y)):
        if game.board[x][y] > max_tile_value:
            max_tile_value = game.board[x][y]

    longest_path = 0
    for x, y in itertools.product(range(game.size_x), range(game.size_y)):
        if game.board[x][y] == max_tile_value:
            longest_path = max(longest_path, process_long_falling_from_tile(game, x, y))

    return longest_path


class LongFallingFitness(IGameFitness):
    def fitness(self, game_stats: GameStats) -> float:
        game = game_stats.game

        return find_max_path_from_tiles(game) / (game.size_x * game.size_y)
