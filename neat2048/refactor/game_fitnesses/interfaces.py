import abc
from dataclasses import dataclass

from neat2048.refactor.game import Game2048


@dataclass
class GameStats:
    game: Game2048
    moves: int
    illegal_moves: int
    total_empty_cells: int


class IGameFitness(abc.ABC):
    @abc.abstractmethod
    def fitness(self, game_stats: GameStats) -> float:
        pass
