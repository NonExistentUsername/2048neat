import abc
from typing import List, Tuple

from neat2048.refactor.game import Game2048


class IGamePredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, net, game: Game2048) -> List[Tuple[int, float]]:
        pass

    @abc.abstractmethod
    def predict_batch(
        self, net, games: List[Game2048]
    ) -> List[List[Tuple[int, float]]]:
        pass


class IGameInputConverter(abc.ABC):
    @abc.abstractmethod
    def convert(self, game: Game2048) -> List[float]:
        pass
