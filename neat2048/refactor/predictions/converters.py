from math import log2
from typing import List

from neat2048.refactor.predictions.interfaces import IGameInputConverter


class NormallyScaledConverter(IGameInputConverter):
    def convert(self, game) -> List[float]:
        cells = [cell for row in game.board for cell in row]
        max_cell = max(cells)
        return [cell / max_cell for cell in cells]


class Log2ScaledConverter(IGameInputConverter):
    def convert(self, game) -> List[float]:
        cells = [cell for row in game.board for cell in row]
        return [log2(cell + 1) for cell in cells]
