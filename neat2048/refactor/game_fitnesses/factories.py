from typing import List

from neat2048.refactor.game_fitnesses.functions import *
from neat2048.refactor.game_fitnesses.functions.constants import *
from neat2048.refactor.game_fitnesses.interfaces import IGameFitness
from neat2048.refactor.game_fitnesses.tools import SumFitness


class GameFitnessFactory:
    @staticmethod
    def create(fitness_type: FitnessTypes, **kwargs) -> IGameFitness:
        if fitness_type == FitnessTypes.EMPTY_CELLS:
            return EmptyCellsFitness()
        elif fitness_type == FitnessTypes.MOVES_COUNT:
            return MovesCountFitness(**kwargs)
        elif fitness_type == FitnessTypes.ILLEGAL_MOVES:
            return IllegalMovesPenalty(**kwargs)
        elif fitness_type == FitnessTypes.GAME_SCORE:
            return GameScoreFitness()
        elif fitness_type == FitnessTypes.MAX_TILE:
            return MaxTileFitness()
        elif fitness_type == FitnessTypes.SMOOTHNESS:
            return SmoothnessFitness()
        elif fitness_type == FitnessTypes.LONG_FALLING:
            return LongFallingFitness()

        raise ValueError(f"Unknown fitness type: {fitness_type}")

    @staticmethod
    def create_sum(fitness_types: List[FitnessTypes], **kwargs) -> IGameFitness:
        return SumFitness(
            [GameFitnessFactory.create(fitness_type) for fitness_type in fitness_types]
        )
