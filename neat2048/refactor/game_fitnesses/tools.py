from typing import List

from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness


class SumFitness(IGameFitness):
    def __init__(self, fitnesses: List[IGameFitness], scale: bool = True) -> None:
        self.fitnesses = fitnesses
        self.scale = scale

    def fitness(self, game_stats: GameStats) -> float:
        summary = sum(fitness.fitness(game_stats) for fitness in self.fitnesses)

        if self.scale:
            summary /= len(self.fitnesses)

        return summary


class SumFitnessAndPenalties(IGameFitness):
    def __init__(
        self,
        fitnesses: List[IGameFitness],
        penalties: List[IGameFitness],
        scale: bool = True,
    ) -> None:
        self.__sum_fitness = SumFitness(fitnesses, scale)
        self.__sum_penalties = SumFitness(penalties, scale)

    def fitness(self, game_stats: GameStats) -> float:
        return self.__sum_fitness.fitness(game_stats) - self.__sum_penalties.fitness(
            game_stats
        )


class Multiplier(IGameFitness):
    def __init__(self, game_fitness: IGameFitness, multiplier: float) -> None:
        self.game_fitness = game_fitness
        self.multiplier = multiplier

    def fitness(self, game_stats: GameStats) -> float:
        return self.game_fitness.fitness(game_stats) * self.multiplier
