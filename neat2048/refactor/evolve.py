import multiprocessing
import os
import pickle
import random

import neat  # type: ignore

from neat2048.refactor.evaluators import ParallelEvaluatorWithRandomSeed
from neat2048.refactor.fitness_functions import FitnessFunction, ParallelFitnessFunction
from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses import (
    EmptyCellsFitness,
    FitnessTypes,
    GameFitnessFactory,
    GameScoreFitness,
    GameStats,
    IGameFitness,
    MaxTileFitness,
    Multiplier,
    SumFitness,
)
from neat2048.refactor.interfaces import IFitnessFunction
from neat2048.refactor.nets import HyperNetCreator, NEATNetCreator
from neat2048.refactor.predictions import (
    IGamePredictor,
    Log2ScaledConverter,
    MinMaxNetPredictor,
    NetPredictor,
)

# CONFIG
ENABLE_HYPER = True
ENABLE_FITNESS_OPTIMIZED = False
GENERATIONS = 200


def get_config_path() -> str:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    if ENABLE_HYPER:
        config_path += "-hyper"
    return config_path


def get_game_fitness() -> IGameFitness:
    # return Multiplier(MaxTileFitness(scale=False), 10)

    return SumFitness(
        fitnesses=[
            GameScoreFitness(scale=False),
            MaxTileFitness(scale=False),
            Multiplier(EmptyCellsFitness(), 5),
        ],
        scale=True,
    )


def build_evaluator() -> ParallelEvaluatorWithRandomSeed:
    num_workers = multiprocessing.cpu_count()

    if ENABLE_HYPER:
        return ParallelEvaluatorWithRandomSeed(
            num_workers,
            ParallelFitnessFunction(
                predictor=MinMaxNetPredictor(
                    converter=Log2ScaledConverter(),
                    depth=2,
                ),
                net_creator=HyperNetCreator(),
                game_fitness=get_game_fitness(),
                games_count=12,
                board_size_x=3,
                board_size_y=3,
            ),
        )

    return ParallelEvaluatorWithRandomSeed(
        num_workers,
        FitnessFunction(
            predictor=NetPredictor(
                converter=Log2ScaledConverter(),
            ),
            net_creator=NEATNetCreator(),
            game_fitness=get_game_fitness(),
            games_count=8,
            board_size_x=4,
            board_size_y=4,
        ),
    )


def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    print(f"Initializing finished. Running for {GENERATIONS} generations.")

    pe = build_evaluator()
    winner = p.run(pe.evaluate, GENERATIONS)

    print("Best genome:\n{!s}".format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    with open("output_network.pkl", "wb") as f:
        pickle.dump(winner_net, f)

    print("Output network saved to output_network.pkl")

    stats.save()

    print("\nStats saved to fitness_history.csv")


if __name__ == "__main__":
    run(get_config_path())
