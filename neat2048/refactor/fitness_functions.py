import random
from typing import List, Tuple

import neat  # type: ignore

from neat2048.refactor.game import Game2048
from neat2048.refactor.game_fitnesses.interfaces import GameStats, IGameFitness
from neat2048.refactor.interfaces import IFitnessFunction, INetCreator
from neat2048.refactor.predictions import IGamePredictor

GAMES_COUNT = 2
BOARD_SIZE_X = 4
BOARD_SIZE_Y = 4


class FitnessFunction(IFitnessFunction):
    def __init__(
        self,
        predictor: IGamePredictor,
        game_fitness: IGameFitness,
        net_creator: INetCreator,
        games_count: int = GAMES_COUNT,
        board_size_x: int = BOARD_SIZE_X,
        board_size_y: int = BOARD_SIZE_Y,
    ) -> None:
        self.predictor = predictor
        self.game_fitness = game_fitness
        self.net_creator = net_creator
        self.games_count = games_count
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y

    def play_game(self, net) -> float:
        game = Game2048(self.board_size_x, self.board_size_y)
        game.add_random_tile()  # Add first tile

        moves_count = 0
        illegal_moves_count = 0
        empty_cells_count = 0

        while not game.game_end:
            moves = self.predictor.predict(net, game)

            for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
                changed = game.move(move)
                if changed:
                    break
                else:
                    illegal_moves_count += 1

            empty_cells_count += game.empty_cells
            moves_count += 1

        return self.game_fitness.fitness(
            GameStats(game, moves_count, illegal_moves_count, empty_cells_count)
        )

    def evaluate(
        self, genome: neat.DefaultGenome, config: neat.Config, global_seed: int
    ) -> float:
        random.seed(global_seed)
        net = self.net_creator.create_net(
            genome, config, self.board_size_x, self.board_size_y
        )

        fitnesses = [self.play_game(net) for _ in range(self.games_count)]
        return sum(fitnesses) / len(fitnesses)


class ParallelFitnessFunction(IFitnessFunction):
    def __init__(
        self,
        predictor: IGamePredictor,
        game_fitness: IGameFitness,
        net_creator: INetCreator,
        games_count: int = GAMES_COUNT,
        board_size_x: int = BOARD_SIZE_X,
        board_size_y: int = BOARD_SIZE_Y,
    ) -> None:
        self.predictor = predictor
        self.game_fitness = game_fitness
        self.net_creator = net_creator
        self.games_count = games_count
        self.board_size_x = board_size_x
        self.board_size_y = board_size_y

    def evaluate(
        self, genome: neat.DefaultGenome, config: neat.Config, global_seed: int
    ) -> float:
        random.seed(global_seed)
        net = self.net_creator.create_net(
            genome, config, self.board_size_x, self.board_size_y
        )

        indexed_games: List[Tuple[int, Game2048]] = [
            (index, Game2048(self.board_size_x, self.board_size_y))
            for index in range(self.games_count)
        ]
        for _, game in indexed_games:
            game.add_random_tile()  # Add first tile

        fitnesses = []
        illegal_moves_count = [0 for _ in range(self.games_count)]
        total_empty_cells = [0 for _ in range(self.games_count)]

        moves_done = 0
        while indexed_games:
            games = [game for _, game in indexed_games]
            games_moves = self.predictor.predict_batch(net, games)

            for moves, index_game in zip(games_moves, indexed_games):
                index, game = index_game
                for move, _ in sorted(moves, key=lambda x: x[1], reverse=True):
                    changed = game.move(move)
                    if changed:
                        break
                    else:
                        illegal_moves_count[index] += 1

                total_empty_cells[index] += game.empty_cells

            moves_done += 1

            for index, game in indexed_games:
                if game.game_end:
                    indexed_games.remove((index, game))
                    fitnesses.append(
                        self.game_fitness.fitness(
                            GameStats(
                                game,
                                moves_done,
                                illegal_moves_count[index],
                                total_empty_cells[index],
                            )
                        )
                    )

        fitness = sum(fitnesses) / len(fitnesses)

        return max(0, fitness)  # to be sure that we don't have negative
