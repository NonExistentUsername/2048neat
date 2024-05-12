import random
from typing import List, Tuple

from neat2048.refactor.game import Game2048
from neat2048.refactor.predictions.interfaces import IGameInputConverter, IGamePredictor


class RandomPredictor(IGamePredictor):
    def predict(self, net, game) -> List[Tuple[int, float]]:
        possible_moves = list(range(4))

        moves = [(move, 1 / len(possible_moves)) for move in possible_moves]

        random.shuffle(moves)

        return moves


class NetPredictor(IGamePredictor):
    def __init__(self, converter: IGameInputConverter) -> None:
        self.converter = converter

    def predict(self, net, game) -> List[Tuple[int, float]]:
        inputs = self.converter.convert(game)
        return net(inputs)

    def predict_batch(self, net, games) -> List[List[Tuple[int, float]]]:
        return [self.predict(net, game) for game in games]


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


class MinMaxNetPredictor(IGamePredictor):
    def __init__(self, converter: IGameInputConverter, depth: int = 1) -> None:
        self.converter = converter
        self.depth = depth
        self.aggregate_function = max

    def evaluate(self, game: Game2048) -> float:
        inputs = self.converter.convert(game)
        return self.evaluation_net([inputs])[0]

    def generate_boards(self, game: Game2048) -> List[Game2048]:
        boards: List[Game2048] = []
        for move in range(4):
            new_game = game.copy()
            new_game.move(move, add_random_tile=False)
            boards.append(new_game)

        return boards

    def generate_boards_in_depth(self, game: Game2048, depth: int) -> List[Game2048]:
        if depth == 0:
            return [game]

        new_games = []
        for move in range(4):
            new_game = game.copy()
            new_game.move(move, add_random_tile=False)
            new_games.extend(self.generate_boards_in_depth(new_game, depth - 1))

        return new_games

    def merge_predictions(
        self, predictions: List[float], depth: int, games_count: int
    ) -> List[float]:
        if depth == 0:
            return predictions

        merged_predictions = []

        predictions_per_move = len(predictions) // 4

        for move in range(4):
            temp_merged_predictions = self.merge_predictions(
                predictions[
                    move * predictions_per_move : (move + 1) * predictions_per_move
                ],
                depth - 1,
                games_count,
            )
            merged_predictions.append(self.aggregate_function(temp_merged_predictions))

        return merged_predictions

    def minmax_evaluate(self, game: Game2048, depth: int) -> float:
        if depth == 0:
            return self.evaluate(game)

        scores = []
        boards = self.generate_boards(game)
        for board in boards:
            score = self.minmax_evaluate(board, depth - 1)
            scores.append(score)

        return self.aggregate_function(scores)

    def predict(self, net, game) -> List[Tuple[int, float]]:
        self.evaluation_net = net

        moves = []

        for move in range(4):
            new_game = game.copy()
            new_game.move(move, add_random_tile=False)
            score = self.minmax_evaluate(new_game, self.depth - 1)
            moves.append((move, score))

        return moves

    def predict_batch(self, net, games) -> List[List[Tuple[int, float]]]:
        games_extended_in_depth = []
        for game in games:
            games_extended_in_depth.extend(
                self.generate_boards_in_depth(game, self.depth)
            )

        converted_games = [
            self.converter.convert(game) for game in games_extended_in_depth
        ]

        predictions = net(converted_games)
        predictions_per_game = len(predictions) // len(games)

        merged_predictions = []
        for i, game in enumerate(games):
            game_predictions = predictions[
                i * predictions_per_game : (i + 1) * predictions_per_game
            ]
            tmp_merged_predictions = self.merge_predictions(
                game_predictions, self.depth, predictions_per_game
            )
            merged_predictions.append(tmp_merged_predictions)

        return [list(enumerate(prediction)) for prediction in merged_predictions]
