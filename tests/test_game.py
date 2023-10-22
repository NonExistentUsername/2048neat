import pytest

from neat2048.game import DOWN, LEFT, RIGHT, UP, Game2048


@pytest.fixture
def game_with_square_of_2_3x3_size():
    game = Game2048()

    for x in range(4):
        for y in range(4):
            if x < 3 and y < 3:
                game[x][y] = 2

    return game


@pytest.fixture
def game_with_square_of_8_4x4_size():
    game = Game2048()

    for x in range(4):
        for y in range(4):
            game[x][y] = 8

    return game


def test_game_creation():
    game = Game2048()

    assert game.size_x == 4
    assert game.size_y == 4
    assert game.score == 0

    assert game.board == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_move_up(game_with_square_of_2_3x3_size):
    game: Game2048 = game_with_square_of_2_3x3_size

    game.move(UP, add_random_tile=False)

    assert game.board == [[4, 4, 4, 0], [2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert game.score == 12

    game.move(UP, add_random_tile=False)  # Should not change anything

    assert game.board == [[4, 4, 4, 0], [2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert game.score == 12


def test_moves_4x4(game_with_square_of_8_4x4_size):
    game: Game2048 = game_with_square_of_8_4x4_size

    game.move(UP, add_random_tile=False)

    assert game.board == [
        [16, 16, 16, 16],
        [16, 16, 16, 16],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    game.move(UP, add_random_tile=False)

    assert game.board == [
        [32, 32, 32, 32],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    game.move(UP, add_random_tile=False)

    assert game.board == [
        [32, 32, 32, 32],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    game.move(LEFT, add_random_tile=False)

    assert game.board == [
        [64, 64, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    game.move(RIGHT, add_random_tile=False)

    assert game.board == [
        [0, 0, 0, 128],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_move_right(game_with_square_of_2_3x3_size):
    game: Game2048 = game_with_square_of_2_3x3_size

    game.move(RIGHT, add_random_tile=False)

    assert game.board == [[0, 0, 2, 4], [0, 0, 2, 4], [0, 0, 2, 4], [0, 0, 0, 0]]
    assert game.score == 12

    game.move(RIGHT, add_random_tile=False)  # Should not change anything

    assert game.board == [[0, 0, 2, 4], [0, 0, 2, 4], [0, 0, 2, 4], [0, 0, 0, 0]]
    assert game.score == 12


def test_move_down(game_with_square_of_2_3x3_size):
    game: Game2048 = game_with_square_of_2_3x3_size

    game.move(DOWN, add_random_tile=False)

    assert game.board == [[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 0], [4, 4, 4, 0]]
    assert game.score == 12

    game.move(DOWN, add_random_tile=False)  # Should not change anything

    assert game.board == [[0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 0], [4, 4, 4, 0]]
    assert game.score == 12


def test_move_left(game_with_square_of_2_3x3_size):
    game: Game2048 = game_with_square_of_2_3x3_size

    game.move(LEFT, add_random_tile=False)

    assert game.board == [[4, 2, 0, 0], [4, 2, 0, 0], [4, 2, 0, 0], [0, 0, 0, 0]]
    assert game.score == 12

    game.move(LEFT, add_random_tile=False)  # Should not change anything

    assert game.board == [[4, 2, 0, 0], [4, 2, 0, 0], [4, 2, 0, 0], [0, 0, 0, 0]]
    assert game.score == 12


def test_score():
    game = Game2048()

    assert game.score == 0

    game[0][0] = 16
    game[1][0] = 16

    game.move(UP, add_random_tile=False)

    assert game.score == 32


def test_game_end(game_with_square_of_2_3x3_size):
    game: Game2048 = game_with_square_of_2_3x3_size

    assert not game.game_end

    game.move(UP, add_random_tile=False)

    assert not game.game_end

    game: Game2048 = game_with_square_of_2_3x3_size

    for x in range(4):
        for y in range(4):
            game[y][x] = x * 4 + y + 1

    assert game.game_end
