import itertools
import random
from copy import deepcopy

import numpy as np

MOVES = [UP, DOWN, RIGHT, LEFT] = range(4)

ADD_4_PROBABILITY = 0.0  # Set to 0.1 for 2048, 0.0 for simpler version of game


class Game2048:
    def __init__(self, size_x: int = 4, size_y: int = 4) -> None:
        self.__size_x = size_x
        self.__size_y = size_y
        self.__score = 0
        self.__board = [[0 for _ in range(size_x)] for _ in range(size_y)]

    @property
    def size_x(self) -> int:
        return self.__size_x

    @property
    def size_y(self) -> int:
        return self.__size_y

    @property
    def score(self) -> int:
        return self.__score

    @property
    def board(self) -> list[list[int]]:
        return deepcopy(self.__board)

    def copy(self) -> "Game2048":
        game = Game2048(self.__size_x, self.__size_y)
        game.__board = deepcopy(self.__board)
        game.__score = self.__score
        return game

    def __hash__(self) -> int:
        p = 1
        k = 31
        mod = 10**9 + 7
        result = 0
        for x, y in itertools.product(range(self.__size_x), range(self.__size_y)):
            result = (result + self.__board[x][y] * p) % mod
            p *= k
        return result

    def __check_if_move_legal(self, move) -> bool:
        board = deepcopy(self.__board)
        game = Game2048(self.__size_x, self.__size_y)
        game.__board = board
        game.move(move, add_random_tile=False)
        if game.__board != self.__board:
            return True
        return False

    @property
    def legal_moves(self) -> list[int]:
        legal_moves = []
        for move in MOVES:
            if self.__check_if_move_legal(move):
                legal_moves.append(move)
        return legal_moves

    @property
    def empty_cells(self) -> int:
        empty_cells = 0
        for row in self.__board:
            empty_cells += row.count(0)

        return empty_cells

    @property
    def game_end(self) -> bool:
        for move in MOVES:
            if self.__check_if_move_legal(move):
                return False
        return True

    def __str__(self) -> str:
        return str(self.__board)

    def __repr__(self) -> str:
        # Get printable version of board
        board = [[str(tile) for tile in row] for row in self.__board]

        # Get length of longest tile
        column_widths = [max(map(len, column)) for column in zip(*board)]
        board_width = sum(column_widths) + len(column_widths) + 1
        horizontal_line = " " * board_width

        string = ""
        for index, row in enumerate(board):
            row_str = " ".join(
                tile.rjust(column_widths[i], " ") for i, tile in enumerate(row)
            )
            string += horizontal_line + "\n"
            string += " " + row_str + " \n"
        string += horizontal_line + "\n"

        return string

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.__board[key]

    def __setitem__(
        self, key: tuple[int, int], value: int
    ) -> None:  # Only for testing purposes
        self.__board[key[0]][key[1]] = value

    def __move_horizontal(self, dx: int, start_l: int, start_r: int) -> None:
        for y in range(self.__size_y):
            l, r = start_l, start_r
            while 0 <= r < self.__size_x:
                if self.__board[y][r] == 0:
                    # skip empty tiles
                    r += dx
                    continue

                if self.__board[y][l] == self.__board[y][r]:
                    # collapse two tiles
                    self.__board[y][l] *= 2
                    self.__score += self.__board[y][l]

                    r += dx
                    l += dx

                    self.__board[y][l] = 0  # clear l, so it can be used
                    continue

                if self.__board[y][l] == 0:
                    # move r to l
                    self.__board[y][l] = self.__board[y][r]
                    r += dx
                    continue

                # move l and place r on l
                l += dx
                self.__board[y][l] = self.__board[y][r]

                r += dx

            # clear rest of the row
            l += dx
            while l != r:
                self.__board[y][l] = 0
                l += dx

    def __move_vertical(self, dy: int, start_u: int, start_d: int) -> None:
        for x in range(self.__size_x):
            u, d = start_u, start_d
            while 0 <= d < self.__size_y:
                if self.__board[d][x] == 0:
                    # skip empty tiles
                    d += dy
                    continue

                if self.__board[u][x] == self.__board[d][x]:
                    # collapse two tiles
                    self.__board[u][x] *= 2
                    self.__score += self.__board[u][x]

                    d += dy
                    u += dy

                    self.__board[u][x] = 0
                    continue

                if self.__board[u][x] == 0:
                    # move d to u
                    self.__board[u][x] = self.__board[d][x]
                    d += dy
                    continue

                # move u and place d on u
                u += dy
                self.__board[u][x] = self.__board[d][x]

                d += dy

            # clear rest of the column
            u += dy
            while u != d:
                self.__board[u][x] = 0
                u += dy

    def __move_left(self) -> None:
        self.__move_horizontal(1, 0, 1)

    def __move_right(self) -> None:
        self.__move_horizontal(-1, self.__size_x - 1, self.__size_x - 2)

    def __move_up(self) -> None:
        self.__move_vertical(1, 0, 1)

    def __move_down(self) -> None:
        self.__move_vertical(-1, self.__size_y - 1, self.__size_y - 2)

    def move(self, move, add_random_tile: bool = True) -> bool:
        if move not in MOVES:
            raise ValueError("Invalid move")

        board_copy = hash(self)

        if move == UP:
            self.__move_up()
        elif move == RIGHT:
            self.__move_right()
        elif move == DOWN:
            self.__move_down()
        elif move == LEFT:
            self.__move_left()

        if board_copy != hash(self):
            if add_random_tile:
                self.add_random_tile()
            return True

        return False

    def add_random_tile(self) -> None:
        empty_tiles = []
        for y in range(self.__size_y):
            for x in range(self.__size_x):
                if self.__board[y][x] == 0:
                    empty_tiles.append((y, x))

        if not empty_tiles:
            return

        tile = random.choice(empty_tiles)

        value = 4 if random.random() < ADD_4_PROBABILITY else 2

        self.__board[tile[0]][tile[1]] = value
