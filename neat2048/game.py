import itertools
import random
from copy import deepcopy

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

    def __move_left(self) -> None:
        collected_tiles = [[] for _ in range(self.__size_y)]

        # Collect tiles
        for y in range(self.__size_y):
            for x in range(self.__size_x):
                if self.__board[y][x] == 0:
                    continue
                # Add current tile to collected tiles
                # but if the last collected tile is the same as the current tile
                # then merge them
                if collected_tiles[y] and collected_tiles[y][-1] == self.__board[y][x]:
                    collected_tiles[y][-1] *= 2
                    self.__score += collected_tiles[y][-1]
                else:
                    collected_tiles[y].append(self.__board[y][x])

        # Fill board with collected tiles
        for y in range(self.__size_y):
            collected_tiles[y].reverse()

            for x in range(self.__size_x):
                if collected_tiles[y]:
                    tile = collected_tiles[y].pop()
                    self.__board[y][x] = tile
                else:
                    self.__board[y][x] = 0

    def __move_right(self) -> None:
        collected_tiles = [[] for _ in range(self.__size_y)]

        # Collect tiles
        for y in range(self.__size_y):
            for x in range(self.__size_x - 1, -1, -1):
                if self.__board[y][x] == 0:
                    continue
                # Add current tile to collected tiles
                # but if the last collected tile is the same as the current tile
                # then merge them
                if collected_tiles[y] and collected_tiles[y][-1] == self.__board[y][x]:
                    collected_tiles[y][-1] *= 2
                    self.__score += collected_tiles[y][-1]
                else:
                    collected_tiles[y].append(self.__board[y][x])

        # Fill board with collected tiles
        for y in range(self.__size_y):
            collected_tiles[y].reverse()

            for x in range(self.__size_x - 1, -1, -1):
                if collected_tiles[y]:
                    tile = collected_tiles[y].pop()
                    self.__board[y][x] = tile
                else:
                    self.__board[y][x] = 0

    def __move_up(self) -> None:
        collected_tiles = [[] for _ in range(self.__size_x)]

        # Collect tiles
        for x in range(self.__size_x):
            for y in range(self.__size_y):
                if self.__board[y][x] == 0:
                    continue
                # Add current tile to collected tiles
                # but if the last collected tile is the same as the current tile
                # then merge them
                if collected_tiles[x] and collected_tiles[x][-1] == self.__board[y][x]:
                    collected_tiles[x][-1] *= 2
                    self.__score += collected_tiles[x][-1]
                else:
                    collected_tiles[x].append(self.__board[y][x])

        # Fill board with collected tiles
        for x in range(self.__size_x):
            collected_tiles[x].reverse()

            for y in range(self.__size_y):
                if collected_tiles[x]:
                    tile = collected_tiles[x].pop()
                    self.__board[y][x] = tile
                else:
                    self.__board[y][x] = 0

    def __move_down(self) -> None:
        collected_tiles = [[] for _ in range(self.__size_x)]

        # Collect tiles
        for x in range(self.__size_x):
            for y in range(self.__size_y - 1, -1, -1):
                if self.__board[y][x] == 0:
                    continue
                # Add current tile to collected tiles
                # but if the last collected tile is the same as the current tile
                # then merge them
                if collected_tiles[x] and collected_tiles[x][-1] == self.__board[y][x]:
                    collected_tiles[x][-1] *= 2
                    self.__score += collected_tiles[x][-1]
                else:
                    collected_tiles[x].append(self.__board[y][x])

        # Fill board with collected tiles
        for x in range(self.__size_x):
            collected_tiles[x].reverse()

            for y in range(self.__size_y - 1, -1, -1):
                if collected_tiles[x]:
                    tile = collected_tiles[x].pop()
                    self.__board[y][x] = tile
                else:
                    self.__board[y][x] = 0

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
