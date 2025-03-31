"""
Author: Grzegorz Kaczor
"""

import random

from board import Board
from player import Player


class RandomPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name)

    def make_move(self, board: Board, your_side: str):
        empty_indexes = board.empty_indexes()
        return empty_indexes[random.randrange(len(empty_indexes))]
