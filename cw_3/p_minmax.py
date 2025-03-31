"""
Author: Katarzyna Nałęcz-Charkiewicz
"""

from board import Board
from player import Player


class MinMaxPlayer(Player):
    def __init__(self, name: str, depth_limit: int):
        super().__init__(name)
        self.depth_limit = depth_limit

    def make_move(self, board: Board, your_side: str):
        # TODO
        return None

    def minimax(self, board: Board, side: str, depth: int):
        # TODO
        return None, None

    def evaluate(self, board: Board, side: str):
        # TODO
        return 0
