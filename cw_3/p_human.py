"""
Author: Grzegorz Kaczor
"""

from board import Board
from player import Player


class HumanPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name)

    def make_move(self, board: Board, your_side: str):
        return self.console.read_empty_board_index(
            "Select one of fields: %s" % board.empty_indexes()
        )
