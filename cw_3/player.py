"""
Author: Grzegorz Kaczor
"""

import board
from console import Console


class Player:
    def __init__(self, name: str):
        self.name = name
        self.player_class = type(self).__name__

    def bind_console(self, console: Console):
        self.console = console

    def description(self):
        return "P[%s][%s]" % (self.name, self.player_class)

    def make_move(self, board: board.Board, your_side: str):
        raise NotImplementedError("make_move should be implemented in the subclass")
