"""
Author: Grzegorz Kaczor
"""

from typing import List
import logging
import time

import player
import board


class Engine:
    def __init__(self, board: board.Board, players: List[player.Player]):
        self.board = board
        self.players = players
        self.time_per_player_ms = {}
        for p in players:
            assert not p.name in self.time_per_player_ms
            self.time_per_player_ms[p.name] = 0

    def play_next_step(self):
        player = self.players[self.board.playing_now_idx]
        logging.info("Moving: %s" % player.description())
        move_start = time.time()
        move = player.make_move(
            self.board, self.board.player_chars[self.board.playing_now_idx]
        )
        move_end = time.time()
        self.time_per_player_ms[player.name] += (move_end - move_start) * 1000.0
        logging.info("Player selected %d" % move)
        if not move in self.board.empty_indexes():
            err_msg = "Player %s did an invalid move %d, could select one of: [%s]"(
                player.description(), move, self.board.empty_indexes()
            )
            logging.error(err_msg)
            raise ValueError(err_msg)
        self.board.register_move(move)
        logging.info("Move done.")
        return self.board.who_is_winner()
