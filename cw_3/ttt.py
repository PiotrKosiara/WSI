#!/usr/bin/python3

"""
Author: Grzegorz Kaczor
Date: March 25, 2023

Tic-Tac-Toe Implementation

This program provides a simple implementation of the classic game Tic-Tac-Toe with a text-based user interface.

The program is provided for the purpose of completing assignment 3 in the subject of WSI.

"""

import logging
import sys

import board
import engine
import player
import p_minmax
import p_human
import p_random
from console import Console


class Game:
    def __init__(self, board_size: int, player1: player.Player, player2: player.Player):
        self.board = board.Board(board_size)
        self.players = [player1, player2]
        self.show_board = True
        self.suspend_after_move = False
        self.max_turns = board_size * board_size
        self.console = Console(self.board)

    def run_game(self):
        logging.info(
            "Starting running of TTT console for players: [%s]"
            % ",".join([p.description() for p in self.players])
        )
        play_engine = engine.Engine(self.board, self.players)
        result = None
        turn_no = 0
        if self.show_board:
            self.console.output(
                "\nTurn %d moving: %s, player %s"
                % (
                    turn_no,
                    self.board.player_chars[self.board.playing_now_idx],
                    self.players[self.board.playing_now_idx].description(),
                )
            )
            self.console.output(self.board.print_board())
        while not result:
            turn_no += 1
            if turn_no > self.max_turns:
                self.console.output("RESULT:DRAW")
                break
            result = play_engine.play_next_step()
            if self.show_board:
                self.console.output(
                    "\nTurn %d moving: %s, player %s"
                    % (
                        turn_no,
                        self.board.player_chars[self.board.playing_now_idx],
                        self.players[self.board.playing_now_idx].description(),
                    )
                )
                self.console.output(self.board.print_board())
            if result:
                self.console.output(
                    "RESULT:%s:%s"
                    % (
                        result,
                        self.players[
                            self.board.player_chars.index(result)
                        ].description(),
                    )
                )
                break
            if self.suspend_after_move:
                self.console.read_any_input("Press ENTER")
        self.console.output("Time played: %s" % play_engine.time_per_player_ms)


def build_player(player_type: str, player_name: str, depth_limit):
    if player_type == "human":
        return p_human.HumanPlayer(player_name)
    if player_type == "random":
        return p_random.RandomPlayer(player_name)
    if player_type == "minmax":
        return p_minmax.MinMaxPlayer(player_name, depth_limit=depth_limit)
    raise ValueError("Unknown player type: %s" % player_type)


if __name__ == "__main__":
    board_size = 3
    logging.getLogger().setLevel(logging.INFO)
    print(sys.argv)
    player1_type = sys.argv[1] if len(sys.argv) > 1 else "random"
    player2_type = sys.argv[2] if len(sys.argv) > 2 else "random"
    depth_limit = 9 if board_size == 3 else 6
    player1 = build_player(player1_type, "o", depth_limit)
    player2 = build_player(player2_type, "x", depth_limit)
    game = Game(board_size, player1, player2)
    player1.bind_console(game.console)
    player2.bind_console(game.console)
    game.suspend_after_move = False
    game.run_game()
