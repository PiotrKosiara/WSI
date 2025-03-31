"""
Author: Grzegorz Kaczor
"""

from __future__ import annotations


class Board:
    def __init__(self, size: int):
        assert size > 0
        self.size = size
        self.char_circle = "o"
        self.char_cross = "x"
        self.player_chars = [self.char_circle, self.char_cross]
        self.char_empty = " "
        self.playing_now_idx = 0
        self.board = [self.char_empty for _ in range(self.size * self.size)]

    def clone(self) -> Board:
        board = Board(self.size)
        board.board = self.board.copy()
        board.playing_now_idx = self.playing_now_idx
        return board

    def print_board(self):
        horizontal_row = " " + "-" * self.size + " \n"
        left_border = "|"
        right_border = "|\n"
        output = horizontal_row
        (row, column) = (0, 0)
        for c in self.board:
            if column == 0:
                output += left_border
            output += c
            if column == self.size - 1:
                output += right_border
                row += 1
                column = 0
            else:
                column += 1
        output += horizontal_row
        return output

    def _check_line(self, start, step):
        single_c = None
        point = start
        while 0 <= point[0] < self.size and 0 <= point[1] < self.size:
            char_at = self.board[point[0] * self.size + point[1]]
            if char_at != self.char_circle and char_at != self.char_cross:
                return None
            if not single_c:
                single_c = char_at
            else:
                if single_c != char_at:
                    return None
            point = (point[0] + step[0], point[1] + step[1])
        return single_c

    def who_is_winner(self):
        lines_to_check = []
        lines_to_check.append(((0, 0), (1, 1)))
        lines_to_check.append(((self.size - 1, 0), (-1, 1)))
        # add vertical and horizontal lines
        for i in range(0, self.size):
            lines_to_check.append(((i, 0), (0, 1)))
            lines_to_check.append(((0, i), (1, 0)))
        for (start, step) in lines_to_check:
            winner = self._check_line(start, step)
            if winner:
                return winner
        return None

    def empty_indexes(self):
        indexes = []
        for index, char_at in enumerate(self.board):
            if char_at == self.char_empty:
                indexes.append(index)
        return indexes

    def register_move(self, field_index: int):
        assert 0 <= field_index < self.size * self.size
        char_at = self.board[field_index]
        if char_at != self.char_empty:
            raise ValueError(
                "Player %s attempted to overwrite %s at index %d [%s]"
                % (
                    self.player_chars[self.playing_now_idx],
                    char_at,
                    field_index,
                    self.print_board(),
                )
            )
        self.board[field_index] = self.player_chars[self.playing_now_idx]
        self.playing_now_idx += 1
        if self.playing_now_idx >= len(self.player_chars):
            self.playing_now_idx = 0
        return self.player_chars[self.playing_now_idx]
