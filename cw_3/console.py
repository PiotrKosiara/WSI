"""
Author: Grzegorz Kaczor
"""

from board import Board


class Validator:
    def __init__(self, min_value=None, max_value=None, board: Board = None):
        self.min_value = min_value
        self.max_value = max_value
        self.board = board

    def validator_any(self, text):
        return None

    def validator_int_in_range(self, text):
        assert self.min_value is not None
        assert self.max_value is not None
        try:
            int_value = int(text)
            if self.min_value <= int_value <= self.max_value:
                return None
            else:
                raise ValueError("Value out of range")
        except ValueError as e:
            return "Error: %s. Please input integer between %d and %d" % (
                e,
                self.min_value,
                self.max_value,
            )

    def validator_empty_field_index(self, text):
        assert self.board is not None
        self.min_value = 0
        self.max_value = self.board.size * self.board.size
        result = self.validator_int_in_range(text)
        if result:
            return result
        field_index = int(text)
        empty_indexes = self.board.empty_indexes()
        if not field_index in empty_indexes:
            return "Enter one of [%s]" % ",".join([str(i) for i in empty_indexes])
        return None


class Console:
    def __init__(self, board: Board):
        self.board = board

    def read_console_input(self, prompt: str, validator):
        while True:
            value = input("%s ? " % prompt)
            err_msg = validator(value)
            if err_msg:
                print(err_msg)
            else:
                return value

    def output(self, text):
        print(text)

    def read_any_input(self, prompt):
        return self.read_console_input(prompt, Validator().validator_any)

    def read_int_input(self, prompt, min, max):
        return int(
            self.read_console_input(
                prompt, Validator(min_value=min, max_value=max).validator_int_in_range
            )
        )

    def read_empty_board_index(self, prompt):
        return int(
            self.read_console_input(
                prompt, Validator(board=self.board).validator_empty_field_index
            )
        )
