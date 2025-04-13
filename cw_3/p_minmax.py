"""
Author: Katarzyna Nałęcz-Charkiewicz
"""

from board import Board
from player import Player


class MinMaxPlayer(Player):
    def __init__(self, name: str, depth_limit: int):
        # Konstruktor inicjalizujący gracza minimax
        # name - nazwa gracza (np. "x" lub "o")
        # depth_limit - maksymalna głębokość przeszukiwania drzewa gry
        super().__init__(name)
        self.depth_limit = depth_limit  # zapamiętanie limitu głębokości

    def make_move(self, board: Board, your_side: str):
        """
        Główna metoda podejmowania decyzji o ruchu przez gracza.
        Tworzy kopię planszy dla każdego możliwego ruchu, uruchamia funkcję minimax
        i wybiera ten ruch, który daje najwyższą ocenę.
        """
        best_score = float('-inf')  # startowa najlepsza ocena (dla maksymalizatora)
        best_move = None  # brak najlepszego ruchu na starcie

        for move in board.empty_indexes():  # dla każdego dostępnego pola na planszy
            new_board = board.clone()  # tworzę kopię planszy
            new_board.register_move(move)  # wykonuję ruch testowy
            # Uruchamiam minimax, przy czym zaczynam od strony przeciwnika,
            # bo pierwszy ruch został już wykonany
            score = self.minimax(new_board, self.opponent(your_side), 1,
                                 False, float('-inf'), float('inf'), your_side)
            # Jeżeli ocena ruchu jest lepsza niż poprzednia najlepsza
            if score > best_score:
                best_score = score  # aktualizuję najlepszą ocenę
                best_move = move  # zapamiętuję ruch

        return best_move  # zwracam optymalny ruch

    def minimax(self, board: Board, current_side: str, depth: int, maximizing: bool,
                alpha: float, beta: float, my_side: str):
        """
        Rekurencyjna implementacja algorytmu minimax z przycinaniem alfa-beta.
        current_side – strona, której ruch analizujemy w danym kroku
        depth – aktualna głębokość drzewa
        maximizing – True, jeśli to runda maksymalizująca (czyli moja), False – przeciwnika
        alpha – najlepsza znana wartość dla gracza maksymalizującego
        beta – najlepsza znana wartość dla gracza minimalizującego
        my_side – litera mojego gracza ('x' lub 'o')
        """
        # Warunek końca rekurencji:
        # jeśli gra zakończona lub nie ma więcej ruchów lub osiągnięto limit głębokości
        if board.who_is_winner() is not None or len(board.empty_indexes()) == 0 or depth >= self.depth_limit:
            return self.evaluate(board, my_side)  # oceniam stan planszy

        if maximizing:
            max_eval = float('-inf')  # maksymalna wartość startowa
            for move in board.empty_indexes():  # dla każdego możliwego ruchu
                new_board = board.clone()  # kopia planszy
                new_board.register_move(move)  # wykonuję ruch
                # rekurencyjne wywołanie dla przeciwnika, zmiana trybu na minimalizujący
                eval = self.minimax(new_board, self.opponent(current_side), depth + 1,
                                    False, alpha, beta, my_side)
                max_eval = max(max_eval, eval)  # aktualizacja najlepszego wyniku
                alpha = max(alpha, eval)  # aktualizacja alfa
                if beta <= alpha:  # przycinanie – dalsze analizy są zbędne
                    break
            return max_eval
        else:
            min_eval = float('inf')  # minimalna wartość startowa
            for move in board.empty_indexes():  # analogicznie do maksymalizatora
                new_board = board.clone()
                new_board.register_move(move)
                eval = self.minimax(new_board, self.opponent(current_side), depth + 1,
                                    True, alpha, beta, my_side)
                min_eval = min(min_eval, eval)  # aktualizacja najgorszego przypadku
                beta = min(beta, eval)  # aktualizacja beta
                if beta <= alpha:  # przycinanie
                    break
            return min_eval

    def evaluate(self, board: Board, my_side: str):
        """
        Funkcja oceny końcowego stanu planszy.
        Zwraca:
        +1 jeśli wygrałem
        -1 jeśli przegrałem
         0 jeśli remis lub gra nierozstrzygnięta
        """
        winner = board.who_is_winner()
        if winner == my_side:
            return 1
        elif winner == self.opponent(my_side):
            return -1
        else:
            return 0  # remis lub gra w toku (przy ograniczonej głębokości)

    def opponent(self, side: str):
        """
        Prosta funkcja pomocnicza zwracająca literę przeciwnika.
        Jeśli 'x', to przeciwnik to 'o' i odwrotnie.
        """
        return 'o' if side == 'x' else 'x'