import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import os
import random

#Pobranie danych:
tic_tac_toe = fetch_ucirepo(id=101)        # Pobiera zbiór danych Tic-Tac-Toe Endgame z repozytorium UCI
X = tic_tac_toe.data.features              # Zapisuje cechy (wszystkie pola planszy) jako DataFrame
y = tic_tac_toe.data.targets.iloc[:, 0]   # Wybiera kolumnę z etykietą klasy ('positive' lub 'negative')


#Podział danych:
def manual_split(X, y, test_ratio=0.2, val_ratio=0.25, seed=42):
    '''
    Dzieli dane na zbiory treningowy, walidacyjny i testowy.

    Parametry:
        X (pd.DataFrame): Dane wejściowe (cechy).
        y (pd.Series): Etykiety klas.
        test_ratio (float): Proporcja danych testowych względem całego zbioru.
        val_ratio (float): Proporcja danych walidacyjnych względem danych treningowych+walidacyjnych.
        seed (int): Ziarno generatora losowego dla powtarzalności wyników.

    Zwraca:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    '''
    np.random.seed(seed)  # Ustawienie ziarna losowego dla powtarzalności wyników
    indices = np.arange(len(X))  # Tworzy listę indeksów wszystkich próbek
    np.random.shuffle(indices)  # Losowo miesza indeksy

    test_size = int(len(X) * test_ratio)  # Liczy liczbę próbek do zbioru testowego
    val_size = int((len(X) - test_size) * val_ratio)  # Liczy liczbę próbek do zbioru walidacyjnego

    test_idx = indices[:test_size]  # Wybiera indeksy do zbioru testowego
    val_idx = indices[test_size:test_size + val_size]  # Indeksy do zbioru walidacyjnego
    train_idx = indices[test_size + val_size:]  # Reszta trafia do zbioru treningowego

    # Zwraca podzielone dane jako sześć elementów: cechy i etykiety dla trenującego, walidacyjnego i testowego
    return (
        X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    )

# Wywołanie funkcji i przypisanie wyników do odpowiednich zmiennych
X_train, X_val, X_test, y_train, y_val, y_test = manual_split(X, y)


#Implementacja drzewa decyzyjnego ID3:
class DecisionTreeID3:
    def __init__(self, max_depth=None):
        '''
        Inicjalizuje drzewo decyzyjne ID3.

        Parametry:
            max_depth (int): Maksymalna głębokość drzewa. Jeśli None, głębokość nie jest ograniczana.
        '''
        self.max_depth = max_depth  # Ustawia maksymalną głębokość drzewa (ogranicza złożoność modelu)
        self.tree = None  # Inicjalizuje pustą strukturę drzewa (zostanie zbudowana podczas treningu)

    def entropy(self, y):
        '''
        Oblicza entropię zbioru etykiet.

        Parametry:
            y (iterable): Lista lub seria etykiet klas.

        Zwraca:
            float: Wartość entropii.
        '''
        counts = Counter(y)  # Liczy wystąpienia każdej klasy w zbiorze
        total = len(y)  # Oblicza łączną liczbę elementów
        return -sum(
            (count / total) * math.log2(count / total)  # Wzór na entropię: -p*log2(p)
            for count in counts.values() if count > 0  # Pomija klasy z zerowym udziałem (ochrona przed log2(0))
        )

    def information_gain(self, X_column, y):
        '''
        Oblicza przyrost informacji dla podziału według jednej kolumny.

        Parametry:
            X_column (pd.Series): Kolumna cech do rozważenia jako podział.
            y (pd.Series): Etykiety klas.

        Zwraca:
            float: Wartość przyrostu informacji.
        '''
        values = X_column.unique()  # Pobiera unikalne wartości atrybutu (np. 'x', 'o', 'b')
        parent_entropy = self.entropy(y)  # Oblicza entropię całego zbioru przed podziałem

        # Oblicza sumę ważoną entropii dla każdego podzbioru po podziale wg wartości atrybutu
        weighted_entropy = sum(
            (X_column == v).mean() * self.entropy(y[X_column == v])  # udział * entropia podzbioru
            for v in values
        )

        return parent_entropy - weighted_entropy  # Różnica to zysk informacji (Information Gain)

    def best_split(self, X, y):
        '''
        Znajduje kolumnę cech, która daje największy przyrost informacji.

        Parametry:
            X (pd.DataFrame): Dane wejściowe (cechy).
            y (pd.Series): Etykiety klas.

        Zwraca:
            str: Nazwa najlepszej kolumny do podziału.
        '''
        gains = {
            col: self.information_gain(X[col], y)  # Oblicza przyrost informacji dla każdej cechy
            for col in X.columns
        }

        return max(gains, key=gains.get)  # Zwraca nazwę cechy z największym przyrostem informacji

    def fit(self, X, y, depth=0):
        '''
        Buduje drzewo decyzyjne rekurencyjnie.

        Parametry:
            X (pd.DataFrame): Dane wejściowe.
            y (pd.Series): Etykiety klas.
            depth (int): Aktualna głębokość rekurencji.

        Zwraca:
            dict lub etykieta klasy: Drzewo decyzyjne lub liść z etykietą klasy.
        '''
        # Zakończenie rekurencji, jeśli:
        # - wszystkie etykiety są takie same (brak potrzeby dalszego dzielenia),
        # - brak cech do dalszego podziału,
        # - osiągnięto maksymalną głębokość drzewa.
        if len(set(y)) == 1 or len(X.columns) == 0 or (self.max_depth is not None and depth == self.max_depth):
            return Counter(y).most_common(1)[0][0]  # Zwróć najczęściej występującą etykietę

        best_attr = self.best_split(X, y)  # Znajdź najlepszy atrybut do podziału (największy gain)
        tree = {best_attr: {}}  # Tworzy nowy węzeł drzewa na podstawie wybranego atrybutu

        # Iteruje po wszystkich unikalnych wartościach wybranego atrybutu
        for val in X[best_attr].unique():
            # Tworzy podzbiór danych, w którym atrybut ma wartość val
            subset_X = X[X[best_attr] == val].drop(columns=[best_attr])  # Usuwa użyty atrybut z dalszych rozgałęzień
            subset_y = y[X[best_attr] == val]  # Dopasowane etykiety

            # Rekurencyjnie buduje poddrzewo dla danego podzbioru i zwiększa głębokość
            tree[best_attr][val] = self.fit(subset_X, subset_y, depth + 1)

        return tree  # Zwraca zbudowane drzewo (rekurencyjna struktura słowników)

    def train(self, X, y):
        '''
        Trenuje model drzewa decyzyjnego na danych wejściowych.

        Parametry:
            X (pd.DataFrame): Dane treningowe.
            y (pd.Series): Etykiety klas.
        '''
        self.tree = self.fit(X, y)  # Buduje drzewo decyzyjne na podstawie danych i zapisuje je w atrybucie self.tree

    def predict_one(self, x, tree=None):
        '''
        Przewiduje klasę dla jednej instancji danych.

        Parametry:
            x (pd.Series): Pojedynczy wiersz danych.
            tree (dict): (opcjonalnie) poddrzewo do przeszukiwania.

        Zwraca:
            str lub None: Przewidziana etykieta klasy lub None, jeśli nie można podjąć decyzji.
        '''
        if tree is None:
            tree = self.tree  # Jeśli nie przekazano poddrzewa, zaczynamy od głównego drzewa
        if not isinstance(tree, dict):
            return tree  # Jeśli osiągnięto liść (etykietę klasy), zwracamy go
        attr = next(iter(tree))  # Pobiera nazwę bieżącego atrybutu (klucza w słowniku)
        value = x[attr]  # Pobiera wartość atrybutu z danego wiersza danych
        subtree = tree[attr].get(value)  # Przechodzi do odpowiedniego poddrzewa na podstawie wartości atrybutu
        if subtree is None:
            return None  # Jeśli nie znaleziono ścieżki dla danej wartości, zwraca None (brak decyzji)
        return self.predict_one(x, subtree)  # Rekurencyjnie przechodzi dalej w głąb drzewa

    def predict(self, X):
        '''
        Przewiduje klasy dla wielu instancji danych.

        Parametry:
            X (pd.DataFrame): Dane do klasyfikacji.

        Zwraca:
            pd.Series: Przewidziane etykiety klas.
        '''
        return X.apply(lambda row: self.predict_one(row),
                       axis=1)  # Dla każdego wiersza danych wywołuje predict_one, tworząc serię przewidywanych klas

    def export_graphviz(self, tree=None, graph=None, parent=None, edge_label=None):
        '''
        Tworzy obiekt grafu drzewa decyzyjnego w formacie Graphviz.

        Parametry:
            tree (dict): (opcjonalnie) struktura drzewa do rysowania.
            graph (graphviz.Digraph): (opcjonalnie) istniejący obiekt grafu.
            parent (str): identyfikator węzła nadrzędnego.
            edge_label (str): etykieta na krawędzi między węzłami.

        Zwraca:
            graphviz.Digraph: obiekt grafu gotowy do zapisania lub renderowania.
        '''
        if tree is None:
            tree = self.tree  # Jeśli nie przekazano konkretnego drzewa, używa drzewa modelu

        if graph is None:
            import graphviz
            graph = graphviz.Digraph()  # Tworzy nowy obiekt grafu, jeśli nie został podany

        if not isinstance(tree, dict):
            # Jeśli dotarliśmy do liścia (klasa decyzyjna), dodaj węzeł z etykietą klasy
            graph.node(str(id(tree)), label=str(tree), shape='box')  # Liście jako prostokąty
            if parent:
                graph.edge(parent, str(id(tree)), label=edge_label)  # Łącze od rodzica z etykietą
            return graph  # Zakończ gałąź

        attr = next(iter(tree))  # Pobiera nazwę atrybutu w bieżącym węźle
        node_id = str(id(tree))  # Unikalny identyfikator węzła (na podstawie id obiektu)
        graph.node(node_id, label=attr)  # Dodaje węzeł do grafu z nazwą atrybutu

        if parent:
            graph.edge(parent, node_id, label=edge_label)  # Łączy węzeł z rodzicem

        # Przechodzi przez każdą możliwą wartość atrybutu (gałąź)
        for val, subtree in tree[attr].items():
            # Rekurencyjnie rysuje poddrzewo, ustawiając bieżący węzeł jako rodzica
            self.export_graphviz(subtree, graph, parent=node_id, edge_label=str(val))

        return graph  # Zwraca gotowy obiekt Graphviz.Digraph


#Trening modelu:
model = DecisionTreeID3(max_depth=5)
model.train(X_train, y_train)

#Predykcja:
y_pred = model.predict(X_test)
y_pred = y_pred.fillna("positive")  # fallback w przypadku braku dopasowania

#Metryki oceny:
def manual_accuracy(y_true, y_pred):
    '''
    Oblicza dokładność klasyfikatora jako odsetek poprawnych predykcji.

    Parametry:
        y_true (list lub pd.Series): Prawdziwe etykiety klas.
        y_pred (list lub pd.Series): Przewidziane etykiety klas.

    Zwraca:
        float: Dokładność (wartość od 0 do 1).
    '''
    return np.mean(np.array(y_true) == np.array(y_pred))  # Porównuje etykiety i liczy średnią trafień (True → 1, False → 0)


def manual_confusion_matrix(y_true, y_pred, labels=None):
    '''
    Tworzy macierz pomyłek dla klasyfikatora.

    Parametry:
        y_true (list lub pd.Series): Prawdziwe etykiety klas.
        y_pred (list lub pd.Series): Przewidziane etykiety klas.
        labels (list, opcjonalnie): Lista wszystkich etykiet (kolejność wierszy i kolumn).
                                    Jeśli nie podana, zostanie wyznaczona automatycznie.

    Zwraca:
        pd.DataFrame: Macierz pomyłek (etykiety rzeczywiste vs przewidziane).
    '''
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))  # Ustala unikalne etykiety występujące w predykcjach lub prawdziwych danych

    label_index = {label: idx for idx, label in enumerate(labels)}  # Mapa etykieta → indeks w macierzy
    matrix = np.zeros((len(labels), len(labels)), dtype=int)        # Inicjalizuje pustą macierz (kwadratowa)

    # Zlicza ile razy dana (true, pred) para występuje
    for true, pred in zip(y_true, y_pred):
        i = label_index[true]   # Indeks rzeczywistej etykiety (wiersz)
        j = label_index[pred]   # Indeks przewidzianej etykiety (kolumna)
        matrix[i, j] += 1       # Inkrementuje odpowiednie pole w macierzy

    # Tworzy DataFrame z etykietami jako nagłówki wierszy i kolumn
    return pd.DataFrame(matrix, index=labels, columns=labels)


# Wyniki:
print("Dokładność:", manual_accuracy(y_test, y_pred))  # Wyświetla dokładność klasyfikatora na zbiorze testowym
print("Macierz pomyłek:\n", manual_confusion_matrix(y_test.tolist(), y_pred.tolist()))  # Wyświetla macierz pomyłek jako tabelę

#grafy i eksperymenty:
depths = range(1, 10)
accuracies = []

for depth in depths:
    model = DecisionTreeID3(max_depth=depth)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test).fillna("positive")

    acc = manual_accuracy(y_test, y_pred)
    accuracies.append(acc)

    cm = manual_confusion_matrix(y_test.tolist(), y_pred.tolist())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek (głębokość={depth})")
    plt.ylabel("Rzeczywista")
    plt.xlabel("Przewidziana")
    plt.tight_layout()
    plt.savefig(f"confusion_depth_{depth}.png")
    plt.close()

    # Save tree graph
    graph = model.export_graphviz()
    graph.render(filename=f"tree_depth_{depth}", format="png", cleanup=True)

# Plot accuracy vs depth
plt.figure(figsize=(8, 5))
plt.plot(depths, accuracies, marker='o')
plt.title("Dokładność vs Głębokość drzewa")
plt.xlabel("Maksymalna głębokość drzewa")
plt.ylabel("Dokładność")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"accuracy_vs_depth.png")
plt.close()
