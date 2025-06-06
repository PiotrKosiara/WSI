import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Słownik zawierający współrzędne miast (punkty na płaszczyźnie 2D)
cities = {
    'A': (0, 0), 'B': (1, 3), 'C': (2, 1), 'D': (4, 6), 'E': (5, 2),
    'F': (6, 5), 'G': (8, 7), 'H': (9, 4), 'I': (10, 8), 'J': (12, 3)
}

# Lista nazw miast oraz macierz współrzędnych
city_names = list(cities.keys())
#print(city_names)
coords = np.array([cities[city] for city in city_names])
#print(coords)
num_cities = len(coords)

# Funkcja obliczająca macierz odległości euklidesowych między wszystkimi miastami
# To działa w taki sposób że: Dodaje nową oś pośrodku. Jeśli coords ma kształt (n, d), to po tej operacji ma (n, 1, d)
# – czyli każdy punkt staje się osobnym blokiem do porównania i to samo, ale nowa oś jest na początku – wynik ma kształt (1, n, d)
def compute_distance_matrix(coords):
    '''
    Oblicza macierz odległości euklidesowych między wszystkimi punktami.

    Parametry:
    coords (ndarray): Tablica NumPy o wymiarach (n, d), gdzie n to liczba punktów,
                      a d to liczba wymiarów (np. 2 dla współrzędnych 2D).

    Zwraca:
    ndarray: Macierz (n, n), w której element [i][j] to odległość euklidesowa między punktem i i punktem j.
    '''
    # Różnica między każdą parą punktów, następnie norma euklidesowa
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))
    return dist_matrix

# Macierz odległości dla wszystkich miast
dist_matrix = compute_distance_matrix(coords)
distance_matrix = dist_matrix

# Rysowanie wykresu miast
# plt.figure(figsize=(10, 8))
# plt.scatter(coords[:, 0], coords[:, 1], c='blue')
#
# # Dodaj nazwy miast
# for i, name in enumerate(city_names):
#     plt.text(coords[i, 0] + 0.2, coords[i, 1] + 0.2, name, fontsize=12)
#
# # Rysuj linie i odległości między każdą parą miast
# for i in range(num_cities):
#     for j in range(i + 1, num_cities):
#         x_values = [coords[i, 0], coords[j, 0]]
#         y_values = [coords[i, 1], coords[j, 1]]
#         plt.plot(x_values, y_values, 'gray', alpha=0.4)
#         # Oblicz środek odcinka
#         mid_x = (coords[i, 0] + coords[j, 0]) / 2
#         mid_y = (coords[i, 1] + coords[j, 1]) / 2
#         # Dodaj etykietę z odległością (zaokrągloną)
#         distance = dist_matrix[i, j]
#         plt.text(mid_x, mid_y, f"{distance:.1f}", fontsize=8, color='darkred')
#
# plt.title("Miasta i odległości między każdą parą")
# plt.axis('equal')
# plt.grid(True)
# plt.show()

# Funkcja obliczająca długość trasy (z powrotem do miasta startowego)
def path_length(path, distance_matrix):
    '''
        Oblicza łączną długość ścieżki na podstawie podanej kolejności miast
        oraz macierzy odległości.

        Parametry:
        path (list or ndarray): Lista indeksów miast w kolejności odwiedzania (np. [0, 2, 1, 3]).
        distance_matrix (ndarray): Kwadratowa macierz odległości, gdzie element [i][j] to odległość z miasta i do j.

        Zwraca:
        float: Całkowita długość ścieżki, w tym powrót do punktu początkowego.
        '''
    total = sum(distance_matrix[path[i], path[(i + 1) % num_cities]] for i in range(num_cities))
    return total

# Funkcja przystosowania: odwrotność długości trasy (im krótsza, tym lepsza)
def fitness(path):
    '''
    Oblicza wartość funkcji dopasowania (fitness) dla danej ścieżki.
    Im krótsza trasa, tym wyższa wartość funkcji dopasowania.

    Parametry:
    path (list or ndarray): Lista indeksów miast w kolejności odwiedzania.

    Zwraca:
    float: Wartość funkcji dopasowania, zdefiniowana jako odwrotność długości trasy (1 / długość).
    '''
    #print(1.0 / path_length(path, distance_matrix))
    return 1.0 / path_length(path, distance_matrix)

# Tworzy początkową populację losowych tras (permutacji indeksów miast)
def init_population(pop_size):
    '''
        Inicjalizuje populację permutacji miast (ścieżek) dla algorytmu genetycznego.

        Parametry:
        pop_size (int): Liczba osobników w populacji (czyli liczba różnych permutacji miast).

        Zwraca:
        ndarray: Tablica NumPy o wymiarach (pop_size, num_cities), gdzie każdy wiersz to losowa permutacja miast.
        '''
    return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

# Selekcja ruletkowa: wybiera osobniki proporcjonalnie do przystosowania
def roulette_selection(population, fitnesses):
    '''
    Przeprowadza selekcję ruletkową (proporcjonalną) na podstawie wartości dopasowania (fitness).

    Każdy osobnik ma szansę bycia wybranym proporcjonalną do swojej wartości fitness.
    W wyniku powstaje nowa populacja tej samej wielkości.

    Parametry:
    population (ndarray): Tablica populacji, gdzie każdy wiersz to jeden osobnik (ścieżka).
    fitnesses (ndarray): Tablica wartości fitness odpowiadających każdemu osobnikowi.

    Zwraca:
    ndarray: Nowa populacja (tej samej wielkości), wybrana na podstawie selekcji ruletkowej.
    '''
    probs = fitnesses / fitnesses.sum()  # Prawdopodobieństwa wyboru
    indices = np.random.choice(len(population), size=len(population), p=probs)
    return population[indices]

# Selekcja turniejowa: wybiera najlepszych z losowych grup
def tournament_selection(population, fitnesses, tournament_size=4):
    '''
    Przeprowadza selekcję turniejową na podstawie wartości fitness.

    W każdej iteracji losowana jest grupa `tournament_size` osobników (bez powtórzeń),
    z której wybierany jest ten z najwyższym fitness. Proces powtarzany jest tyle razy,
    ile wynosi liczebność populacji, aby utworzyć nową populację tej samej wielkości.

    Parametry:
    population (ndarray): Tablica populacji, gdzie każdy wiersz to jeden osobnik (ścieżka).
    fitnesses (ndarray): Tablica wartości fitness odpowiadających każdemu osobnikowi.
    tournament_size (int): Liczba osobników biorących udział w jednym turnieju (domyślnie 4).

    Zwraca:
    ndarray: Nowa populacja wybrana metodą selekcji turniejowej.
    '''
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants_idx = np.random.choice(pop_size, tournament_size, replace=False)
        best_idx = participants_idx[np.argmax(fitnesses[participants_idx])]
        selected.append(population[best_idx])
    return np.array(selected)

# Jednopunktowe krzyżowanie z naprawą duplikatów przez wymianę między dziećmi
def custom_crossover_exchange_duplicates(parent1, parent2):
    '''
    Wykonuje krzyżowanie jednopunktowe dwóch permutacji (rodziców) i usuwa ewentualne duplikaty,
    aby uzyskać poprawne dzieci (permutacje bez powtórzeń).

    Proces:
    - Losowany jest punkt cięcia (pozycja w chromosomie).
    - Tworzone są dwa dzieci poprzez połączenie fragmentów rodziców.
    - Następnie naprawiane są ewentualne duplikaty poprzez wymianę genów między dziećmi.

    Parametry:
    parent1 (ndarray): Pierwszy rodzic (permutacja miast).
    parent2 (ndarray): Drugi rodzic (permutacja miast).

    Zwraca:
    tuple: Dwie poprawne permutacje (ndarray), będące potomkami parent1 i parent2.
    '''
    size = len(parent1)
    cut = np.random.randint(1, size - 1)  # Losowy punkt cięcia (nie na skraju)

    # Tworzenie dzieci przez połączenie fragmentów obu rodziców
    child1 = np.concatenate((parent1[:cut], parent2[cut:]))
    child2 = np.concatenate((parent2[:cut], parent1[cut:]))

    # Naprawa duplikatów przez wymianę pomiędzy dziećmi
    def fix_duplicates(c1, c2):
        '''
                Naprawia duplikaty w dwóch chromosomach (permutacjach) poprzez wymianę genów między nimi.

                Proces:
                - Wyszukiwanie duplikatów (czyli genów powtarzających się) w obu dzieciach.
                - Wymiana duplikatów pomiędzy dziećmi tak, aby uzyskać poprawne permutacje.

                Parametry:
                c1 (ndarray): Pierwsze dziecko z potencjalnymi duplikatami.
                c2 (ndarray): Drugie dziecko z potencjalnymi duplikatami.

                Zwraca:
                tuple: Dwie permutacje po naprawie, bez powtórzonych elementów.
        '''
        seen1 = set()
        seen2 = set()
        duplicates1 = []
        duplicates2 = []

        # Zbieranie indeksów duplikatów
        for i in range(size):
            if c1[i] in seen1:
                duplicates1.append(i)
            else:
                seen1.add(c1[i])
            if c2[i] in seen2:
                duplicates2.append(i)
            else:
                seen2.add(c2[i])

        # Wymiana duplikatów pomiędzy dziećmi
        for i, j in zip(duplicates1, duplicates2):
            c1[i], c2[j] = c2[j], c1[i]

        return c1, c2

    # Zwraca naprawione dzieci (z legalnymi permutacjami)
    child1, child2 = fix_duplicates(child1.copy(), child2.copy())
    return child1, child2

# Mutacja – z ustalonym prawdopodobieństwem zamienia miejscami dwa miasta w trasie
def mutate(individual, mutation_rate):
    '''
    Przeprowadza mutację permutacji (osobnika) z określonym prawdopodobieństwem.

    Mutacja polega na zamianie miejscami dwóch losowo wybranych genów (miast) w permutacji.
    Jest stosowana z prawdopodobieństwem określonym przez mutation_rate.

    Parametry:
    individual (ndarray): Permutacja reprezentująca jednego osobnika (trasę).
    mutation_rate (float): Prawdopodobieństwo wystąpienia mutacji (wartość z zakresu [0, 1]).

    Zwraca:
    ndarray: Zmutowany (lub niezmieniony) osobnik.
    '''
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(num_cities, size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Główna funkcja realizująca algorytm genetyczny
def genetic_algorithm(pop_size=100, generations=500, mutation_rate=0.05, selection_method='roulette'):
    '''
    Wykonuje algorytm genetyczny w celu rozwiązania problemu komiwojażera (TSP – Travelling Salesman Problem).

    Algorytm ewoluuje populację permutacji miast w wielu pokoleniach, starając się znaleźć najkrótszą możliwą trasę
    odwiedzającą każde miasto dokładnie raz i wracającą do punktu startowego.

    Etapy działania:
    1. Inicjalizacja populacji — tworzona jest początkowa populacja losowych permutacji miast.
    2. Ewaluacja — każdemu osobnikowi przypisywana jest wartość fitness (odwrotność długości trasy).
    3. Selekcja — wybór osobników do rozmnażania (ruletka lub turniej).
    4. Krzyżowanie — tworzenie potomstwa poprzez krzyżowanie jednopunktowe z naprawą duplikatów.
    5. Mutacja — z losowym prawdopodobieństwem zamieniane są miejscami dwa miasta w trasie.
    6. Sukcesja — nowa populacja zastępuje starą.
    7. Powtórzenie — powyższe kroki są wykonywane przez zadaną liczbę pokoleń.

    Parametry:
    pop_size (int): Liczba osobników (tras) w populacji. Domyślnie 100.
    generations (int): Liczba pokoleń, przez które będzie działać algorytm. Domyślnie 500.
    mutation_rate (float): Prawdopodobieństwo mutacji jednego osobnika (0.0 – 1.0). Domyślnie 0.05.
    selection_method (str): Metoda selekcji do rozmnażania. Dozwolone: 'roulette' (selekcja ruletkowa),
                            'tournament' (selekcja turniejowa). Domyślnie 'roulette'.

    Wyjątki:
    ValueError: Gdy `selection_method` nie jest jedną z obsługiwanych wartości ('roulette' lub 'tournament').

    Zwraca:
    tuple:
        - best (ndarray): Najlepsza znaleziona permutacja miast (czyli trasa o najwyższym dopasowaniu / najkrótsza).
        - best_distance (float): Długość tej trasy (czyli suma odległości między kolejnymi miastami w trasie, z powrotem na początek).

    Uwagi:
    - Funkcja zakłada, że globalna zmienna `distance_matrix` jest wcześniej zdefiniowana i zawiera macierz odległości
      między miastami, zgodną z permutacjami osobników.
    - Funkcja korzysta z wielu pomocniczych funkcji, m.in.:
        - `init_population`
        - `fitness`
        - `roulette_selection` i `tournament_selection`
        - `custom_crossover_exchange_duplicates`
        - `mutate`
        - `path_length`
    - Liczba miast (num_cities) musi być zgodna z długością permutacji zwracanych przez `init_population`.
    '''

    population = init_population(pop_size)
    best = None
    #best_fit na początku minus nieskończoność
    best_fit = -np.inf

    for gen in range(generations):
        #pętla pokoleniowa
        fitnesses = np.array([fitness(ind) for ind in population])
        #Dla każdego osobnika w populacji liczymy jego wartość dopasowania (fitness)
        new_population = []
        #pusta lista nowej populacji z poprzedniej populacji

        # Zapamiętaj najlepszego osobnika
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best = population[gen_best_idx]
            #Jeśli jego fitness jest lepszy niż dotychczasowy best_fit, to: aktualizujemy best_fit i zapamiętujemy jego trasę jako best

        # Wybór selekcji
        if selection_method == 'roulette':
            selected = roulette_selection(population, fitnesses)
        elif selection_method == 'tournament':
            selected = tournament_selection(population, fitnesses)
        else:
            raise ValueError("Nieznana metoda selekcji: użyj 'roulette' lub 'tournament'")

        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[(i + 1) % pop_size]
            c1, c2 = custom_crossover_exchange_duplicates(p1, p2)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_population.extend([c1, c2])
            #Dla każdej pary rodziców (p1 i p2):
                #Tworzymy potomków (c1, c2) przez krzyżowanie z naprawą duplikatów (bo to permutacje).
                # Mutujemy każdego z potomków z prawdopodobieństwem mutation_rate.
                # Dodajemy ich do nowej populacji.
            #Uwaga: (i + 1) % pop_size zapewnia, że jeśli i jest ostatnim indeksem, to parujemy go z pierwszym (żeby zawsze mieć parę).

        population = np.array(new_population)

    return best, path_length(best, distance_matrix)


# Uruchomienie programu i wypisanie najlepszego wyniku
if __name__ == "__main__":
    #bazowe uruchomienie
    best_solution, best_distance = genetic_algorithm(pop_size=160, generations=300, mutation_rate=0.08, selection_method ='roulette')
    print("Najlepsza trasa:", [city_names[i] for i in best_solution])
    print("Długość trasy:", round(best_distance, 2))

    #Wykresy do porównania selekcji ruletkowej z turniejową
    # pop_size = 160
    # generations = 300
    # mutation_rate = 0.08
    # num_runs = 50
    # methods = ['roulette', 'tournament']
    # results = {'metoda': [], 'długość_trasy': [], 'czas_s': []}
    #
    # # A + C: 100 uruchomień każdej metody
    # for method in methods:
    #     for _ in range(num_runs):
    #         start = time.time()
    #         _, best_distance = genetic_algorithm(pop_size=pop_size, generations=generations,
    #                                              mutation_rate=mutation_rate, selection_method=method)
    #         end = time.time()
    #         results['metoda'].append(method)
    #         results['długość_trasy'].append(best_distance)
    #         results['czas_s'].append(end - start)
    #
    # df = pd.DataFrame(results)
    #
    # # A. Wykres słupkowy: średnia długość trasy z błędem standardowym
    # means = df.groupby("metoda")["długość_trasy"].mean()
    # stds = df.groupby("metoda")["długość_trasy"].std()
    #
    # plt.figure(figsize=(8, 6))
    # plt.bar(means.index, means.values, yerr=stds.values, capsize=10, color=['skyblue', 'lightgreen'])
    # plt.title("Średnia długość trasy (50 uruchomień)")
    # plt.ylabel("Długość trasy")
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()
    #
    # # B. Wykres liniowy: zbieżność algorytmu dla jednej próbki
    # def get_progress(selection_method):
    #     population = init_population(pop_size)
    #     progress = []
    #     for _ in range(generations):
    #         fitnesses = np.array([fitness(ind) for ind in population])
    #         best_idx = np.argmax(fitnesses)
    #         progress.append(path_length(population[best_idx], distance_matrix))
    #         if selection_method == 'roulette':
    #             selected = roulette_selection(population, fitnesses)
    #         elif selection_method == 'tournament':
    #             selected = tournament_selection(population, fitnesses)
    #         new_population = []
    #         for i in range(0, pop_size, 2):
    #             p1, p2 = selected[i], selected[(i + 1) % pop_size]
    #             c1, c2 = custom_crossover_exchange_duplicates(p1, p2)
    #             new_population.extend([mutate(c1, mutation_rate), mutate(c2, mutation_rate)])
    #         population = np.array(new_population)
    #     return progress
    #
    # roulette_progress = get_progress('roulette')
    # tournament_progress = get_progress('tournament')
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(roulette_progress, label="Ruletkowa", linestyle='--')
    # plt.plot(tournament_progress, label="Turniejowa", linestyle='-')
    # plt.xlabel("Generacja")
    # plt.ylabel("Długość najlepszej trasy")
    # plt.title("Zbieżność algorytmu genetycznego")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # C. Tabela porównawcza
    # summary_df = df.groupby("metoda").agg({
    #     "długość_trasy": ['mean', 'std'],
    #     "czas_s": 'mean'
    # }).reset_index()
    # summary_df.columns = ["Metoda", "Średnia długość trasy", "Odchylenie std", "Średni czas [s]"]
    #
    # print("\n=== Tabela porównawcza ===")
    # print(summary_df)


    #Wykresy do heatmap
    # --- Eksperyment: Heatmap dla populacji vs generacje ---
    # pop_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # generations_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # num_runs = 5
    #
    # distance_matrix_result = np.zeros((len(pop_sizes), len(generations_list)))
    # time_matrix_result = np.zeros((len(pop_sizes), len(generations_list)))
    #
    # for i, pop_size in enumerate(pop_sizes):
    #     for j, gens in enumerate(generations_list):
    #         distances = []
    #         times = []
    #         for _ in range(num_runs):
    #             start = time.time()
    #             dist = genetic_algorithm(pop_size=pop_size, generations=gens)
    #             end = time.time()
    #             distances.append(dist)
    #             times.append(end - start)
    #         distance_matrix_result[i, j] = np.mean(distances)
    #         time_matrix_result[i, j] = np.mean(times)
    #
    # # --- Wykres 1: heatmap długości trasy ---
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(distance_matrix_result, xticklabels=generations_list, yticklabels=pop_sizes, annot=True, fmt=".2f",
    #             cmap="YlGnBu")
    # plt.xlabel("Liczba generacji")
    # plt.ylabel("Rozmiar populacji")
    # plt.title("Średnia długość najlepszej trasy (TSP)")
    # plt.tight_layout()
    # plt.savefig("heatmap_trasa.png")
    # plt.show()
    #
    # # --- Wykres 2: heatmap czasu działania ---
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(time_matrix_result, xticklabels=generations_list, yticklabels=pop_sizes, annot=True, fmt=".4f",
    #             cmap="OrRd")
    # plt.xlabel("Liczba generacji")
    # plt.ylabel("Rozmiar populacji")
    # plt.title("Średni czas działania algorytmu [s]")
    # plt.tight_layout()
    # plt.savefig("heatmap_czas.png")
    # plt.show()

    #Testowanie parametrów populacji
    # --- Parametry eksperymentu ---
    # population_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # generations = 500
    # mutation_rate = 0.05
    # num_runs = 5  # liczba odpaleń dla każdej populacji
    #
    # avg_distances = []
    # avg_times = []
    #
    # for pop_size in population_sizes:
    #     print(f"Test dla populacji: {pop_size}")
    #     run_distances = []
    #     run_times = []
    #     for run in range(num_runs):
    #         start_time = time.time()
    #         best_distance = genetic_algorithm(pop_size=pop_size, generations=generations, mutation_rate=mutation_rate)
    #         end_time = time.time()
    #         elapsed = end_time - start_time
    #
    #         run_distances.append(best_distance)
    #         run_times.append(elapsed)
    #         print(f"  Run {run + 1}: {best_distance:.2f} | Czas: {elapsed:.2f} s")
    #     avg_distance = np.mean(run_distances)
    #     avg_time = np.mean(run_times)
    #
    #     avg_distances.append(avg_distance)
    #     avg_times.append(avg_time)
    #
    #     print(f"Średnia dla populacji {pop_size}: {avg_distance:.2f}, czas: {avg_time:.2f} s\n")
    #
    # # --- Wykres 1: Średnia długość trasy ---
    # plt.figure(figsize=(10, 6))
    # plt.bar([str(p) for p in population_sizes], avg_distances, color='skyblue')
    # plt.xlabel("Rozmiar populacji")
    # plt.ylabel("Średnia najlepsza długość trasy (z 5 uruchomień)")
    # plt.title("Wpływ rozmiaru populacji na jakość rozwiązania (TSP)")
    # plt.grid(axis='y')
    # plt.tight_layout()
    # plt.show()
    #
    # # --- Wykres 2: Średni czas wykonania ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(population_sizes, avg_times, marker='o', linestyle='-', color='orange')
    # plt.xlabel("Rozmiar populacji")
    # plt.ylabel("Średni czas działania algorytmu [s]")
    # plt.title("Wpływ rozmiaru populacji na czas wykonania algorytmu")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


    #Testowanie mutacji
    # --- Parametry eksperymentu ---
    # mutation_rates = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
    # generations = 300
    # pop_size = 160
    # num_runs = 15
    #
    # avg_distances = []
    # avg_times = []
    #
    # for mutation_rate in mutation_rates:
    #     print(f"\nTest dla mutacji: {mutation_rate}")
    #     run_distances = []
    #     run_times = []
    #     for run in range(num_runs):
    #         start_time = time.time()
    #         best_distance = genetic_algorithm(pop_size=pop_size, generations=generations, mutation_rate=mutation_rate)
    #         end_time = time.time()
    #         elapsed = end_time - start_time
    #
    #         run_distances.append(best_distance)
    #         run_times.append(elapsed)
    #         print(f"  Run {run + 1}: {best_distance:.2f} | Czas: {elapsed:.2f} s")
    #
    #     avg_distance = np.mean(run_distances)
    #     avg_time = np.mean(run_times)
    #
    #     avg_distances.append(avg_distance)
    #     avg_times.append(avg_time)
    #
    #     print(f"Średnia długość trasy: {avg_distance:.2f}, Średni czas: {avg_time:.2f} s")
    #
    # # --- Wykres 1: Średnia długość trasy vs mutacja ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(mutation_rates, avg_distances, marker='o', linestyle='-', color='steelblue')
    # plt.xlabel("Prawdopodobieństwo mutacji")
    # plt.ylabel("Średnia długość najlepszej trasy (5 uruchomień)")
    # plt.title("Wpływ prawdopodobieństwa mutacji na jakość rozwiązania")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # --- Wykres 2: Średni czas działania vs mutacja ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(mutation_rates, avg_times, marker='o', linestyle='-', color='darkorange')
    # plt.xlabel("Prawdopodobieństwo mutacji")
    # plt.ylabel("Średni czas działania [s]")
    # plt.title("Wpływ prawdopodobieństwa mutacji na czas wykonania algorytmu")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    #Testowanie ilości generacji
    # --- Eksperyment ---
    # generation_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # pop_size_fixed = 200
    # mutation_rate_fixed = 0.05
    # num_runs = 5
    #
    # avg_distances = []
    # avg_times = []
    #
    # print("Generacje;Średnia długość trasy;Średni czas [s]")
    # for gens in generation_counts:
    #     run_distances = []
    #     run_times = []
    #     for _ in range(num_runs):
    #         start = time.time()
    #         result = genetic_algorithm(pop_size=pop_size_fixed, generations=gens, mutation_rate=mutation_rate_fixed)
    #         end = time.time()
    #         run_distances.append(result)
    #         run_times.append(end - start)
    #     avg_d = np.mean(run_distances)
    #     avg_t = np.mean(run_times)
    #     avg_distances.append(avg_d)
    #     avg_times.append(avg_t)
    #     print(f"{gens};{avg_d:.2f};{avg_t:.4f}")
    #
    # # --- Wykres 1: jakość rozwiązania ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(generation_counts, avg_distances, marker='o', linestyle='-', color='green')
    # plt.xlabel("Liczba generacji")
    # plt.ylabel("Średnia najlepsza długość trasy")
    # plt.title("Wpływ liczby generacji na jakość rozwiązania (TSP)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("jakosc_vs_generacje.png")
    # plt.show()
    #
    # # --- Wykres 2: czas działania ---
    # plt.figure(figsize=(10, 6))
    # plt.plot(generation_counts, avg_times, marker='o', linestyle='-', color='red')
    # plt.xlabel("Liczba generacji")
    # plt.ylabel("Średni czas działania [s]")
    # plt.title("Wpływ liczby generacji na czas działania algorytmu")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("czas_vs_generacje.png")
    # plt.show()