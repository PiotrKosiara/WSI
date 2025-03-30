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
def compute_distance_matrix(coords):
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
    total = sum(distance_matrix[path[i], path[(i + 1) % num_cities]] for i in range(num_cities))
    return total

# Funkcja przystosowania: odwrotność długości trasy (im krótsza, tym lepsza)
def fitness(path):
    #print(1.0 / path_length(path, distance_matrix))
    return 1.0 / path_length(path, distance_matrix)

# Tworzy początkową populację losowych tras (permutacji indeksów miast)
def init_population(pop_size):
    return np.array([np.random.permutation(num_cities) for _ in range(pop_size)])

# Selekcja ruletkowa: wybiera osobniki proporcjonalnie do przystosowania
def roulette_selection(population, fitnesses):
    probs = fitnesses / fitnesses.sum()  # Prawdopodobieństwa wyboru
    indices = np.random.choice(len(population), size=len(population), p=probs)
    return population[indices]

# Selekcja turniejowa: wybiera najlepszych z losowych grup
def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants_idx = np.random.choice(pop_size, tournament_size, replace=False)
        best_idx = participants_idx[np.argmax(fitnesses[participants_idx])]
        selected.append(population[best_idx])
    return np.array(selected)

# Jednopunktowe krzyżowanie z naprawą duplikatów przez wymianę między dziećmi
def custom_crossover_exchange_duplicates(parent1, parent2):
    size = len(parent1)
    cut = np.random.randint(1, size - 1)  # Losowy punkt cięcia (nie na skraju)

    # Tworzenie dzieci przez połączenie fragmentów obu rodziców
    child1 = np.concatenate((parent1[:cut], parent2[cut:]))
    child2 = np.concatenate((parent2[:cut], parent1[cut:]))

    # Naprawa duplikatów przez wymianę pomiędzy dziećmi
    def fix_duplicates(c1, c2):
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
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(num_cities, size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Główna funkcja realizująca algorytm genetyczny
def genetic_algorithm(pop_size=100, generations=500, mutation_rate=0.05, selection_method='roulette'):
    population = init_population(pop_size)
    best = None
    best_fit = -np.inf

    for gen in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        new_population = []

        # Zapamiętaj najlepszego osobnika
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best = population[gen_best_idx]

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