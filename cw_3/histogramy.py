import os
import re
import matplotlib.pyplot as plt

# Pliki z wynikami (upewnij się, że są w tym samym folderze co ten skrypt)
filenames = [
    "wyniki_minmax_minmax.txt",
    "wyniki_minmax_random.txt",
    "wyniki_random_minmax.txt",
    "wyniki_random_random.txt"
]

def parse_results(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    games = content.split("-------------------------")
    o_times = []
    x_times = []
    results = []

    for game in games:
        result_match = re.search(r"RESULT:(\w+)", game)
        time_match = re.search(r"Time played: \{'o': ([\d\.]+), 'x': ([\d\.]+)\}", game)

        if result_match and time_match:
            results.append(result_match.group(1))
            o_times.append(float(time_match.group(1)))
            x_times.append(float(time_match.group(2)))

    return o_times, x_times, results

for filename in filenames:
    if not os.path.exists(filename):
        print(f"Nie znaleziono pliku: {filename}")
        continue

    o_times, x_times, results = parse_results(filename)
    game_type = filename.replace("wyniki_", "").replace(".txt", "").replace("_", " vs ")

    # Histogram czasu gracza "o"
    plt.figure()
    plt.hist(o_times, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(sum(o_times) / len(o_times), color='red', linestyle='dashed', linewidth=1, label='Średnia')
    plt.title(f'Czas gry gracza "o" ({game_type})')
    plt.xlabel("Czas (ms)")
    plt.ylabel("Liczba gier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Histogram czasu gracza "x"
    plt.figure()
    plt.hist(x_times, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(sum(x_times) / len(x_times), color='red', linestyle='dashed', linewidth=1, label='Średnia')
    plt.title(f'Czas gry gracza "x" ({game_type})')
    plt.xlabel("Czas (ms)")
    plt.ylabel("Liczba gier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Histogram wyników
    plt.figure()
    plt.hist(results, bins=len(set(results)), edgecolor='black', alpha=0.7)
    plt.title(f'Wyniki gier ({game_type})')
    plt.xlabel("Zwycięzca")
    plt.ylabel("Liczba gier")
    plt.grid(True)
    plt.tight_layout()

plt.show()
