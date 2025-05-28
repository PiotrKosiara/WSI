import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time


def q_learning(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=500, max_steps=100):
    '''
    Wykonuje algorytm Q-learning w środowisku o dyskretnej przestrzeni stanów i akcji.

    Parametry:
    - env: środowisko zgodne z API Gymnasium (musi mieć przestrzeń dyskretną stanów i akcji),
    - alpha (float): współczynnik uczenia, określający tempo aktualizacji wartości Q,
    - gamma (float): współczynnik dyskontujący, decydujący o znaczeniu przyszłych nagród,
    - epsilon (float): współczynnik eksploracji w strategii epsilon-greedy,
    - episodes (int): liczba epizodów treningowych,
    - max_steps (int): maksymalna liczba kroków w pojedynczym epizodzie.

    Zwraca:
    - Q (ndarray): wyuczona tablica Q o wymiarach (liczba stanów × liczba akcji),
    - rewards (list): lista sum nagród z każdego epizodu.
    '''
    n_states = env.observation_space.n  # Liczba możliwych stanów w środowisku
    n_actions = env.action_space.n      # Liczba dostępnych akcji w środowisku

    Q = np.zeros((n_states, n_actions))  # Inicjalizacja tablicy Q zerami
    rewards = []  # Lista do przechowywania sum nagród z każdego epizodu

    for ep in range(episodes):  # Pętla po epizodach treningowych
        state, _ = env.reset()  # Reset środowiska i pobranie początkowego stanu
        total_reward = 0  # Inicjalizacja sumy nagród dla danego epizodu

        for _ in range(max_steps):  # Pętla po maksymalnej liczbie kroków
            if np.random.uniform(0, 1) < epsilon:  # Strategia eksploracyjna (epsilon-greedy)
                action = env.action_space.sample()  # Wybierz losową akcję z prawdopodobieństwem epsilon
            else:
                action = np.argmax(Q[state])  # W przeciwnym razie wybierz najlepszą znaną akcję

            next_state, reward, terminated, truncated, _ = env.step(action)  # Wykonaj akcję i pobierz wynik
            done = terminated or truncated  # Sprawdź, czy epizod się zakończył

            best_next_action = np.argmax(Q[next_state])  # Wybierz najlepszą przyszłą akcję
            td_target = reward + gamma * Q[next_state][best_next_action]  # Oblicz wartość celu (target)
            Q[state][action] += alpha * (td_target - Q[state][action])  # Aktualizacja wartości Q

            state = next_state  # Przejdź do nowego stanu
            total_reward += reward  # Dodaj nagrodę do sumy za dany epizod

            if done:  # Przerwij epizod, jeśli zakończony
                break

        rewards.append(total_reward)  # Zapisz sumę nagród z epizodu

    return Q, rewards  # Zwróć wytrenowaną tablicę Q oraz listę nagród


def avg_q_learning_rewards(env, alpha, episodes, runs=100, max_steps=100):
    '''
    Wykonuje wielokrotne treningi Q-learningu i zwraca średnią sumę nagród oraz średni czas trwania jednego treningu.

    Parametry:
    - env: środowisko zgodne z API Gymnasium (np. 'CliffWalking-v0'),
    - alpha (float): współczynnik uczenia (learning rate),
    - episodes (int): liczba epizodów w jednym treningu,
    - runs (int): liczba powtórzeń treningu do uśrednienia (domyślnie 100),
    - max_steps (int): maksymalna liczba kroków w epizodzie (domyślnie 100).

    Zwraca:
    - avg_rewards (ndarray): średnia suma nagród z każdego epizodu, uśredniona po wszystkich powtórzeniach,
    - avg_duration (float): średni czas trwania jednego pełnego treningu (w sekundach).
    '''
    all_rewards = []  # Lista do przechowywania wszystkich przebiegów (list nagród z epizodów)

    start_time = time.time()  # Rozpocznij pomiar czasu całego eksperymentu

    for _ in range(runs):  # Wykonaj 'runs' powtórzeń treningu
        _, rewards = q_learning(env, alpha=alpha, episodes=episodes, max_steps=max_steps)  # Uruchom pojedynczy trening Q-learning
        all_rewards.append(rewards)  # Dodaj uzyskane nagrody z epizodów do listy

    end_time = time.time()  # Zakończ pomiar czasu po wszystkich treningach

    avg_rewards = np.mean(all_rewards, axis=0)  # Oblicz średnią z nagród w każdym epizodzie (po wszystkich powtórzeniach)
    avg_duration = (end_time - start_time) / runs  # Oblicz średni czas trwania pojedynczego treningu

    return avg_rewards, avg_duration  # Zwróć średnie nagrody oraz średni czas jednego treningu


def visualize_q_policy(Q, title="Polityka wyuczona przez Q-learning"):
    """
    Rysuje graficzne przedstawienie wyuczonej polityki na planszy CliffWalking 4x12.
    Zakłada, że środowisko ma 48 stanów (4 wiersze x 12 kolumn) i 4 możliwe akcje.
    """
    action_symbols = ['→', '↓', '←', '↑']  # Przypisanie symboli do akcji 0–3
    arrows = {
        0: (0, 1),   # right
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (-1, 0)   # up
    }

    policy = np.argmax(Q, axis=1).reshape(4, 12)  # Wybór najlepszej akcji w każdym stanie
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(title)
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(4))
    ax.invert_yaxis()

    for y in range(4):
        for x in range(12):
            state = y * 12 + x
            action = np.argmax(Q[state])
            dx, dy = arrows[action]
            ax.arrow(x, y, dx * 0.3, dy * 0.3, head_width=0.2, head_length=0.2, fc='black', ec='black')

    # Dodanie oznaczeń klifu
    for x in range(1, 11):
        ax.add_patch(plt.Rectangle((x-0.5, 3.5), 1, 1, color='black', alpha=0.3))
    ax.text(0, 3.7, 'S', fontsize=12, ha='center')  # Start
    ax.text(11, 3.7, 'G', fontsize=12, ha='center')  # Goal

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    # Inicjalizacja środowiska CliffWalking
    env = gym.make("CliffWalking-v0")

    # Trening jednorazowy, by zobaczyć konkretną politykę
    Q, _ = q_learning(env, alpha=0.1, episodes=500, max_steps=100)
    visualize_q_policy(Q, title="Wytrenowana polityka Q-learning (α=0.1, ep=500)")

    # Test 1: badanie wpływu liczby epizodów (przy stałym alpha = 0.1)
    configs_episodes = [
        {"alpha": 0.1, "episodes": 500}  # Przykładowa konfiguracja: 500 epizodów
    ]

    episode_counts = []  # Lista do zapisu liczby epizodów
    durations_ep = []    # Lista do zapisu średnich czasów trwania treningu

    print("### Test 1: Średnia z 100 treningów – różne liczby epizodów, α=0.1 ###\n")
    plt.figure(figsize=(14, 6))  # Przygotowanie wykresu nagród

    for cfg in configs_episodes:
        print(f"Trening: alpha={cfg['alpha']}, episodes={cfg['episodes']}")

        # Wykonanie 100 treningów Q-learning i uśrednienie wyników
        avg_rewards, avg_time = avg_q_learning_rewards(env, alpha=cfg["alpha"], episodes=cfg["episodes"], runs=100)

        # Wykres średnich nagród w funkcji epizodów
        plt.plot(avg_rewards, label=f"ep={cfg['episodes']}")

        # Zapisywanie liczby epizodów i średniego czasu do osobnych list
        episode_counts.append(cfg["episodes"])
        durations_ep.append(avg_time)

        print(f"Średni czas jednej symulacji: {avg_time:.4f} sekundy")

    # Formatowanie wykresu nagród
    plt.xlabel("Epizody")
    plt.ylabel("Średnia suma nagród")
    plt.title("Q-learning – średnia z 100 treningów (α=0.1)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Drugi wykres: średni czas vs liczba epizodów
    plt.figure(figsize=(10, 4))
    plt.plot(episode_counts, durations_ep, marker='o')  # Wykres punktowy (czas vs liczba epizodów)
    plt.xlabel("Liczba epizodów")
    plt.ylabel("Średni czas [s]")
    plt.title("Średni czas treningu vs liczba epizodów (α=0.1)")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Test 2: badanie wpływu współczynnika uczenia alpha (przy stałej liczbie epizodów)
    configs_alpha = [
        {"alpha": 0.01, "episodes": 300},
        {"alpha": 0.02, "episodes": 300},
        {"alpha": 0.04, "episodes": 300},
        {"alpha": 0.08, "episodes": 300},
        {"alpha": 0.16, "episodes": 300},
        {"alpha": 0.32, "episodes": 300},
        {"alpha": 0.64, "episodes": 300},
        {"alpha": 1.28, "episodes": 300}
    ]

    alphas = []           # Lista wartości alpha
    durations_alpha = []  # Lista czasów dla każdego alpha

    print("\n### Test 2: Średnia z 100 treningów – różne alpha, epizody=300 ###\n")
    plt.figure(figsize=(14, 6))  # Przygotowanie wykresu nagród

    for cfg in configs_alpha:
        print(f"Trening: alpha={cfg['alpha']}, episodes={cfg['episodes']}")

        # Trening i uśrednianie wyników dla danej wartości alpha
        avg_rewards, avg_time = avg_q_learning_rewards(env, alpha=cfg["alpha"], episodes=cfg["episodes"], runs=100)

        # Wykres nagród
        plt.plot(avg_rewards, label=f"α={cfg['alpha']}")

        # Zapisywanie wartości alpha i czasu trwania treningu
        alphas.append(cfg["alpha"])
        durations_alpha.append(avg_time)

        print(f"Średni czas jednej symulacji: {avg_time:.4f} sekundy")

    # Formatowanie wykresu nagród
    plt.xlabel("Epizody")
    plt.ylabel("Średnia suma nagród")
    plt.title("Q-learning – średnia z 100 treningów (epizody=300)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Drugi wykres: czas vs alpha
    plt.figure(figsize=(10, 4))
    plt.plot(alphas, durations_alpha, marker='o')  # Wykres czasów w zależności od alpha
    plt.xlabel("Alpha (współczynnik uczenia)")
    plt.ylabel("Średni czas [s]")
    plt.title("Średni czas treningu vs alpha (epizody=300)")
    plt.grid()
    plt.tight_layout()
    plt.show()


