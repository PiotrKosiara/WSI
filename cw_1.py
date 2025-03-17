import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as anp


# https://staff.elka.pw.edu.pl/~knalecz/WSI/cw1.html
# https://staff.elka.pw.edu.pl/~knalecz/
#Do poprawy przeźroczystośc dla kropek

def gradient_descent(f, x0, learning_rate, max_iter=1000, tol=1e-6):
    '''
        Implementacja algorytmu spadku gradientowego (Gradient Descent).

        Parametry:
        - f : funkcja celu, którą chcemy zminimalizować.
        - x0 : lista lub tablica NumPy reprezentująca punkt startowy [x1, x2, ...].
        - learning_rate : współczynnik uczenia (krok gradientu).
        - max_iter : maksymalna liczba iteracji (domyślnie 1000).
        - tol : tolerancja normy gradientu do zatrzymania algorytmu (domyślnie 1e-6).

        Zwraca:
        - x : znalezione minimum lokalne funkcji.
        - history : lista zawierająca historię punktów (x, f(x)) w trakcie optymalizacji.
        - iterations : lista numerów iteracji odpowiadających historii.

        Algorytm iteracyjnie aktualizuje wartość x zgodnie ze wzorem:
            x = x - learning_rate * grad(f)
        aż do spełnienia kryterium stopu (mały gradient lub osiągnięcie max_iter).
        '''
    grad_f = grad(f)
    x = anp.array(x0, dtype=anp.float64)
    history = []
    iterations = []

    for i in range(max_iter):
        grad_value = grad_f(x)
        history.append((x.copy(), f(x)))
        iterations.append(i)

        if anp.linalg.norm(grad_value) < tol:
            break

        x = x - learning_rate * grad_value

    return x, history, iterations


# Funkcja 1: f(x) = x1^2 + x2^2
def f1(x):
    return x[0] ** 2 + x[1] ** 2


# Funkcja Matyasa
def f2(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


# Sprawdzane wartości
learning_rates = [0.01, 0.1, 0.25, 0.5, 0.8]
initial_points = [[-8, 9], [7, -6], [5, 5], [-10, -10]]


# Test różnych kroków dla f1 dla punktu startowego [-8,9]
plt.figure(figsize=(12, 5))
for lr in learning_rates:
    _, history, iterations = gradient_descent(f1, [-8, 9], lr)
    variables, values = zip(*history)
    plt.plot(iterations, values, label=f"lr={lr}")

plt.xlabel("Nr iteracji")
plt.ylabel("Wartość funkcji celu")
plt.title("Wpływ kroku na zbieżność dla f1")
plt.legend()
plt.show()


# Test różnych kroków dla f1 (skala logarytmiczna) dla punktu startowego [-8,9]
plt.figure(figsize=(12, 5))
for lr in learning_rates:
    _, history, iterations = gradient_descent(f1, [-8, 9], lr)
    variables, values = zip(*history)
    plt.plot(iterations, values, label=f"lr={lr}")
plt.xscale("log")  # Ustawienie skali logarytmicznej dla numeru iteracji
plt.xlabel("Nr iteracji (log)")
plt.ylabel("Wartość funkcji celu")
plt.title("Wpływ kroku na zbieżność dla f1 (skala log)")
plt.legend()
plt.show()

# Wizualizacja trajektorii dla różnych punktów startowych
plt.figure(figsize=(12, 5))
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[f1([x, y]) for x in x_vals] for y in y_vals])
plt.contour(X, Y, Z, levels=50)
for x0 in initial_points:
    x_opt, history, iterations = gradient_descent(f1, x0, 0.25)
    x_hist, y_hist = zip(*[point for point, _ in history])
    plt.plot(x_hist, y_hist, marker='o', markersize=3, label=f"Start: {x0}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Trajektoria gradientu dla różnych punktów startowych (f1)")
plt.legend()
plt.show()

# Test różnych kroków dla f2 dla punktu startowego [-8,9]
plt.figure(figsize=(12, 5))
for lr in learning_rates:
    _, history, iterations = gradient_descent(f2, [-8, 9], lr)
    variables, values = zip(*history)
    plt.plot(iterations, values, label=f"lr={lr}")
plt.xlabel("Nr iteracji")
plt.ylabel("Wartość funkcji celu")
plt.title("Wpływ kroku na zbieżność dla f2")
plt.legend()
plt.show()

# Test różnych kroków dla f2 (skala logarytmiczna) dla punktu startowego [-8,9]
plt.figure(figsize=(12, 5))
for lr in learning_rates:
    _, history, iterations = gradient_descent(f2, [-8, 9], lr)
    variables, values = zip(*history)
    plt.plot(iterations, values, label=f"lr={lr}")
plt.xscale("log")  # Ustawienie skali logarytmicznej dla numeru iteracji
plt.xlabel("Nr iteracji (log)")
plt.ylabel("Wartość funkcji celu")
plt.title("Wpływ kroku na zbieżność dla f2 (skala log)")
plt.legend()
plt.show()

# Wizualizacja trajektorii dla różnych punktów startowych
plt.figure(figsize=(12, 5))
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[f2([x, y]) for x in x_vals] for y in y_vals])
plt.contour(X, Y, Z, levels=50)
for x0 in initial_points:
    x_opt, history, iterations = gradient_descent(f2, x0, 0.25)
    x_hist, y_hist = zip(*[point for point, _ in history])
    plt.plot(x_hist, y_hist, marker='o', markersize=3, label=f"Start: {x0}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Trajektoria gradientu dla różnych punktów startowych (f2)")
plt.legend()
plt.show()