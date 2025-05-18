import numpy as np
import matplotlib.pyplot as plt
import time


def laplace(x, mu=0, b=1):
    """
    Oblicza wartość funkcji gęstości rozkładu Laplace'a.

    Parametry:
        x (float): Wartość zmiennej losowej.
        mu (float, opcjonalnie): Wartość oczekiwana (średnia) rozkładu. Domyślnie 0.
        b (float, opcjonalnie): Parametr skali (rozrzut) rozkładu. Domyślnie 1.

    Zwraca:
        float: Wartość funkcji gęstości dla podanej wartości x.
    """
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)


def sigmoid(x):
    """
    Oblicza wartość funkcji sigmoid.

    Parametry:
        x (float lub ndarray): Wartość wejściowa lub tablica wartości.

    Zwraca:
        float lub ndarray: Wartość funkcji sigmoid dla podanego argumentu x.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Oblicza pochodną funkcji sigmoid.

    Parametry:
        x (float lub ndarray): Wartość wejściowa lub tablica wartości.

    Zwraca:
        float lub ndarray: Wartość pochodnej funkcji sigmoid dla podanego argumentu x.
    """
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron2Layers:
    """
    Dwuwarstwowy perceptron do uczenia maszynowego.

    Atrybuty:
        learning_rate (float): Współczynnik uczenia.
        weights_to_hidden (ndarray): Wagi połączeń wejściowych do warstwy ukrytej.
        bias_to_hidden (ndarray): Bias warstwy ukrytej.
        weights_to_out (ndarray): Wagi połączeń warstwy ukrytej do wyjściowej.
        bias_to_out (ndarray): Bias warstwy wyjściowej.

    Metody:
        __init__(input_size, hidden_size, learning_rate):
            Inicjalizuje perceptron z losowymi wagami i zerowym biasem.

        forward(X):
            Przeprowadza propagację w przód przez sieć neuronową.
            Zwraca wynik wyjściowy.

        backward(X, y, output):
            Przeprowadza propagację wsteczną i aktualizuje wagi oraz biasy
            na podstawie błędu.

        train(X, y, epochs):
            Trenuje perceptron przez zadany licznik epok.

        predict(X):
            Przewiduje wyjście na podstawie danych wejściowych.
    """

    def __init__(self, input_size, hidden_size, learning_rate):
        """
        Inicjalizuje perceptron dwuwarstwowy.

        Parametry:
            input_size (int): Liczba neuronów na warstwie wejściowej.
            hidden_size (int): Liczba neuronów na warstwie ukrytej.
            learning_rate (float): Współczynnik uczenia.
        """
        self.learning_rate = learning_rate
        self.weights_to_hidden = np.random.randn(input_size, hidden_size)
        self.bias_to_hidden = np.zeros((1, hidden_size))
        self.weights_to_out = np.random.randn(hidden_size, 1)
        self.bias_to_out = np.zeros((1, 1))

    def forward(self, X):
        """
        Wykonuje propagację w przód.

        Parametry:
            X (ndarray): Dane wejściowe.

        Zwraca:
            ndarray: Wynik działania sieci neuronowej.
        """
        self.results_from_hidden = X @ self.weights_to_hidden + self.bias_to_hidden
        self.after_sigmoid = sigmoid(self.results_from_hidden)
        self.outer_results = self.after_sigmoid @ self.weights_to_out + self.bias_to_out
        return self.outer_results

    def backward(self, X, y, output):
        """
        Wykonuje propagację wsteczną i aktualizuje wagi.

        Parametry:
            X (ndarray): Dane wejściowe.
            y (ndarray): Prawdziwe wartości wyjściowe.
            output (ndarray): Wynik działania sieci.
        """
        samples = X.shape[0]
        d_outer_results_2 = (output - y)
        d_weights_to_out_2 = self.after_sigmoid.T @ d_outer_results_2 / samples
        d_bias_to_out_2 = np.sum(d_outer_results_2, axis=0, keepdims=True) / samples

        d_results_from_hidden_1 = d_outer_results_2 @ self.weights_to_out.T * sigmoid_derivative(self.results_from_hidden)
        d_weights_to_hidden_1 = X.T @ d_results_from_hidden_1 / samples
        d_bias_to_hidden_1 = np.sum(d_results_from_hidden_1, axis=0, keepdims=True) / samples

        # Aktualizacja wag metodą gradientową
        self.weights_to_hidden -= self.learning_rate * d_weights_to_hidden_1
        self.bias_to_hidden -= self.learning_rate * d_bias_to_hidden_1
        self.weights_to_out -= self.learning_rate * d_weights_to_out_2
        self.bias_to_out -= self.learning_rate * d_bias_to_out_2

    def train(self, X, y, epochs):
        """
        Trenuje perceptron przez określoną liczbę epok.

        Parametry:
            X (ndarray): Dane wejściowe.
            y (ndarray): Prawdziwe wartości wyjściowe.
            epochs (int): Liczba epok treningowych.
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        """
        Przewiduje wyjście na podstawie danych wejściowych.

        Parametry:
            X (ndarray): Dane wejściowe.

        Zwraca:
            ndarray: Przewidywane wyniki.
        """
        return self.forward(X)



# 1. Test ilości epok (iteracji) - działanie dla 10 ukrytych neuronów (uśrednianie z 25 powtórzeń)
def test_epochs(x, y):
    """
    Testuje wpływ liczby epok na dokładność modelu perceptronowego.

    Funkcja wykonuje eksperyment polegający na trenowaniu perceptronu
    dwuwarstwowego dla różnych liczb epok i oblicza średnie wartości MSE, MAE
    oraz czas treningu. Wyniki są uśredniane z 25 powtórzeń. Dodatkowo
    funkcja generuje wykres rzeczywistej funkcji Laplace'a oraz wyników
    sieci neuronowej dla różnych liczby epok.

    Parametry:
        x (ndarray): Dane wejściowe (np. wartości funkcji Laplace'a).
        y (ndarray): Rzeczywiste wartości funkcji Laplace'a.

    Wynik:
        Wykres porównujący rzeczywistą funkcję Laplace'a i wyniki predykcji
        dla różnych liczby epok. Wypisane wartości MSE, MAE i czas treningu
        dla każdej liczby epok.
    """
    plt.plot(x, y, label="Rzeczywista funkcja Laplace'a")
    plt.xlabel('x')
    plt.ylabel('y')
    hidden_neurons = 10
    epochs_amount = [10, 50, 100, 500, 1000, 2500, 5000, 10000, 25000]
    learning_rate = 0.2
    results_rate = []
    REPETITIONS = 25
    for epochs in epochs_amount:
        mse_sum = 0
        mae_sum = 0
        time_elapsed_sum = 0
        all_y_preds = []
        for i in range(REPETITIONS):
            model = Perceptron2Layers(input_size=1, hidden_size=hidden_neurons, learning_rate=learning_rate)
            start = time.time()
            model.train(x, y, epochs=epochs)
            end = time.time()
            time_elapsed = end - start
            y_pred = model.predict(x)
            all_y_preds.append(y_pred)

            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))

            mse_sum += mse
            mae_sum += mae
            time_elapsed_sum += time_elapsed

        mse_final = np.round(mse_sum / REPETITIONS, 4)
        mae_final = np.round(mae_sum / REPETITIONS, 4)
        time_final = np.round(time_elapsed_sum / REPETITIONS, 4)
        results_rate.append((mse_final, mae_final))
        print(f"------ WYNIKI DLA epochs = {epochs}: avg_MSE = {mse_final} | avg_MAE = {mae_final} | avg_TIME = {time_final} s")

        all_y_preds_array = np.stack(all_y_preds)
        avg_y_pred = np.mean(all_y_preds_array, axis=0)
        # Wykres
        plt.plot(x, avg_y_pred, label=f"Sieć neuronowa epochs = {epochs} | MSE={mse_final} | MAE={mae_final}", linestyle='dashed')
        plt.legend()
        plt.title(f"Odwzorowanie funkcji dla różnej liczby iteracji (epok)")

    plt.show()


# 2. Test learning rate - działanie dla 10 ukrytych neuronów dla liczby iteracji (uśrednianie z 25 powtórzeń)
def test_learning_rate(x, y):
    """
    Testuje wpływ współczynnika uczenia na dokładność modelu perceptronowego.

    Funkcja przeprowadza eksperyment, w którym trenuje perceptron dwuwarstwowy
    dla różnych wartości współczynnika uczenia (learning rate) oraz oblicza średnie
    wartości MSE i MAE. Wyniki są uśredniane z 25 powtórzeń. Dodatkowo funkcja
    generuje wykres rzeczywistej funkcji Laplace'a oraz wyników sieci neuronowej
    dla różnych wartości learning rate.

    Parametry:
        x (ndarray): Dane wejściowe (np. wartości funkcji Laplace'a).
        y (ndarray): Rzeczywiste wartości funkcji Laplace'a.

    Wynik:
        Wykres porównujący rzeczywistą funkcję Laplace'a i wyniki predykcji
        dla różnych wartości learning rate. Wypisane wartości MSE i MAE
        dla każdego współczynnika uczenia.
    """
    plt.plot(x, y, label="Rzeczywista funkcja Laplace'a")
    plt.xlabel('x')
    plt.ylabel('y')
    hidden_neurons = 10
    rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results_rate = []
    REPETITIONS = 25
    for rate in rates:
        mse_sum = 0
        mae_sum = 0
        all_y_preds = []
        for i in range(REPETITIONS):
            model = Perceptron2Layers(input_size=1, hidden_size=hidden_neurons, learning_rate=rate)
            model.train(x, y, epochs=5000)
            y_pred = model.predict(x)
            all_y_preds.append(y_pred)

            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))

            mse_sum += mse
            mae_sum += mae

        mse_final = np.round(mse_sum / REPETITIONS, 4)
        mae_final = np.round(mae_sum / REPETITIONS, 4)
        results_rate.append((mse_final, mae_final))
        print(f"------ WYNIKI DLA learning_rate = {rate}: avg_MSE = {mse_final} | avg_MAE = {mae_final}")

        all_y_preds_array = np.stack(all_y_preds)
        avg_y_pred = np.mean(all_y_preds_array, axis=0)
        # Wykres
        plt.plot(x, avg_y_pred, label=f"Sieć neuronowa lr = {rate} | MSE={mse_final} | MAE={mae_final}", linestyle='dashed')
        plt.legend()
        plt.title(f"Odwzorowanie funkcji dla różnych szybkości uczenia")

    plt.show()



# 3. Test hidden_neurons - działanie dla różnej liczby neuronów w ukrytej warstwie (5000 epok, 0.1 learning_rate)
# (uśrednianie z 25 powtórzeń)
def test_hidden_neurons(x, y):
    """
    Testuje wpływ liczby neuronów w warstwie ukrytej na dokładność modelu perceptronowego.

    Funkcja przeprowadza eksperyment, w którym trenuje perceptron dwuwarstwowy
    dla różnych liczby neuronów w warstwie ukrytej oraz oblicza średnie wartości
    MSE, MAE oraz czas treningu. Wyniki są uśredniane z 25 powtórzeń.
    Dodatkowo funkcja generuje wykres rzeczywistej funkcji Laplace'a oraz wyników
    sieci neuronowej dla różnych liczby neuronów w warstwie ukrytej.

    Parametry:
        x (ndarray): Dane wejściowe (np. wartości funkcji Laplace'a).
        y (ndarray): Rzeczywiste wartości funkcji Laplace'a.

    Wynik:
        Wykresy:
        - Porównanie rzeczywistej funkcji Laplace'a i wyników predykcji
          dla różnych liczby neuronów w warstwie ukrytej.
        - Wykres błędu MSE w zależności od liczby neuronów w warstwie ukrytej.
        - Wykres błędu MAE w zależności od liczby neuronów w warstwie ukrytej.
        - Wykres średniego czasu treningu w zależności od liczby neuronów.
        Wypisane wartości MSE, MAE i czas treningu dla każdej liczby neuronów.
    """
    plt.plot(x, y, label="Rzeczywista funkcja Laplace'a")
    plt.xlabel('x')
    plt.ylabel('y')
    # Ustalone eksperymentalnie parametry wejściowe (epochs, learning_rate)
    epochs = 5000
    learning_rate = 0.1

    hidden_neurons_amount = [1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60]
    results_rate = []
    mse_array = []
    mae_array = []
    time_array = []
    REPETITIONS = 25
    for hidden_neurons in hidden_neurons_amount:
        mse_sum = 0
        mae_sum = 0
        time_elapsed_sum = 0
        all_y_preds = []
        for i in range(REPETITIONS):
            model = Perceptron2Layers(input_size=1, hidden_size=hidden_neurons, learning_rate=learning_rate)
            start = time.time()
            model.train(x, y, epochs=epochs)
            end = time.time()
            time_elapsed = end - start
            y_pred = model.predict(x)
            all_y_preds.append(y_pred)

            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))

            mse_sum += mse
            mae_sum += mae
            time_elapsed_sum += time_elapsed

        mse_final = np.round(mse_sum / REPETITIONS, 4)
        mae_final = np.round(mae_sum / REPETITIONS, 4)
        mse_array.append(mse_final)
        mae_array.append(mae_final)
        time_final = np.round(time_elapsed_sum / REPETITIONS, 4)
        time_array.append(time_final)
        results_rate.append((mse_final, mae_final))
        print(f"------ WYNIKI DLA hidden_neurons = {hidden_neurons}: avg_MSE = {mse_final} | avg_MAE = {mae_final} | avg_TIME = {time_final} s")

        all_y_preds_array = np.stack(all_y_preds)
        avg_y_pred = np.mean(all_y_preds_array, axis=0)
        # Wykres
        plt.plot(x, avg_y_pred, label=f"Sieć neuronowa hidden_neurons = {hidden_neurons} | MSE={mse_final} | MAE={mae_final}", linestyle='dashed')
        plt.legend()
        plt.title(f"Odwzorowanie funkcji dla różnej liczby neuronów w ukrytej warstwie")

    plt.show()

    # Wykres MSE
    plt.xlabel('Liczba neuronów w ukrytej warstwie')
    plt.ylabel('MSE')
    plt.plot(hidden_neurons_amount, mse_array, label="Błąd MSE", linestyle='--', marker='o', color='blue')
    plt.legend()
    plt.title("Błąd MSE w zależności od liczby neuronów w ukrytej warstwie")
    plt.show()

    # Wykres MAE
    plt.xlabel('Liczba neuronów w ukrytej warstwie')
    plt.ylabel('MAE')
    plt.plot(hidden_neurons_amount, mae_array, label="Błąd MAE", linestyle='--', marker='o', color='red')
    plt.legend()
    plt.title("Błąd MAE w zależności od liczby neuronów w ukrytej warstwie")
    plt.show()

    # Wykres średniego czasu treningu
    plt.xlabel('Liczba neuronów w ukrytej warstwie')
    plt.ylabel('AVG_TIME [s]')
    plt.plot(hidden_neurons_amount, time_array, label="Średni czas treningu", linestyle='--', marker='o', color='green')
    plt.legend()
    plt.title("Średni czas treningu w zależności od liczby neuronów w ukrytej warstwie")
    plt.show()



# ================= TESTOWANIE =================
if __name__ == "__main__":
    while True:
        while True:
            ans = input("Wybór testu: \n1 - liczba iteracji, \n2 - szybkość uczenia, \n3 - liczba neuronów ukrytych.\nPodaj wartość tutaj: ")
            if ans.isdigit():
                ans = int(ans)
                if ans in (1, 2, 3):
                    break
                else:
                    print("Podaj poprawną wartość!")
            else:
                print("Podaj poprawną wartość liczbową!")
        x = np.linspace(-8, 8, 300).reshape(-1, 1)
        y = laplace(x, mu=0, b=1)

        if ans == 1:
            print("========== TEST 1 - dobór liczby iteracji (epok) ==========")
            test_epochs(x, y)
        if ans == 2:
            print("========== TEST 2 - dobór szybkości uczenia ==========")
            test_learning_rate(x, y)
        if ans == 3:
            print("========== TEST 3 - liczba neutronów w warstwie ukrytej ==========")
            test_hidden_neurons(x, y)
        while True:
            ans2 = input("Czy chcesz jeszcze raz wykonać jakiś test (t/n)? ")
            if ans2 == 't':
                break
            elif ans2 == 'n':
                print("Kończenie programu...")
                exit(-1)
            else:
                print("Błędna wartość!")


#rysowanie sieci:
# import matplotlib.pyplot as plt
# import numpy as np
#
# def draw_neural_network_v3(input_size, hidden_size, output_size):
#     """
#     Rysuje diagram sieci neuronowej z jedną warstwą ukrytą i sigmoidą jako funkcją aktywacyjną.
#     Neurony jako kółka, wyrównanie inputu i outputu na tej samej wysokości.
#
#     Parametry:
#         input_size (int): Liczba neuronów wejściowych.
#         hidden_size (int): Liczba neuronów w warstwie ukrytej.
#         output_size (int): Liczba neuronów wyjściowych.
#     """
#     fig, ax = plt.subplots(figsize=(8, 8))
#
#     input_layer_y = np.array([0])  # Input na tej samej wysokości co output
#     hidden_layer_y = np.linspace(-2, 2, hidden_size)
#     output_layer_y = np.array([0])
#
#     for i, y in enumerate(input_layer_y):
#         circle = plt.Circle((-1, y), 0.2, color='blue', ec='black', zorder=2)
#         ax.add_artist(circle)
#         ax.text(-1, y, f'Input', ha='center', va='center', fontsize=12, color='white')
#
#     for j, y in enumerate(hidden_layer_y):
#         circle = plt.Circle((0, y), 0.2, color='orange', ec='black', zorder=2)
#         ax.add_artist(circle)
#         ax.text(0, y, f'Σ', ha='center', va='center', fontsize=18, fontweight='bold', color='white')
#
#     circle = plt.Circle((1, output_layer_y[0]), 0.2, color='green', ec='black', zorder=2)
#     ax.add_artist(circle)
#     ax.text(1, output_layer_y[0], 'Σ', ha='center', va='center', fontsize=18, fontweight='bold', color='white')
#
#     for i, y1 in enumerate(input_layer_y):
#         for j, y2 in enumerate(hidden_layer_y):
#             ax.plot([-0.8, -0.2], [y1, y2], color='black', alpha=0.7, linestyle='--')
#
#     for j, y2 in enumerate(hidden_layer_y):
#         ax.plot([0.2, 0.8], [y2, output_layer_y[0]], color='black', alpha=0.7, linestyle='--')
#
#     ax.text(0, -3, 'Funkcja aktywacyjna: sigmoid(x) = 1 / (1 + exp(-x))',
#             ha='center', va='center', fontsize=12, color='red')
#     ax.text(0, -3.3, 'Każdy neuron ukryty: Σ(w * x) + b -> sigmoid',
#             ha='center', va='center', fontsize=12, color='black')
#     ax.text(0, -3.6, 'Neuron wyjściowy: Σ(w * hidden_output) + b -> sigmoid',
#             ha='center', va='center', fontsize=12, color='black')
#
#     ax.axis('off')
#     plt.title("Diagram sieci neuronowej: 1 input, 10 hidden neurons, 1 output (Sigmoida)")
#     plt.show()
#
# draw_neural_network_v3(input_size=1, hidden_size=10, output_size=1)
