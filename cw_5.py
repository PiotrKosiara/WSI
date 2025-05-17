import numpy as np
import matplotlib.pyplot as plt
import time


def laplace(x, mu=0, b=1):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mu) / b)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron2Layers:
    def __init__(self, input_size, hidden_size, learning_rate):
        self.learning_rate = learning_rate
        self.weights_to_hidden = np.random.randn(input_size, hidden_size)
        self.bias_to_hidden = np.zeros((1, hidden_size))
        self.weights_to_out = np.random.randn(hidden_size, 1)
        self.bias_to_out = np.zeros((1, 1))

    def forward(self, X):
        # @ - możenie macierzowe
        self.results_from_hidden = X @ self.weights_to_hidden + self.bias_to_hidden
        self.after_sigmoid = sigmoid(self.results_from_hidden)
        self.outer_results = self.after_sigmoid @ self.weights_to_out + self.bias_to_out
        return self.outer_results

    def backward(self, X, y, output):
        samples = X.shape[0]
        d_outer_results_2 = (output - y)
        d_weights_to_out_2 = self.after_sigmoid.T @ d_outer_results_2 / samples
        d_bias_to_out_2 = np.sum(d_outer_results_2, axis=0, keepdims=True) / samples

        d_results_from_hidden_1 = d_outer_results_2 @ self.weights_to_out.T * sigmoid_derivative(self.results_from_hidden)
        d_weights_to_hidden_1 = X.T @ d_results_from_hidden_1 / samples
        d_bias_to_hidden_1 = np.sum(d_results_from_hidden_1, axis=0, keepdims=True) / samples

        # Aktualizacja wag metodą gradientową (spadek)
        self.weights_to_hidden -= self.learning_rate * d_weights_to_hidden_1
        self.bias_to_hidden -= self.learning_rate * d_bias_to_hidden_1
        self.weights_to_out -= self.learning_rate * d_weights_to_out_2
        self.bias_to_out -= self.learning_rate * d_bias_to_out_2

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)


# 1. Test ilości epok (iteracji) - działanie dla 10 ukrytych neuronów (uśrednianie z 25 powtórzeń)
def test_epochs(x, y):
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


# 2. Test hidden_neurons - działanie dla różnej liczby neuronów w ukrytej warstwie (5000 epok, 0.1 learning_rate)
# (uśrednianie z 25 powtórzeń)
def test_hidden_neurons(x, y):
    plt.plot(x, y, label="Rzeczywista funkcja Laplace'a")
    plt.xlabel('x')
    plt.ylabel('y')
    epochs = 5000
    learning_rate = 0.2
    hidden_neurons_amount = [2, 4, 6, 8, 10, 15, 20, 25, 50]
    results_rate = []
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
        time_final = np.round(time_elapsed_sum / REPETITIONS, 4)
        results_rate.append((mse_final, mae_final))
        print(f"------ WYNIKI DLA hidden_neurons = {hidden_neurons}: avg_MSE = {mse_final} | avg_MAE = {mae_final} | avg_TIME = {time_final} s")

        all_y_preds_array = np.stack(all_y_preds)
        avg_y_pred = np.mean(all_y_preds_array, axis=0)
        # Wykres
        plt.plot(x, avg_y_pred, label=f"Sieć neuronowa hidden_neurons = {hidden_neurons} | MSE={mse_final} | MAE={mae_final}", linestyle='dashed')
        plt.legend()
        plt.title(f"Odwzorowanie funkcji dla różnej liczby neuronów w ukrytej warstwie")

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
                continue
            elif ans2 == 'n':
                print("Kończenie programu...")
                exit(-1)
            else:
                print("Błędna wartość!")

    #
    # # Obliczanie błędów
    # mse = np.mean((y - y_pred)**2)
    # mae = np.mean(np.abs(y - y_pred))
    #
    # print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
    #
    # # Wykres
    # plt.plot(x, y, label="Rzeczywista funkcja Laplace'a")
    # plt.plot(x, y_pred, label="Sieć neuronowa", linestyle='dashed')
    # plt.legend()
    # plt.title(f"Ukrytych neuronów: {hidden_neurons}")
    # plt.show()