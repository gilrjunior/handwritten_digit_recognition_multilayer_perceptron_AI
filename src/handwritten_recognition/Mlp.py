import numpy as np
import random as rd
import matplotlib.pyplot as plt

class Mlp:
    def __init__(self, number_neurons, sample_size, x_min, x_max, learning_rate):
        self.number_neurons = number_neurons
        self.sample_size = sample_size
        self.x_min = x_min
        self.x_max = x_max
        self.learning_rate = learning_rate
        
        # Camada oculta: vi (bias) e wi (pesos) - cada um com shape (N,)
        self.vi = np.random.uniform(-0.5, 0.5, self.number_neurons)
        self.wi = np.random.uniform(-0.5, 0.5, self.number_neurons)
        
        # Camada de saída: vy (bias) escalar e wy (pesos) com shape (N,)
        self.vy = rd.uniform(-0.5, 0.5)
        self.wy = np.random.uniform(-0.5, 0.5, self.number_neurons)

        self.inputs = self.get_inputs(x_min, x_max, sample_size)
        self.targets = self.get_targets(self.inputs)

    def get_inputs(self, x_min, x_max, sample_size):
        # Retorna um array 1D de amostras entre x_min e x_max
        return np.linspace(x_min, x_max, sample_size)

    def get_targets(self, inputs):
        # Gera os valores desejados (targets) para cada entrada
        # f(x) = sin(x/2)*cos(2x)
        targets = []
        for x in inputs:
            targets.append(np.sin(x/2)*np.cos(2*x))
        return np.array(targets)
    
    def forward(self, x):
        # 1) Camada oculta
        zin = []
        for j in range(self.number_neurons):
            net_in_j = self.vi[j] + self.wi[j] * x
            z_out_j = np.tanh(net_in_j)
            zin.append(z_out_j)  # Reaproveitando o array zin só para exemplo

        # 2) Camada de saída
        sum_value = 0.0
        for k in range(self.number_neurons):
            sum_value += zin[k] * self.wy[k]
        yin = self.vy + sum_value
        y = np.tanh(yin)
        return y
    
    def get_approximation(self):
        outputs = []
        for x in self.inputs:
            y = self.forward(x)
            outputs.append(y)
        return np.array(outputs)

    def optimized_train(self, min_error, update_callback=None, should_stop_callback=None):
        epochs = 0
        number_entries = len(self.inputs)
        error_history = []

        while epochs <= 10000:
            epochs += 1
            epoch_error = 0.0

            # Loop por amostra
            for i in range(number_entries):
                x = self.inputs[i]
                t = self.targets[i]

                # Feedforward
                # net_in (shape: (N,)) = bias + peso * x
                net_in = self.vi + self.wi * x
                # z (shape: (N,)) = tanh(net_in)
                z = np.tanh(net_in)

                # Saída: soma ponderada + bias
                # sum_value é escalar = z . wy
                sum_value = np.dot(z, self.wy)
                yin = self.vy + sum_value
                y = np.tanh(yin)

                # Erro quadrático para essa amostra
                sample_error = 0.5 * (t - y)**2
                epoch_error += sample_error

                # BACKPROPAGATION (cálculo dos gradientes)
                # delta_k = (y - t) * derivada da tanh(yin)
                delta_k = (t - y) * (1 - y**2)  # pois y = tanh(yin), deriv = (1 - tanh^2(yin))

                # Camada de saída (oculta -> saída)
                # gradientes: delta_wy (shape: (N,)) e delta_vy (escalar)
                delta_wy = self.learning_rate * delta_k * z
                delta_vy = self.learning_rate * delta_k

                # Atualização
                self.wy += delta_wy
                self.vy += delta_vy

                # Camada oculta (entrada -> oculta)
                # erro que chega a cada neurônio oculto j: delta_in_j = wy[j] * delta_k
                delta_in = self.wy * delta_k  # shape: (N,)
                # derivada da tanh(net_in[j]) = (1 - z[j]^2)
                delta_j = delta_in * (1 - z**2)  # shape: (N,)

                # gradiente dos pesos de entrada
                # wi[j] recebe a correção: eta * delta_j[j] * x
                delta_wi = self.learning_rate * delta_j * x  # shape: (N,)
                # gradiente do bias
                delta_vi = self.learning_rate * delta_j      # shape: (N,)

                # Atualiza
                self.wi += delta_wi
                self.vi += delta_vi

            # Armazena erro total da época
            error_history.append(epoch_error)

            # Callback para atualizar interface (se houver)
            if update_callback:
                approx = self.get_approximation()
                update_callback(epochs, error_history, approx)

            # Verifica se devemos parar (se o usuário clicou em "Parar")
            if should_stop_callback and should_stop_callback():
                print("Treinamento interrompido pelo usuário.")
                break

            # Critério de parada
            if epoch_error <= min_error:
                break

        print("Treinamento finalizado em", epochs, "épocas.")

    def train(self, min_error, update_callback=None, should_stop_callback=None):

        epochs = 0
        number_entries = len(self.inputs)
        error_history = [] # Histórico de erros para plotar o gráfico que será montado em tempo real na interface

        while epochs <= 1000:
            epochs += 1
            epoch_error = 0.0 # Erro de cada época que é zerado novamente a cada época
            # Percorre cada amostra de treinamento
            for i in range(number_entries):

                # Vetores para armazenar valores de entrada (zin) e saída (z)
                # de cada neurônio da camada oculta.
                zin = []
                z = []

                # 1) Camada Oculta: calcula saída de cada neurônio
                for j in range(self.number_neurons):
                    # net input do neurônio j:
                    net_in_j = self.vi[j] + self.wi[j] * self.inputs[i]
                    zin.append(net_in_j)

                    # saída do neurônio j após ativação tanh:
                    z_out_j = np.tanh(net_in_j)
                    z.append(z_out_j)

                # 2) Camada de Saída: combina as saídas dos neurônios ocultos
                sum_value = 0  # zera o acumulador para calcular a soma ponderada
                for k in range(self.number_neurons):
                    sum_value += z[k] * self.wy[k]

                # net input da saída
                yin = self.vy + sum_value
                # saída final (após ativação tanh)
                y = np.tanh(yin)

                # Erro quadrático para a amostra atual
                sample_error = 0.5 * ((y - self.targets[i]) ** 2)

                # Atualiza erro da época, somando o erro da amostra atual aos anteriormente calculados
                epoch_error += sample_error
                # epoch_error = sample_error

                # delta_k: derivada do erro em relação à saída (para o neurônio de saída)
                # (y - target) * derivada da tanh (1 - tanh^2(yin))
                delta_k = (self.targets[i] - y) * (1 - np.tanh(yin) ** 2)

                # Atualização dos pesos da camada de oculta x saída
                for j in range(self.number_neurons):
                    # Gradiente do peso que liga o neurônio oculto j à saída
                    delta_wy = self.learning_rate * delta_k * z[j]
                    # Gradiente do bias da saída
                    delta_vy = self.learning_rate * delta_k

                    # Atualiza pesos e bias
                    self.wy[j] += delta_wy
                    self.vy += delta_vy

                # Cada neurônio j na camada oculta recebe parte do erro vindo da saída
                for j in range(self.number_neurons):
                    # delta_in = erro que chega ao neurônio j
                    # como há apenas 1 neurônio de saída, é delta_k * peso_que_liga_j_à_saída
                    delta_in = self.wy[j] * delta_k

                    # delta_j = delta_in * derivada da ativação tanh(zin[j])
                    delta_j = delta_in * (1 - np.tanh(zin[j]) ** 2)

                    # Gradientes para os pesos de entrada do neurônio j
                    delta_wi = self.learning_rate * delta_j * self.inputs[i]
                    delta_vi = self.learning_rate * delta_j

                    # Atualiza pesos e bias do neurônio j na camada oculta x entrada
                    self.wi[j] += delta_wi
                    self.vi[j] += delta_vi

            # Armazena erro total da época
            error_history.append(epoch_error)

            # Callback para atualizar interface (se houver)
            if update_callback:
                approx = self.get_approximation()
                update_callback(epochs, error_history, approx)

            # Verifica se devemos parar (se o usuário clicou em "Parar")
            if should_stop_callback and should_stop_callback():
                print("Treinamento interrompido pelo usuário.")
                break

            # Critério de parada
            if epoch_error <= min_error:
                break

        print("Treinamento finalizado em", epochs, "épocas.")
