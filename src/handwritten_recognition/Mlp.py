import numpy as np
import random as rd
import matplotlib.pyplot as plt
import handwritten_recognition.data_processing as dp

class Mlp:
    def __init__(self, number_neurons, sample_size, x_min, x_max, learning_rate):

        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.threshold = 0.00

        # Carrega os dados
        self.inputs = dp.load_inputs()
        self.targets = dp.load_targets()
        
        # Camada oculta: vi (bias) e wi (pesos) - cada um com shape
        self.vi = np.random.uniform(-0.5, 0.5, self.number_neurons)
        self.wi = np.random.uniform(-0.5, 0.5, self.inputs.shape[1], self.number_neurons)
        
        # Camada de saída: vy (bias) escalar e wy (pesos)
        self.vy = np.random.uniform(-0.5, 0.5, self.targets.shape[1])
        self.wy = np.random.uniform(-0.5, 0.5, self.number_neurons, self.targets.shape[1])

            
    def forward(self, x):
        #Testing
        # 1) Camada oculta
        net_in = self.vi + self.wi * x
        z_out = np.tanh(net_in)

        # 2) Camada de saída
        yin = self.vy + np.dot(z_out, self.wy)

        # 3) Saída
        y = np.tanh(yin)

        # Limiarização
        y = np.where(y > self.threshold, 1, -1)

        return y
    
    def get_approximation(self):
        
        return self.forward(self.inputs)
    
    def optimized_train(self, min_error):
        epochs = 0
        number_entries = len(self.inputs.shape[0])

        while epochs <= 10000:
            epochs += 1
            epoch_error = 0.0

            # Loop por amostra
            for i in range(number_entries):
                
                # Entrada
                x = self.inputs[i]

                # Saída desejada
                t = self.targets[i%10]

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

                # Limiarização
                y = np.where(y > self.threshold, 1, -1)

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

            # Critério de parada
            if epoch_error <= min_error:
                break

        print("Treinamento finalizado em", epochs, "épocas.")