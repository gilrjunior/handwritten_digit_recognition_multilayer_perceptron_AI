import numpy as np
import random as rd
import matplotlib.pyplot as plt
import handwritten_recognition.data_processing as dp

class Mlp:
    def __init__(self, number_neurons, learning_rate):

        self.number_neurons = number_neurons
        self.learning_rate = learning_rate
        self.threshold = 0.00

        # Carrega os dados
        self.inputs = dp.load_inputs()
        self.targets = dp.load_targets()
        
        # Camada oculta: vi (bias) e wi (pesos) - cada um com shape
        self.vi = np.random.uniform(-0.5, 0.5, (self.number_neurons, 1))
        self.wi = np.random.uniform(-0.5, 0.5, (self.number_neurons, self.inputs.shape[1]))
        
        # Camada de saída: vy (bias) escalar e wy (pesos)
        self.vy = np.random.uniform(-0.5, 0.5, (self.targets.shape[1], 1))
        self.wy = np.random.uniform(-0.5, 0.5, (self.targets.shape[1], self.number_neurons))

            
    def forward(self, x):
        #Testing
        # 1) Camada oculta
        net_in = self.vi + np.dot(self.wi,x)
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
        number_entries = self.inputs.shape[0]
        prev_epoch_error = float('inf')   # Erro da época anterior
        prev_delta_bar = 0.0       # Média móvel dos deltas
        
        # Parâmetros de adaptação (conforme o artigo)
        alpha = 0.1    # para a média móvel (0 < α ≤ 1)
        theta = 0.5   # limiar para pequenas oscilações
        u = 1.1        # fator de aumento (u > 1)
        d = 0.5        # fator de diminuição (0 < d < 1)

        while epochs <= 100000:
            epochs += 1
            epoch_error = 0.0

            # Loop por amostra
            for i in range(number_entries):
                
                # Entrada
                x = self.inputs[i].reshape(-1, 1) 

                # Saída desejada
                t = self.targets[i%10].reshape(-1, 1)

                # Feedforward
                # Camada oculta: soma ponderada + bias            
                sum_value = self.wi @ x 
                net_in = self.vi + sum_value
                z = 1.7159 * np.tanh((2.00/3.00) * net_in)

                # Saída: soma ponderada + bias
                # sum_value é escalar = z . wy
                sum_value = self.wy @ z
                yin = self.vy + sum_value
                y = 1.7159 * np.tanh((2.00/3.00) * yin)

                # Erro quadrático para essa amostra
                sample_error = 0.5 * np.sum((t - y)**2)
                epoch_error += sample_error

                # BACKPROPAGATION (cálculo dos gradientes)
                delta_k = (t - y) * 1.7159 * (2.00/3.00) * (1 - np.tanh((2.00/3.00) * yin)**2) 

                # Camada de saída (oculta -> saída)
                # gradiente dos pesos de saída
                delta_wy = self.learning_rate * delta_k @ z.T
                delta_vy = self.learning_rate * delta_k

                # Atualização
                self.wy += delta_wy
                self.vy += delta_vy  

                # Camada oculta (entrada -> oculta)
                # erro que chega a cada neurônio oculto j: delta_in_j = wy[j] * delta_k
                delta_in = self.wy.T @ delta_k
                delta_j = delta_in * 1.7159 * (2.0/3.0) * (1 - np.tanh((2.0/3.0) * net_in)**2)


                # gradiente dos pesos de entrada
                # wi[j] recebe a correção: eta * delta_j[j] * x
                delta_wi = self.learning_rate * delta_j @ x.T
                # gradiente do bias
                delta_vi = self.learning_rate * delta_j

                # Atualiza
                self.wi += delta_wi
                self.vi += delta_vi

            # Critério de parada
            print("Erro da época", epochs, ":", epoch_error)

            if epoch_error != 0:
                delta = (epoch_error - prev_epoch_error) / epoch_error

            delta_bar = alpha * delta + (1 - alpha) * prev_delta_bar

            # Regra principal para atualização da taxa de aprendizado:
            if delta * prev_delta_bar < 0 and abs(prev_delta_bar) > theta:
                self.learning_rate *= d
            else:
                self.learning_rate *= u

            # Prepara para a próxima iteração
            prev_delta_bar = delta_bar
            prev_epoch_error = epoch_error

            if epoch_error <= min_error:
                break

        print("Treinamento finalizado em", epochs, "épocas.")