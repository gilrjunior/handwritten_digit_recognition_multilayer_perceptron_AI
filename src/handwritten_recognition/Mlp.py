import glob
import numpy as np
import os
# import random as rd
# import matplotlib.pyplot as plt
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

        # Supondo que self.targets tenha shape (10,10)
        # num_samples = self.inputs.shape[0]  # 900
        # # Cria um array de índices que se repete de 0 a 9
        # indices = np.arange(num_samples) % 10
        # # Expande os targets para ter 900 linhas, cada uma sendo o one-hot correto
        # self.full_targets = self.targets[indices, :]  # shape (900, 10)
        self.expanded_targets = self.targets[np.arange(self.inputs.shape[0]) % 10, :]

            
    def forward(self, x):
        # 1) Camada oculta
        net_in = self.vi + np.dot(self.wi, x)
        z_out = 1.7159 * np.tanh((2.0 / 3.0) * net_in)

        # 2) Camada de saída
        yin = self.vy + np.dot(self.wy, z_out)
        y = 1.7159 * np.tanh((2.0 / 3.0) * yin)

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

                # Limiarização
                # y_l = np.where(y > self.threshold, 1, -1)

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

    def batch_train(self, min_error, batch_size=32):
        epochs = 0
        number_entries = self.inputs.shape[0]

        while epochs <= 100000:
            epochs += 1
            epoch_error = 0.0

            # Embaralha os dados a cada época para evitar viés na formação dos mini-batches
            indices = np.arange(number_entries)
            np.random.shuffle(indices)
            inputs_shuffled = self.inputs[indices]
            targets_shuffled = self.expanded_targets[indices]

            # Divide os dados em mini-batches
            for start in range(0, number_entries, batch_size):
                end = start + batch_size
                # Transpõe para ter cada mini-batch com dimensão (features, batch_size)
                batch_inputs = inputs_shuffled[start:end].T  
                batch_targets = targets_shuffled[start:end].T  

                # --- Feedforward ---
                # Camada Oculta
                net_in = self.vi + np.dot(self.wi, batch_inputs)  # (n_hidden, batch_size)
                z = 1.7159 * np.tanh((2.0/3.0) * net_in)
                
                # Camada de Saída
                yin = self.vy + np.dot(self.wy, z)  # (n_output, batch_size)
                y = 1.7159 * np.tanh((2.0/3.0) * yin)
                
                # Cálculo do erro para o mini-batch
                error = batch_targets - y
                batch_error = 0.5 * np.sum(error**2)
                epoch_error += batch_error

                # --- Backpropagation ---
                # Camada de Saída
                delta_k = error * 1.7159 * (2.0/3.0) * (1 - np.tanh((2.0/3.0)*yin)**2)
                delta_wy = self.learning_rate * np.dot(delta_k, z.T)  # (n_output, n_hidden)
                delta_vy = self.learning_rate * np.sum(delta_k, axis=1, keepdims=True)  # (n_output, 1)

                # Camada Oculta
                delta_in = np.dot(self.wy.T, delta_k)  # (n_hidden, batch_size)
                delta_j = delta_in * 1.7159 * (2.0/3.0) * (1 - np.tanh((2.0/3.0)*net_in)**2)
                delta_wi = self.learning_rate * np.dot(delta_j, batch_inputs.T)  # (n_hidden, n_features)
                delta_vi = self.learning_rate * np.sum(delta_j, axis=1, keepdims=True)  # (n_hidden, 1)

                # Atualiza os pesos e vieses
                self.wy += delta_wy
                self.vy += delta_vy
                self.wi += delta_wi
                self.vi += delta_vi

            print("Erro da época", epochs, ":", epoch_error)
            if epoch_error <= min_error:
                break

        print("Treinamento finalizado em", epochs, "épocas.")

    def batch_train_adaptive_lr_early_stopping(self, min_error, batch_size=32, alpha=0.1, theta=0.01, d=0.9, u=1.05, patience=10):
        # Realiza o split dos dados: 70% treino, 20% validação, 10% teste (teste não é usado aqui)
        num_samples = self.inputs.shape[0]
        n_train = int(num_samples * 0.7)
        n_val = int(num_samples * 0.2)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        X_train = self.inputs[train_indices]
        Y_train = self.expanded_targets[train_indices]
        X_val = self.inputs[val_indices]
        Y_val = self.expanded_targets[val_indices]

        epochs = 0
        best_val_error = float('inf')
        patience_counter = 0

        # # Inicializa variáveis para a atualização dinâmica do learning rate
        # prev_epoch_error = float('inf')
        # prev_delta_bar = 0

        # Opcional: guarda os melhores pesos para restaurar no early stopping
        best_wi, best_vi, best_wy, best_vy = self.wi.copy(), self.vi.copy(), self.wy.copy(), self.vy.copy()

        while epochs <= 100000:
            epochs += 1
            epoch_error = 0.0
            num_train = X_train.shape[0]

            # Embaralha os dados de treinamento
            train_perm = np.random.permutation(num_train)
            X_train_shuffled = X_train[train_perm]
            Y_train_shuffled = Y_train[train_perm]

            # Processa os dados em mini-batches
            for start in range(0, num_train, batch_size):
                end = start + batch_size
                # Transpõe para que cada mini-batch tenha shape (n_features, batch_size)
                batch_inputs = X_train_shuffled[start:end].T      
                batch_targets = Y_train_shuffled[start:end].T      

                # --- Feedforward ---
                net_in = self.vi + np.dot(self.wi, batch_inputs)
                z = 1.7159 * np.tanh((2.0/3.0) * net_in)
                yin = self.vy + np.dot(self.wy, z)
                y = 1.7159 * np.tanh((2.0/3.0) * yin)

                # Calcula o erro para o mini-batch
                error = batch_targets - y
                batch_error = 0.5 * np.sum(error**2)
                epoch_error += batch_error

                # --- Backpropagation ---
                delta_k = error * 1.7159 * (2.0/3.0) * (1 - np.tanh((2.0/3.0)*yin)**2)
                delta_wy = self.learning_rate * np.dot(delta_k, z.T)
                delta_vy = self.learning_rate * np.sum(delta_k, axis=1, keepdims=True)
                delta_in = np.dot(self.wy.T, delta_k)
                delta_j = delta_in * 1.7159 * (2.0/3.0) * (1 - np.tanh((2.0/3.0)*net_in)**2)
                delta_wi = self.learning_rate * np.dot(delta_j, batch_inputs.T)
                delta_vi = self.learning_rate * np.sum(delta_j, axis=1, keepdims=True)

                # Atualiza os pesos e vieses
                self.wy += delta_wy
                self.vy += delta_vy
                self.wi += delta_wi
                self.vi += delta_vi

            # # --- Atualização dinâmica do learning rate ---
            # if epoch_error != 0:
            #     delta = (epoch_error - prev_epoch_error) / epoch_error
            # else:
            #     delta = 0

            # delta_bar = alpha * delta + (1 - alpha) * prev_delta_bar

            # if delta * prev_delta_bar < 0 and abs(prev_delta_bar) > theta:
            #     self.learning_rate *= d
            # else:
            #     self.learning_rate *= u

            # prev_delta_bar = delta_bar
            # prev_epoch_error = epoch_error

            # --- Avaliação no conjunto de validação ---
            net_in_val = self.vi + np.dot(self.wi, X_val.T)
            z_val = 1.7159 * np.tanh((2.0/3.0) * net_in_val)
            yin_val = self.vy + np.dot(self.wy, z_val)
            y_val = 1.7159 * np.tanh((2.0/3.0) * yin_val)
            error_val = Y_val.T - y_val
            val_error = 0.5 * np.sum(error_val**2)

            print(f"Época {epochs} - Erro Treino: {epoch_error:.4f} | Erro Validação: {val_error:.4f} | Learning Rate: {self.learning_rate:.6f}")

            # --- Early stopping ---
            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = 0
                # Armazena os melhores pesos
                best_wi = self.wi.copy()
                best_vi = self.vi.copy()
                best_wy = self.wy.copy()
                best_vy = self.vy.copy()
            else:
                patience_counter += 1

            # if (epoch_error <= min_error * 10) and (patience_counter >= patience or epoch_error <= min_error): # Early stopping só é ativado após 10x o erro mínimo
            if (epoch_error <= min_error) and (patience_counter >= patience):
                print("Early stopping acionado. Treinamento interrompido na época", epochs)
                # Opcional: restaura os pesos do melhor modelo
                self.wi = best_wi
                self.vi = best_vi
                self.wy = best_wy
                self.vy = best_vy
                break

        print("Treinamento finalizado em", epochs, "épocas.")

    def test_from_files(self, test_folder):
        """
        Lê arquivos de teste com o formato "digit_nthsample.txt" (por exemplo, "9_65.txt"),
        constrói os dados de entrada e os targets, e realiza o teste no modelo.
        """
        file_list = glob.glob(os.path.join(test_folder, "*.txt"))
        test_inputs = []
        test_targets = []
        
        for filepath in file_list:
            # Exemplo de nome de arquivo: "9_65.txt"
            filename = os.path.basename(filepath)
            parts = filename.split("_")
            if len(parts) != 2:
                print("Formato inesperado no arquivo:", filename)
                continue
            
            try:
                # O primeiro elemento é o dígito (rótulo)
                label = int(parts[0])
            except ValueError:
                print("Erro ao converter o rótulo para int no arquivo:", filename)
                continue

            # Carrega o sample (espera-se que seja uma linha com 256 colunas)
            sample = np.loadtxt(filepath)
            # Se o sample tiver mais de uma dimensão, achata para um vetor
            if sample.ndim > 1:
                sample = sample.flatten()
            if sample.shape[0] != 256:
                print(f"Tamanho inesperado no arquivo {filename}: {sample.shape}")
                continue
            
            test_inputs.append(sample)
            
            # Cria o vetor one-hot para o target (usando 1 para a classe correta e -1 para as demais)
            one_hot = np.full(10, -1, dtype=float)
            one_hot[label] = 1.0
            test_targets.append(one_hot)
        
        # Converte as listas em arrays numpy
        test_inputs = np.array(test_inputs)       # Shape: (n_amostras, 256)
        test_targets = np.array(test_targets)       # Shape: (n_amostras, 10)
        
        # # --- Feedforward no conjunto de teste ---
        # # Aqui usamos a mesma estrutura que você já utiliza
        # net_in = self.vi + np.dot(self.wi, test_inputs.T)
        # z = 1.7159 * np.tanh((2.0/3.0) * net_in)
        # yin = self.vy + np.dot(self.wy, z)
        # y = 1.7159 * np.tanh((2.0/3.0) * yin)

        y = self.forward(test_inputs.T)
        
        # Para classificação, usamos argmax: o índice com valor máximo representa a classe prevista
        predictions = np.argmax(y, axis=0)
        true_labels = np.argmax(test_targets, axis=1)
        accuracy = np.mean(predictions == true_labels) * 100
        
        print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")
        return accuracy
    
    def save_model(self, folder_prefix=None):
        """
        Salva os parâmetros do modelo (pesos e vieses) em uma pasta.
        
        Se folder_prefix não for informado, a pasta será nomeada de acordo com
        o número de neurônios, a taxa de aprendizagem e a função de ativação.
        Os arquivos salvos serão:
        - vnovo.csv: pesos da camada oculta (self.wi)
        - v0novo.csv: bias da camada oculta (self.vi)
        - wnovo.csv: pesos da camada de saída (self.wy)
        - w0novo.csv: bias da camada de saída (self.vy)
        """
        import os
        # Define o nome da pasta com base nos parâmetros, se não for fornecido um prefixo.
        if folder_prefix is None:
            folder_name = f"exp_neurons_{self.number_neurons}_lr_{self.learning_rate:.6f}_act_tanh"
        else:
            folder_name = folder_prefix

        os.makedirs(folder_name, exist_ok=True)

        # Salva os pesos e vieses nos arquivos CSV, utilizando ponto e vírgula como delimitador
        np.savetxt(os.path.join(folder_name, "vnovo.csv"), self.wi, delimiter=";")
        np.savetxt(os.path.join(folder_name, "v0novo.csv"), self.vi, delimiter=";")
        np.savetxt(os.path.join(folder_name, "wnovo.csv"), self.wy, delimiter=";")
        np.savetxt(os.path.join(folder_name, "w0novo.csv"), self.vy, delimiter=";")
        
        print("Modelo salvo na pasta:", folder_name)


    def save_model_porco(self, folder_prefix=None):
        # Define o nome da pasta com base nos parâmetros
        if folder_prefix is None:
            folder_name = f"exp_neurons_{self.number_neurons}_lr_{self.learning_rate:.6f}_act_tanh"
        else:
            folder_name = folder_prefix

        os.makedirs(folder_name, exist_ok=True)

        # Adapta os arrays para os formatos esperados pelo código de teste

        # vanterior deve ter shape (256, 100)
        vanterior = self.wi.T  # Supondo que self.wi já esteja com shape (256, 100)

        # v0anterior deve ter shape (100, 1)
        v0anterior = self.vi.reshape(-1, 1)
        if v0anterior.ndim == 1:
            v0anterior = v0anterior.reshape(-1, 1)

        # wanterior deve ter shape (100, 10)
        wanterior = self.wy.T  # Supondo que self.wy já esteja com shape (100, 10)

        # w0anterior deve ter shape (10, 1)
        w0anterior = self.vy.reshape(-1, 1)
        if w0anterior.ndim == 1:
            w0anterior = w0anterior.reshape(-1, 1)

        # Salva os arquivos CSV com ponto e vírgula como delimitador
        np.savetxt(os.path.join(folder_name, "vnovo.csv"), vanterior, delimiter=";")
        np.savetxt(os.path.join(folder_name, "v0novo.csv"), v0anterior, delimiter=";")
        np.savetxt(os.path.join(folder_name, "wnovo.csv"), wanterior, delimiter=";")
        np.savetxt(os.path.join(folder_name, "w0novo.csv"), w0anterior, delimiter=";")
        
        print("Modelo salvo na pasta:", folder_name)