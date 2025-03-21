import numpy as np
import os

# Obtém o diretório do arquivo atual
current_dir = os.path.dirname(os.path.abspath(__file__))

targets_filename = "targets10.csv"

inputs_filename = "digitostreinamento900.txt"

# Constrói o caminho absoluto para o arquivo de dados
training_file_path = os.path.join(current_dir, f"../../data/{inputs_filename}")

# Carrega o arquivo
digit_9 = np.loadtxt(training_file_path)
print(digit_9.shape)