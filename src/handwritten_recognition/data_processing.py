import numpy as np
import pandas as pd
import os

 # Obtém o diretório do arquivo atual
current_dir = os.path.dirname(os.path.abspath(__file__))

inputs_filename = "digitostreinamento900.txt"
targets_filename = "targets10.csv"

# Constrói o caminho absoluto para o arquivo de dados
training_file_path = os.path.join(current_dir, f"../../data/{inputs_filename}")
targets_file_path = os.path.join(current_dir, f"../../data/{targets_filename}")

def load_inputs(): 

    # Carrega o arquivo
    digit_9 = np.loadtxt(training_file_path)

    print(digit_9.shape)

def load_targets(): 
    
    # Carrega o arquivo
    targets = pd.read_csv(targets_file_path, delimiter=";", header=None).to_numpy()

    print(targets)
    print(targets.shape)