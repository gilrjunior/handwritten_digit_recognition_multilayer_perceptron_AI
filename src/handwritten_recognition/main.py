from handwritten_recognition.Mlp import *
from handwritten_recognition.data_processing import *
import os

def main():
    
    mlp =  Mlp(100, 0.01)

    # mlp.optimized_train(0.01)
    # mlp.batch_train(0.01, 32)
    mlp.batch_train_adaptive_lr_early_stopping(
        min_error = 0.15, 
        batch_size=32, 
        alpha=0.05, 
        theta=0.05, 
        d = 0.95, 
        u = 1.01, 
        patience=500
    )

    test_folder_path = "/data"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder_path = os.path.join(current_dir, f"../../{test_folder_path}")
    accuracy = mlp.test_from_files(test_folder=test_folder_path)


    if accuracy >= 70:
        ...
        # mlp.save_model()
        # mlp.save_model_porco()

if __name__ == "__main__":
    main()