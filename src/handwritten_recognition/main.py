from handwritten_recognition.Mlp import *
from handwritten_recognition.data_processing import *

def main():
    
    mlp =  Mlp(150, 0.1)

    # mlp.optimized_train(0.01)
    # mlp.batch_train(0.01, 32)

if __name__ == "__main__":
    main()