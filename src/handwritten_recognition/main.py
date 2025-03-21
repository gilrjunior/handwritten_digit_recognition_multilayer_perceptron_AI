from Mlp import *

def main():
    
    mlp = Mlp(200, 50, -1, 1, 0.01)

    mlp.optimized_train(0.01)

if __name__ == "__main__":
    main()