# Loads a WAV to tensor dataset from directory passed as argument
import tensorflow
from sys import argv

def load_dataset(path):
    return tensorflow.data.experimental.load("mini_dataset/tensors/mini_dataset", tensorflow.TensorSpec(shape=([88200, 2]), dtype=tensorflow.float32))

if __name__ == "__main__":
    dataset = load_dataset(argv[1])
