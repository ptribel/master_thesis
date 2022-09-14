# Builds a dataset from WAV files passed as argument
from wav_tensor_conversion import encode_wav_to_tensor
import tensorflow.data
from sys import argv

if __name__ == "__main__":
    tensors = []
    for file in argv[2:]:
        print(file)
        tensors.append(encode_wav_to_tensor(file)[0])
    dataset = tensorflow.data.Dataset.from_tensor_slices(tensors)
    tensorflow.data.experimental.save(dataset, argv[1])
