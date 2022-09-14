# Transforms the WAV files passed as argument to tensors, or the inverse, depending on the requested argument
from tensorflow.audio import decode_wav
from tensorflow.io import read_file
from sys import argv


def encode_wav_to_tensor(filename):
    """
    Returns two tensors:
        - one containing the audio (in float32)
        - one containing the sample rate (in int32)
    """
    source = read_file(filename)
    audio, sr = decode_wav(contents=source)
    return audio, sr

if __name__ == "__main__":
    audio, sr = encode_wav_to_tensor(argv[1])
