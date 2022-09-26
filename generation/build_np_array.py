# Builds a dataset from WAV files passed as argument
# Syntax: python3 build_dataset dest name $(ls path/*.wav)
from wav_tensor_conversion import encode_wav_to_tensor
from scipy.io import wavfile
import tensorflow.data
from sys import argv
import numpy as np

if __name__ == "__main__":
    waves = []
    a = 0
    for file in argv[3:]:
        if a < 100:
            waves.append(wavfile.read(file)[1])
        else:
            break
        a += 1
    data = np.concatenate(waves)
    f = open(argv[1], 'w')
    f.write("import numpy as np\n")
    p = ','.join(str(x) for x in data)
    f.write(argv[2] + " = np.array([" + p + "])")
    f.close()
