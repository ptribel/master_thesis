import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, stft, istft
import matplotlib.pyplot as plt
import librosa
import librosa.display

def getWavFileAsNpArray(filename):
    return np.array(wavfile.read(filename)[1])

def displaySpectrogram(stft_array, sr=12000, axe=0):
    #S_db = librosa.amplitude_to_db(np.abs(stft_array), ref=np.max)
    if axe == 0:
        axe = plt.subplot(1, 1, 1)
    librosa.display.specshow(stft_array, sr=sr, ax=axe)
