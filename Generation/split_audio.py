# Splits the audio passed as argument into fragments of 2s, and store them in the same format, in the directory passed as argument.
import wave
from sys import argv
from os.path import splitext

def getAudio(filename):
    return wave.open(filename, mode='rb')

def splitAudio(filename):
    source = getAudio(filename)
    index = 0
    frame = 0
    framerate = source.getframerate()
    nchannels = source.getnchannels()
    sampwidth = source.getsampwidth()
    while frame < source.getnframes():
        newFile = wave.open(splitext(filename)[0]+"_"+str(index)+splitext(filename)[1], mode='wb')
        newFile.setnchannels(nchannels)
        newFile.setsampwidth(sampwidth)
        newFile.setframerate(framerate)
        window = source.readframes(framerate*2)
        newFile.writeframes(window)
        index += 1
        frame += framerate*2
        newFile.close()
    source.close()

if __name__ == "__main__":
    splitAudio(argv[1])
