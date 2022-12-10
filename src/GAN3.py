#!/usr/bin/env python
# coding: utf-8

# # Convolutionnal Generative Adversarial Networks
# ## Initialisation and dataset preparation
# 
# First, let us import the required libraries.

# In[1]:


import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, stft, istft
import matplotlib.pyplot as plt
import librosa
import librosa.display
from misceallaneous import getWavFileAsNpArray, displaySpectrogram
from IPython.display import Audio


# Then, let us include the dataset.
# 
# The dataset is made of two files: `clean/p1.wav`and `white/p1.wav` which are converted into arrays of `int32` and then split into segments of `samples_length`.
# 
# The goal of the CGAN here is to predict the clean sample, when fed with the white one.

# In[2]:


samplerate = 12000
nperseg = 1024

clean = getWavFileAsNpArray("../dataset_2/clean/p1.wav")
white = getWavFileAsNpArray("../dataset_2/white/p1.wav")
clean = np.array(clean, dtype="int32")
white = np.array(white, dtype="int32")

clean_dataset = []
white_dataset = []

samples_length = nperseg

for i in range(0, clean.shape[0]-samples_length, samples_length):
    clean_dataset.append(clean[i:i+samples_length])
    white_dataset.append(white[i:i+samples_length])
clean_dataset = np.array(clean_dataset)
white_dataset = np.array(white_dataset)


# In[3]:


stft_clean_dataset_real = []
stft_clean_dataset_imag = []
stft_white_dataset_real = []
stft_white_dataset_imag = []

for i in clean_dataset:
    c, t, inp = stft(i, fs=samplerate, nperseg=nperseg)
    stft_clean_dataset_real.append(np.real(inp).T)
    stft_clean_dataset_imag.append(np.imag(inp).T)
    
for i in white_dataset:
    c, t, inp = stft(i, fs=samplerate, nperseg=nperseg)
    stft_white_dataset_real.append(np.real(inp).T)
    stft_white_dataset_imag.append(np.imag(inp).T)

stft_clean_dataset_real = np.array(stft_clean_dataset_real)
stft_clean_dataset_imag = np.array(stft_clean_dataset_imag)
stft_white_dataset_real = np.array(stft_white_dataset_real)
stft_white_dataset_imag = np.array(stft_white_dataset_imag)
print(stft_clean_dataset_real.shape, stft_clean_dataset_imag.shape, stft_white_dataset_real.shape, stft_white_dataset_imag.shape)


# In[228]:


def view_output(stft_white_dataset_real, gan, p):
    outputs = []
    for i in range(10):
        y = np.reshape(stft_white_dataset_real[i, :, :], (-1, stft_white_dataset_real.shape[1], stft_white_dataset_real.shape[2]))
        t, y1 = istft(np.reshape(gan.g.predict(y).T, (513, 3))+np.imag(stft_white_dataset_imag[i]).T)
        y2 = np.reshape(y1.T, (clean_dataset.shape[1],))
        outputs.append(y2)
    b = np.concatenate(outputs)
    c, t, bxx = stft(b, fs=samplerate, nperseg=nperseg)
    displaySpectrogram(bxx)
    plt.savefig('stft'+str(p)+'png', format='png')


# # CGAN Model
# The main idea of a GAN model is to create two networks who play an adversarial game:
# - A Generator, whose goal is to produce the most realistic samples possible to fool the Discriminator
# - A Discriminator, whose goal is to correctly guess if its input is a real sample from the clean dataset or an output created by the Generator

# ### Discriminator
# 
# The discriminator here uses a layer to process the Short-Time Fourier Transform (https://en.wikipedia.org/wiki/Short-time_Fourier_transform) before reducing the problem dimension to one single boolean prediction layer.

# In[216]:


def discriminator(input_shape):
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]))
    x2 = tf.keras.layers.Dense(512, activation="tanh")(inputs)
    x3 = tf.keras.layers.Dense(256, activation="tanh")(x2)
    x4 = tf.keras.layers.Dense(128, activation="tanh")(x3)
    x5 = tf.keras.layers.Dense(1, activation="tanh")(x4)
    x6 = tf.keras.layers.Flatten()(x5)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x6)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    model.summary()
    model.compile(optimizer= 'adam', loss='mse', metrics=['accuracy'])
    return model


# ## Generator
# The generator itself is a Convolutionnal Autoencoder.
# 
# Its input size and output size are both the size of the stft array.

# In[217]:


def generator(sizes):
    inputs = tf.keras.Input(shape=(sizes[1], sizes[2]))
    x1 = tf.keras.layers.Dense(10, activation='tanh')(inputs)
    x4 = tf.keras.layers.Dense(sizes[2], activation='tanh')(x1)
    x5 = tf.keras.layers.Add()([inputs, x4])
    outputs = tf.keras.layers.Dense(sizes[2], activation='linear')(x5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


# Take care, the distance between the raw audio might be 'too continuous' to use a classical distance function. Maybe, use the distance function on the STFT's, or another loss function on the raw audio.

# In[218]:


def evaluate_generator(g, inputs, outputs, size=100):
    res = 0
    s = min(size, inputs.shape[0])
    for i in range(s):
        error = (g.predict(np.reshape(inputs[i], (-1, inputs[i].shape[0], inputs[i].shape[1])))-outputs[i])**2
        res += np.sum(error)
    return res/(s*100000000)


# ## Building the GAN

# In[219]:


def get_generator_outputs(white, train_size, g, nperseg, clean):
    steps = train_size//20
    rng = np.random.default_rng()
    g_outputs = []
    batch = rng.choice(white, train_size)
    for i in range(train_size):
        if i%steps == 0:
            print("=", end='')
        t = np.reshape(white[i, :, :], (-1, white.shape[1], white.shape[2]))
        m = g.predict(t)
        g_outputs.append(m)
    print()
    g_outputs = np.reshape(np.array(g_outputs), (train_size,  white.shape[1], white.shape[2]))
    input_data = np.concatenate((g_outputs, clean[:train_size,]))
    output_data = np.concatenate((np.zeros((train_size,)), np.ones((train_size,))))
    return input_data, output_data


# **To read again: are the generator and discriminator reset at each build ?**

# In[230]:


class GAN:
    def __init__(self, size, g, d):
        self.g = g
        self.d = d
        self.size = size
        self.build()
        
    def build(self):
        self.z = self.g.inputs
        self.image = self.g(self.z)
        self.valid = self.d(self.image)
        self.combined_network = tf.keras.Model(self.z, self.valid)
        self.compile()
        
    def block_discriminator(self):
        self.d.trainable = False
        self.g.trainable = True
        self.build()
        
    def block_generator(self):
        self.g.trainable = False
        self.d.trainable = True
        self.build()
        
    def compile(self):
        self.combined_network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

g = generator(stft_white_dataset_real.shape)
d = discriminator(stft_white_dataset_real.shape)


# In[231]:


def train_on_batch(d, i, o, validation_split=0, verbose=True):  
    history = d.fit(i, o, batch_size=16, validation_split=validation_split, verbose=verbose)
    return np.mean(history.history['accuracy'])


# In[232]:


train_size = 500#stft_white_dataset_real.shape[0]


# In[ ]:


gan = GAN(stft_white_dataset_real.shape, g, d)
disc_acc = []
gen_loss = [0]
gan_acc = []
p = 0
for e in range(5):
    g_accuracy = 0
    d_accuracy = 0
    print("Step", e)
    if d_accuracy < 1:
        i, o = get_generator_outputs(stft_white_dataset_real, train_size, gan.g, nperseg, stft_clean_dataset_real)
    gan.block_generator()
    print("Training the discriminator")
    err = evaluate_generator(gan.g, stft_white_dataset_real, stft_clean_dataset_real, 100)
    while d_accuracy < 0.95:
        d_accuracy = train_on_batch(gan.d, i, o, verbose=True)
        disc_acc.append(d_accuracy)
        gan_acc.append(0)
        gen_loss.append(err)
    gan = GAN(stft_white_dataset_real.shape, g, d)
    gan.block_discriminator()
    print("Training the generator")
    while g_accuracy < 0.95:
        view_output(stft_white_dataset_real, gan, p)
        p += 1
        g_accuracy = train_on_batch(gan.combined_network, stft_white_dataset_real[:train_size], np.ones(train_size), validation_split=0.3, verbose=True)
        gan_acc.append(g_accuracy)
        disc_acc.append(0)
        err = evaluate_generator(gan.g, stft_white_dataset_real, stft_clean_dataset_real, 100)
        print(err)
        gen_loss.append(err)
    #print(evaluate_generator(gan.g, white_dataset, clean_dataset))
plt.plot(disc_acc)
plt.plot(gan_acc)
plt.show()
plt.plot(gen_loss[1:])
plt.show()


# In[229]:


view_output(stft_white_dataset_real, gan)


# In[ ]:


Audio(a, rate=samplerate)


# In[ ]:


Audio(b, rate=samplerate)


# In[ ]:




