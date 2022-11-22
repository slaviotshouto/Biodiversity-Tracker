# %%
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import tensorflow as tf
import tensorflow_io as tfio

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

import pandas as pd
from pathlib import Path


# %%
import os
import pathlib
import pandas as pd

download_path = Path.cwd()/'ukxcmany'
metadata_file = download_path/'xcmeta.csv'
df = pd.read_csv(metadata_file, delimiter = "\t")
df.head()

df_data = pd.DataFrame({'Species' : [], 'Path' : []})

spec_list = [str(df['gen'][i]) + " " + str(df['sp'][i]) for i in range(len(df[['id']]))]
df_data['Species'] = spec_list

file_list = []
for i in range( len(df['id']) ):
    path = str(download_path) + '\\flac' + '\\xc' + str(df['id'][i]) + '.flac'
    file_list.append(path)
    
df_data['Path'] = file_list

df_data = df_data.sample(frac=1) # shuffle the rows arround 
df_data.head()

# %%
names_set = {0}

y = df_data['Species']

for i in range(len(df_data['Species'])):
    name = df_data['Species'][i]
    names_set.add(name)
names_set.remove(0)

print("There are", len(names_set), "species in this dataset.")
print(len(y))

# %%
def open_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

# def rechannel(aud, new_channel):
#     sig, sr = aud

#     if (sig.shape[0] == new_channel):
#       # Nothing to do
#       return aud

#     if (new_channel == 1):
#       # Convert from stereo to mono by selecting only the first channel
#       resig = sig[:1, :]
#     else:
#       # Convert from mono to stereo by duplicating the first channel
#       resig = torch.cat([sig, sig])

#     return ((resig, sr))

def pad_trunc(audio_file, max_ms):
    sig, sr = audio_file
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
        sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)

def spectrogram(audio_file, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = audio_file
    top_db = 80
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)


# %%
from torch.utils.data import DataLoader, Dataset, random_split

class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        #self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        audio_file = df_data['Path'][idx]
        aud = open_file(audio_file)
        dur_aud = pad_trunc(aud, self.duration)
        sgram = spectrogram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        
        return sgram, df_data['Species'][idx]
        

# %%
from torch.utils.data import random_split

dataset = SoundDS(df_data)

num_items = len(dataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_data, val_data = random_split(dataset, [num_train, num_val])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

# %%
print(len(train_loader), len(val_loader))

# %%
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()


# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        try:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        except:
            continue

print('Finished Training')


