import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchaudio import load
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC

#from data_dir.directivityDataset import DirectivityDataset, DirectivityDataset2Targets
from Directivity.directivityDataset import DirectivityDataset
from CNNs import CNN, CNN_2channel, CNN_horizontal
from train import train_model

parser = argparse.ArgumentParser(description="Script for Directivity Estimation")

parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("--train_size", type=float, default=0.8)
parser.add_argument("-f", "--feature", type=str, default="mel")
parser.add_argument("-m", "--model", type=str, default="cnn")
parser.add_argument("-p", "--pattience", type=int, default=5)
parser.add_argument("-d","--destination_path", type=str, default="./Weights/")
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

SEED = args.seed
FEATURE = args.feature
MODEL = args.model
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
TRAIN_SIZE = args.train_size
VALIDATION_SIZE = 1 - TRAIN_SIZE
PATTIENCE = args.pattience

DATA_DIR = os.getcwd() + "/Directivity/"
DATA_ANNOTATION = os.path.join(DATA_DIR, "MatrixLabelsDirectivityCategoriesBalanced.csv")
DESTINATION_PATH = args.destination_path
TARGET_SAMPLE_RATE = 44100
NUM_SAMPLES = 200000
DEVICE = 'cuda'

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

logging.basicConfig(format='%(asctime)s %(message)s', filename= DESTINATION_PATH + MODEL +"-"+ FEATURE +'-log.log', encoding='utf-8', level=logging.DEBUG, filemode='w')
logger = logging.getLogger(__name__)
logger.info("Preparacion de entrenamiento")

if FEATURE == "mel":
    feature = MelSpectrogram(
            sample_rate = TARGET_SAMPLE_RATE,
            n_fft = 1024,
            hop_length = 512,
            n_mels = 64
        ).to(DEVICE)
elif FEATURE == "spectogram":
    feature = Spectrogram(n_fft = 512).to(DEVICE)
elif FEATURE == "mfcc":
    feature = MFCC(
            sample_rate = TARGET_SAMPLE_RATE,
            n_mfcc = 128,
            melkwargs = {"n_fft": 2048, "n_mels": 128, "hop_length": 512, "mel_scale": "htk"}
            ).to(DEVICE)
elif FEATURE == "spectogram_octave":
    feature = "spec_poctave"

dir = DirectivityDataset(DATA_ANNOTATION, DATA_DIR, feature, TARGET_SAMPLE_RATE, NUM_SAMPLES, DEVICE)

train_len = int(0.8*len(dir))
val_len = int(0.2*len(dir))

train_dataset = Subset(dir, range(train_len))
val_dataset = Subset(dir, range(train_len, train_len+val_len))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

logger.info("Dataset Cargado correctamente")

signal, _ = train_dataset[0]
input_dims = signal[0].shape

if MODEL == "cnn":
    model = CNN_2channel(W = input_dims[0], H=input_dims[1]).to(DEVICE)
elif MODEL == "cnn-diff":
    model = CNN(W = input_dims[0], H=input_dims[1]).to(DEVICE)
elif MODEL == "cnn-rectangular":
    model = CNN_horizontal(input_dims=input_dims, octaves=9, bins_octave=12).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

logger.info("Modelo en memoria")
logger.info("Entrenamiento Iniciado")
model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, 
        EPOCHS, 
        loss_fn, 
        opt, 
        train_loader, 
        val_loader,
        DESTINATION_PATH + MODEL +"-"+ FEATURE + "model.pth",
        pattience = PATTIENCE)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.tight_layout()
ax[0].plot(train_loss, label="train_loss")
ax[0].plot(val_loss, label="validation_loss")
ax[0].set_title("Train Loss")
ax[0].set_ylabel("Loss Function")
ax[0].set_xlabel("Epochs")
ax[0].legend()

ax[1].plot(train_acc, label="train_acc")
ax[1].plot(val_acc, label="validation_acc")
ax[1].set_title("Train Accuracy")
ax[1].set_ylabel("Loss Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].legend()
plt.savefig(DESTINATION_PATH + MODEL +"_"+ FEATURE + "train.png")

df = pd.DataFrame({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
df.to_csv(DESTINATION_PATH + MODEL +"_"+ FEATURE + "train_data.csv", index=False)

logger.info(f"Modelo Entrenado con exito en {DESTINATION_PATH}")
logger.info(f"Train Loss: {np.min(train_loss)}")
logger.info(f"Train max Accuracy: {np.max(train_acc)}")
logger.info(f"Validation Loss: {np.min(val_loss)}")
logger.info(f"Train max Accuracy: {np.max(val_acc)}")
