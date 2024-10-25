import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import pad
from torchaudio import load
from torchaudio.transforms import Resample, MelSpectrogram

DATA_DIR = os.getcwd()+'/Dataset/Datos/'
DOA_ANNOTATION = os.path.join(DATA_DIR, "DoA_Labels.csv")
DISTANCE_ANNOTATION = os.path.join(DATA_DIR, "Distance_Labels.csv")
DIRECTIVITY_ANNOTATION = os.path.join(DATA_DIR, "Directivity_Labels.csv")

class Sound_Dataset(Dataset):
    def __init__(
            self,
            annotations_file,
            audio_dir,
            transform,
            target_sr,
            num_samples,
            device
            ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self):
        pass

    def _get_audio_sample_path(self, index):
        return os.path.join(DATA_DIR, self.annotations.iloc[index, 0])
    
    def _get_audio_sample_label(self, index, column):
        return self.annotations.iloc[index, column]
    
    def _resample(self, signal, sr):
        if sr != self.target_sr:
            resampler = Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal
    
    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad(self, signal):
        if signal.shape[1] < self.num_samples:
            num_samples = self.num_samples - signal.shape[1]
            signal = pad(signal, (0, num_samples))
        return signal   

class DoA_Dataset(Sound_Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_sr, num_samples, device):
        super().__init__(annotations_file, audio_dir, transform, target_sr, num_samples, device)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        target = self._get_audio_sample_label(index, 5)
        target = self._fix_angle(target)
        target = torch.tensor(target,dtype=torch.float32)
        target = target.to(self.device)

        signal, sr = load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal,sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        signal_ch1, signal_ch2 = signal
        signal_ch1, signal_ch2 = self.transform(signal_ch1), self.transform(signal_ch2)

        signal = torch.stack((signal_ch1, signal_ch2), dim=0)

        return signal, target
    
    def _fix_angle(self, target):
        if target > 180:
            target-=360
        
        target /= 180

        return target
    
class DirectivityDataset(DoA_Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_sr, num_samples, device):
        super().__init__(annotations_file, audio_dir, transform, target_sr, num_samples, device)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        target = self._get_audio_sample_label(index, 7)
        target = self._fix_angle(target)
        target = torch.tensor(target,dtype=torch.float32)
        target = target.to(self.device)

        doa = self._get_audio_sample_label(index, 5)
        doa = self._fix_angle(doa)
        doa = torch.tensor(doa, dtype=torch.float32)
        doa = doa.to(self.device)

        signal, sr = load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal,sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        signal_ch1, signal_ch2 = signal
        signal_ch1, signal_ch2 = self.transform(signal_ch1), self.transform(signal_ch2)

        signal = torch.stack((signal_ch1, signal_ch2), dim=0)

        return signal, doa, target
    
class DistanceDataset(DoA_Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_sr, num_samples, device):
        super().__init__(annotations_file, audio_dir, transform, target_sr, num_samples, device)
        self._targets = self.annotations.iloc[:, 7]

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        distance = self._get_audio_sample_label(index, 7)
        distance = self._scale_distance(distance)
        distance = torch.tensor(distance,dtype=torch.float32)
        distance = distance.to(self.device)

        doa = self._get_audio_sample_label(index, 5)
        doa = self._fix_angle(doa)
        doa = torch.tensor(doa,dtype=torch.float32)
        doa = doa.to(self.device)

        signal, sr = load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal,sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        signal_ch1, signal_ch2 = signal
        signal_ch1, signal_ch2 = self.transform(signal_ch1), self.transform(signal_ch2)

        signal = torch.stack((signal_ch1, signal_ch2), dim=0)

        return signal, doa, distance
    
    def _scale_distance(self, distance):
        distances = self.annotations["Distance"]
        return (distance - distances.min())/(distances.max() - distances.min())

class CartesianDataset(Sound_Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_sr, num_samples, device):
        super().__init__(annotations_file, audio_dir, transform, target_sr, num_samples, device)
        self._targets = self.annotations.iloc[:, 7]

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        
        x = torch.tensor(self._get_audio_sample_label(index, 2), dtype=torch.float32)
        #y = torch.tensor(self._get_audio_sample_label(index, 3), dtype=torch.float32)
        z = torch.tensor(self._get_audio_sample_label(index, 4), dtype=torch.float32)
        
        x = x.to(self.device)
        z = z.to(self.device)

        signal, sr = load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal,sr)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        signal_ch1, signal_ch2 = signal
        signal_ch1, signal_ch2 = self.transform(signal_ch1), self.transform(signal_ch2)

        signal = torch.stack((signal_ch1, signal_ch2), dim=0)

        return signal, x, z
    
if __name__ == "__main__":

    TARGET_SAMPLE_RATE = 16000
    NUM_SAMPLES = 16000

    mel_spec = MelSpectrogram(
        sample_rate = 16000,
        n_fft = 1024,
        hop_length=512,
        n_mels=64
    )

    doa = DoA_Dataset(DOA_ANNOTATION, DATA_DIR, mel_spec, TARGET_SAMPLE_RATE, NUM_SAMPLES, 'cpu')
    distance = DistanceDataset(DISTANCE_ANNOTATION, DATA_DIR, mel_spec, TARGET_SAMPLE_RATE, NUM_SAMPLES, 'cpu')

    signal, doa, dis = distance[0]
    print(doa, dis)
