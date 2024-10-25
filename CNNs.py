import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, W, H):
        super().__init__()
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=self._calculate_output_size(W, H), out_features = 64),
            nn.ReLU(),
            )
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=64, out_features = 32),
            nn.ReLU(),
            )
        self.dense3 = nn.Linear(in_features=32, out_features=8)

    def _calculate_output_size(self, W, H): #Ws = (We - k_size + 2*pad)/stride #pad=0, stride=1
        # Conv1
        W = (W - 3 + 2*0)//1 + 1
        H = (H - 3 + 2*0)//1 + 1
        W, H = W // 2, H // 2

        # Conv2
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        # Conv3
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2
        
        # Conv4
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        return int(W * H * 512)

    def forward(self, x:torch.Tensor):
        x1 = x[:,0:1,:,:]
        x2 = x[:,1:2,:,:]

        x1 = self.conv11(x1)
        x2 = self.conv12(x2)  

        x = x1 - x2

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

class CNN_2channel(nn.Module):
    def __init__(self, W, H):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=self._calculate_output_size(W, H), out_features = 64),
            nn.ReLU(),
            )
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=64, out_features = 32),
            nn.ReLU(),
            )
        self.dense3 = nn.Linear(in_features=32, out_features=8)

    def _calculate_output_size(self, W, H): #Ws = (We - k_size + 2*pad)/stride #pad=0, stride=1
        # Conv1
        W = (W - 3 + 2*0)//1 + 1
        H = (H - 3 + 2*0)//1 + 1
        W, H = W // 2, H // 2

        # Conv2
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        # Conv3
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2
        
        # Conv4
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        return int(W * H * 512)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

class CNN_horizontal(nn.Module):
    def __init__(self, input_dims, octaves=9, bins_octave=12):
        super().__init__()
        self.input_dims = input_dims
        self.bins_octave = bins_octave
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(int(self.bins_octave/2), 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(int(4/2), 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(int(4/2), 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(int(4/2), 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=self._calculate_output_size(self.input_dims[0], self.input_dims[1]), out_features = 64),
            nn.ReLU(),
            )
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=64, out_features = 32),
            nn.ReLU(),
            )
        self.dense3 = nn.Linear(in_features=32, out_features=8)

    def _calculate_output_size(self, W, H): #Ws = (We - k_size + 2*pad)/stride #pad=0, stride=1
        # Conv1

        W = (W - int(self.bins_octave/2) + 2*0)//1 + 1
        H = (H - 2 + 2*0)//1 + 1
        W, H = W // 2, H // 2

        # Conv2
        W = W - 2 + 1
        H = H - 2 + 1
        W, H = W // 2, H // 2

        # Conv3
        W = W - 2 + 1
        H = H - 2 + 1
        W, H = W // 2, H // 2
        
        # Conv4
        W = W - 2 + 1
        H = H - 2 + 1
        W, H = W // 2, H // 2

        print(int(W * H * 512))
        
        return int(W * H * 512)

    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
