from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(4*4*64, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
