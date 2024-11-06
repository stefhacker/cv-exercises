import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        # START TODO #################
        # see model description in exercise pdf
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.functional.relu()
        self.pool1 = nn.functional.max_pool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(8, 120)
        self.relu3 = nn.functional.relu()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.functional.relu()
        self.fc3 = nn.Linear(84, 10)


        # END TODO #################

    def forward(self, x):
        # START TODO #################
        # see model description in exercise pdf
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        #vectorize, x.size(0) is batch size and -1 flattens images (f. e. 3 * 28 * 28 = 2352)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)

        return x



        # END TODO #################
