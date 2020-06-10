import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, data_dim):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(data_dim, 1000)
        self.lin2 = nn.Linear(1000, 1000)
        self.lin3 = nn.Linear(1000, 500)
        self.lin4 = nn.Linear(500, 50)
        self.lin5 = nn.Linear(50, 1)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(50)

    def forward(self, x):
        out = F.relu(self.bn1(self.lin1(x)))
        out = F.relu(self.bn2(self.lin2(out)))
        out = F.relu(self.bn3(self.lin3(out)))
        out = F.relu(self.bn4(self.lin4(out)))
        out = self.lin5(out)
        out = out.view(-1)

        return out
