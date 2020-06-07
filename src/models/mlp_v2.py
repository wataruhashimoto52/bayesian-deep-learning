import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal, Delta
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.predictive import Predictive



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


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim, w_sigma, y_sigma):
        super(BayesianNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.w_sigma = w_sigma
        self.y_sigma = y_sigma
        """
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(50)
        """
        
        
        
    def model(self, x, y):
        with pyro.plate("w1_plate_dim2", 1000):
            with pyro.plate("w1_plate_dim1", self.input_dim):
                w1 = pyro.sample("w1", Normal(0, self.w_sigma))
                
        with pyro.plate("w2_plate_dim2", 1000):
            with pyro.plate("w2_plate_dim1", 1000):
                w2 = pyro.sample("w2", Normal(0, self.w_sigma))

        with pyro.plate("w3_plate_dim2", 500):
            with pyro.plate("w3_plate_dim1", 1000):
                w3 = pyro.sample("w3", Normal(0, self.w_sigma))
                
        with pyro.plate("w4_plate_dim2", 50):
            with pyro.plate("w4_plate_dim1", 500):
                w4 = pyro.sample("w4", Normal(0, self.w_sigma))
                
        with pyro.plate("w5_plate_dim2", 1):
            with pyro.plate("w5_plate_dim1", 50):
                w5 = pyro.sample("w5", Normal(0, self.w_sigma))
                
                
        def forward(x):
            out = F.relu(torch.bmm(x, w1.squeeze()))
            out = F.relu(torch.bmm(out, w2.squeeze()))
            out = F.relu(torch.bmm(out, w3.squeeze()))
            out = F.relu(torch.bmm(out, w4.squeeze()))
            out = torch.bmm(out, w5).squeeze()
            
            return out
        
        with pyro.plate("map", x.size()[1]):
            prediction_mean = forward(x)
            pyro.sample("obs", Normal(prediction_mean, self.y_sigma), obs=y)
            
            return prediction_mean
    
    def predict(self, x_pred):
        def wrapped_model(x_data, y_data):
            
            pyro.sample("prediction", Delta(self.model(x_data, y_data)))

        predictive = Predictive(wrapped_model, self.posterior_samples)
        samples =  predictive.get_samples(x_pred, None)
        return samples["prediction"], samples["obs"]
    