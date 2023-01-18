import torch
import torch.nn as nn
import torch.nn.functional as F

class george_SAE(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE, self).__init__()
        # encoder
        self.en1 = nn.Linear(n_features, 200,dtype=torch.float64)
        self.en2 = nn.Linear(200, 100,dtype=torch.float64)
        self.en3 = nn.Linear(100, 50,dtype=torch.float64)
        self.en4 = nn.Linear(50, z_dim,dtype=torch.float64)
        # decoder
        self.de1 = nn.Linear(z_dim, 50,dtype=torch.float64)
        self.de2 = nn.Linear(50, 100,dtype=torch.float64)
        self.de3 = nn.Linear(100, 200,dtype=torch.float64)
        self.de4 = nn.Linear(200, n_features,dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def loss(self, model_children, true_data, reconstructed_data, reg_param):
        mse = nn.MSELoss()
        mse_loss = mse(reconstructed_data, true_data)
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))
        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss

class george_SAE_BN(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_BN, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
                nn.Linear(n_features, 200,dtype=torch.float64),        
                nn.Dropout(p=0.5),
                nn.BatchNorm1d(200,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(200, 100,dtype=torch.float64),        
                nn.Dropout(p=0.4),
                nn.BatchNorm1d(100,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(100,50,dtype=torch.float64),
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(50,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(50, z_dim,dtype=torch.float64),        
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(z_dim,dtype=torch.float64),
                nn.LeakyReLU()                
                )


        # decoder
        self.dec_nn = nn.Sequential(
                nn.Linear(z_dim, 50,dtype=torch.float64),        
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(50,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(50, 100,dtype=torch.float64),        
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(100,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(100,200,dtype=torch.float64),
                nn.Dropout(p=0.4),
                nn.BatchNorm1d(200,dtype=torch.float64),
                nn.LeakyReLU(),
                nn.Linear(200, n_features,dtype=torch.float64),        
                nn.Dropout(p=0.5),
                nn.BatchNorm1d(n_features,dtype=torch.float64),
                nn.ReLU()                
                )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def loss(self, model_children, true_data, reconstructed_data, reg_param):
        mse = nn.MSELoss()
        mse_loss = mse(reconstructed_data, true_data)
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))
           # print('l1',(l1_loss))
        # print('mse',(mse_loss))
        loss = mse_loss + reg_param * l1_loss
        return loss, mse_loss, l1_loss