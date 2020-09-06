
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import multivariate_normal as mvn


def Z(d, sigma):
    """
        Returns a d-dimensional vector drawn from \mathcal{N}(0,\sigma^{2}I_{d})

        Args:
            - d (int): Dimensionality
            - sigma (float): Standard deviation

        Returns:
            d-dimensional vector
    """
    return mvn.MultivariateNormal(torch.zeros(d), torch.diag(torch.ones(d) * sigma**2))


class DNN(torch.nn.Module):
    def __init__(self, nn, in_channels, hidden_channels, out_channels, device, mathcal_Z=None, dropout=0.5):
        """
            DNN pipeline

            Args:
                - nn: Neural network layer
                - in_channels (int): Number of input units
                - hidden_channels (int): Number of units in each hidden layer
                - out_channels (int): Number of output units
                - device: torch device
                - mathcal_Z: AWGN
                - dropout (float): Dropout probability

            Returns:
                - z: List of representations learnt by each of the NN layers
                - hat_Y: Predicted node labels
                - s: List of representations learnt by each of the non-linearities
        """
        super(DNN, self).__init__()

        self.NN_1 = nn(in_channels, hidden_channels)
        self.NN_2 = nn(hidden_channels, hidden_channels)
        self.NN_3 = nn(hidden_channels, out_channels)

        self.BN_1 = torch.nn.BatchNorm1d(hidden_channels)
        self.BN_2 = torch.nn.BatchNorm1d(hidden_channels)

        self.device = device
        self.mathcal_Z = mathcal_Z
        self.dropout = dropout

    def reset_parameters(self):
        self.NN_1.reset_parameters()
        self.NN_2.reset_parameters()
        self.NN_3.reset_parameters()

        self.BN_1.reset_parameters()
        self.BN_2.reset_parameters()

    def forward(self, x, emb=None, layer=0):
        z, s = [], []

        z_0 = x
        
        # f_1
        z_1 = emb if layer == 1 else self.NN_1(z_0)
        z.append(z_1.detach())
        z_1 = self.BN_1(z_1)
        S_1 = F.relu(z_1)
        s.append(S_1.detach())

        if self.mathcal_Z is None: # dropout
            T_1 = F.dropout(S_1, p=self.dropout, training=self.training)
        else: # AWGN channel
            n_1 = torch.zeros(S_1.size(0))
            mathcal_Z_1 = self.mathcal_Z.sample(n_1.size()).to(self.device)
            T_1 = S_1 + mathcal_Z_1
        
        # f_2
        z_2 = emb if layer == 2 else self.NN_2(T_1)
        z.append(z_2.detach())
        z_2 = self.BN_2(z_2)
        S_2 = F.relu(z_2)
        s.append(S_2.detach())
        
        if self.mathcal_Z is None: # dropout
            T_2 = F.dropout(S_2, p=self.dropout, training=self.training)
        else: # AWGN channel
            n_2 = torch.zeros(S_2.size(0))
            mathcal_Z_2 = self.mathcal_Z.sample(n_2.size()).to(self.device)
            T_2 = S_2 + mathcal_Z_2

        # f_3
        z_3 = emb if layer == 3 else self.NN_3(T_2)
        z.append(z_3.detach())
        hat_Y = torch.log_softmax(z_3, dim=-1)

        return z, hat_Y, s


def train(model, x, y_true, train_idx, optimizer):
    """
        One step of optimization for DNN

        Args:
            - model: DNN model
            - x (num_nodes, num_node_features): Input
            - y_true (num_nodes): True node labels
            - train_idx: Training node index
            - optimizer: Optimizer

        Returns:
            - Training loss
    """
    model.train()
    
    optimizer.zero_grad()
    _, out, _ = model(x[train_idx])
    loss = F.nll_loss(out, y_true[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()