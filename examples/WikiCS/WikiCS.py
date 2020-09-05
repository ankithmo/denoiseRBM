
import argparse
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import multivariate_normal as mvn

from torch_geometric.datasets import WikiCS as pyg_WikiCS
from torch_geometric.utils import accuracy

from distort_X import distort_x
from distort_A import distort_a


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


class WikiCS_DNN(torch.nn.Module):
    def __init__(self, nn, in_channels, hidden_channels, out_channels, device, mathcal_Z=None, dropout=0.5):
        """
            DNN pipeline for WikiCS

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


def WikiCS_train(model, x, y_true, train_idx, optimizer):
    """
        One step of optimization for WikiCS_DNN

        Args:
            - model: WikiCS_DNN model
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


@torch.no_grad()
def WikiCS_test(model, x, y_true, idx, emb=None, layer=0):
    """
        Inference module for WikiCS_DNN

        Args:
            - model: WikiCS_DNN model
            - x (num_nodes, num_node_features): Input
            - y_true (num_nodes): True node labels
            - idx: [train_idx, val_idx, test_idx]
            - emb (num_nodes, num_node_features): Embeddings of `layer`-th hidden layer
            - layer (int): Hidden layer whose representations is `emb`

        Returns:
            - acc: [train_acc, val_acc, test_acc]
    """
    model.eval()

    _, pred, _ = model(x, emb, layer)
    y_pred = pred.max(1)[1]

    train_idx = idx["train_idx"]
    val_idx = idx["val_idx"]
    test_idx = idx["test_idx"]

    train_acc = accuracy(y_pred[train_idx], y_true[train_idx])
    val_acc = accuracy(y_pred[val_idx], y_true[val_idx])
    test_acc = accuracy(y_pred[test_idx], y_true[test_idx])

    acc = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc
    }

    return acc


def WikiCS(storage, split, MI=False):
    """
        WikiCS citation graph (https://arxiv.org/pdf/2007.02901.pdf) denoising

        Args:
            - storage: Absolute path to store WikiCS dataset
            - split: Which of the 20 splits to use?
            - MI: Estimation of mutual information
                    Default: False

        Returns:
            - x (num_nodes, num_node_features): Node feature matrix
            - y_true (num_nodes): True node label
            - edge_index (2, num_edges): Edge index
            - idx = {
                "train_idx": Boolean list indicating whether or not a node is 
                           part of the training set,
                "val_idx": Boolean list indicating whether or not a node is 
                           part of the validation set,
                "test_idx": Boolean list indicating whether or not a node is 
                           part of the test set
              }
            - nodes = {
                "train_nodes": Integer list of training nodes,
                "val_nodes": Integer list of validation nodes,
                "test_nodes": Integer list of test nodes
              }
            - x_distorted = {
                "X_c": {
                    0: Concatenation of original train node feature matrix, 
                       0% corrupted (original) validation node feature matrix, 
                       0% corrupted (original) test node feature matrix,
                    0+step: Concatenation of original train node feature matrix, 
                       (0+step)% corrupted (original) validation node feature matrix, 
                       (0+step)% corrupted (original) test node feature matrix,
                    .
                    .
                    .
                    100: Concatenation of original train node feature matrix, 
                         100% corrupted validation node feature matrix, 
                         100% corrupted test node feature matrix
                }
                "X_z": {
                    0: Concatenation of original train node feature matrix, 
                       0% blanked out (original) validation node feature matrix, 
                       0% blanked out (original) test node feature matrix,
                    0+step: Concatenation of original train node feature matrix, 
                       (0+step)% blanked out (original) validation node feature matrix, 
                       (0+step)% blanked out (original) test node feature matrix,
                    .
                    .
                    .
                    100: Concatenation of original train node feature matrix, 
                         100% blanked out validation node feature matrix, 
                         100% blanked out test node feature matrix
                }
            }
            - A_distorted = {
                "A_c": {
                    0: Concatenation of edges originating from the train nodes, 
                       0% corrupted (original) edges originating from the validation nodes, 
                       0% corrupted (original) edges originating from the test nodes,
                    0+step: Concatenation of edges originating from the train nodes, 
                       (0+step)% corrupted (original) edges originating from the validation nodes, 
                       (0+step)% corrupted (original) edges originating from the test nodes,
                    .
                    .
                    .
                    100: Concatenation of edges originating from the train nodes, 
                         100% corrupted edges originating from the validation nodes, 
                         100% corrupted edges originating from the test nodes
                }
                "A_z": {
                    0: Concatenation of edges originating from the train nodes, 
                       0% blanked out (original) edges originating from the validation nodes, 
                       0% blanked out (original) edges originating from the test nodes,
                    0+step: Concatenation of edges originating from the train nodes, 
                       (0+step)% blanked out (original) edges originating from the validation nodes, 
                       (0+step)% blanked out (original) edges originating from the test nodes,
                    .
                    .
                    .
                    100: Concatenation of edges originating from the train nodes, 
                         100% blanked out edges originating from the validation nodes, 
                         100% blanked out edges originating from the test nodes
                }
            }
    """
    # Get node feature matrix, labels and edge index
    dataset = pyg_WikiCS(storage)
    data = dataset[0]
    x = data.x
    y_true = data.y
    edge_index = data.edge_index

    # Get indices corresponding to train, validation and test splits
    train_idx = data.train_mask[:, split]
    val_idx = data.val_mask[:, split]
    test_idx = data.test_mask
    # Since WikiCS focuses on semi-supervised node classification, there are a
    # lot of nodes which are not in any of these sets
    # Putting those nodes in limbo into the training set
    all_idx = train_idx + val_idx + test_idx
    train_idx += ~all_idx
    idx = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }

    # Get the nodes corresponding to train, validation and test splits
    nodes = torch.tensor(np.arange(data.num_nodes))
    train_nodes = nodes[train_idx]
    val_nodes = nodes[val_idx]
    test_nodes = nodes[test_idx]
    nodes = {
        "train_nodes": train_nodes,
        "val_nodes": val_nodes,
        "test_nodes": test_nodes
    }

    # Distort the node feature matrix
    x_path = osp.join(storage, "WikiCS_x_distorted.pt")
    if not osp.exists(x_path):
        x_distorted = distort_x(x, idx)
        torch.save(x_distorted, x_path)
    else:
        x_distorted = torch.load(x_path)

    # Distort the edge index
    A_path = osp.join(storage, "WikiCS_A_distorted.pt")
    if not osp.exists(A_path):
        A_distorted = distort_a(edge_index, nodes, idx, "WikiCS")
        torch.save(A_distorted, A_path)
    else:
        A_distorted = torch.load(A_path)

    return x, y_true, edge_index, idx, nodes, x_distorted, A_distorted


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="WikiCS")
    parser.add_argument("--storage", default=osp.join(osp.dirname(osp.realpath(__file__)), ".", "data"),
                        help="Absolute path to store WikiCS dataset")
    parser.add_argument("--split", default=0, help="Which of the 20 splits to use?")
    args = parser.parse_args()
    
    WikiCS(args.storage, args.split)