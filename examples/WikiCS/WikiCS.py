
import argparse

import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import multivariate_normal as mvn

from torch_geometric.datasets import WikiCS as pyg_WikiCS
from torch_geometric.utils import accuracy

import sys
sys.path.insert(0, osp.join("..", "denoiseRBM"))
from denoiseRBM.distort_X import distort_x
from denoiseRBM.distort_A import distort_a


@torch.no_grad()
def test(model, x, y_true, idx, emb=None, layer=0):
    """
        Inference module for DNN

        Args:
            - model: DNN model
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


def WikiCS(storage, split):
    """
        WikiCS citation graph (https://arxiv.org/pdf/2007.02901.pdf) denoising

        Args:
            - storage: Absolute path to store WikiCS dataset
            - split: Which of the 20 splits to use?

        Returns:
            - x (num_nodes, num_node_features): Node feature matrix
            - y_true (num_nodes): True node label
            - C (int): Number of classes
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
    # Ensure that storage directory exists
    assert osp.exists(storage), ValueError(f"{storage} directory does not exist.")

    # Ensure split between 0 and 20
    assert 0 <= split <= 20, ValueError(f"Split expected to be in [0,20], got {split} instead.")

    # Get node feature matrix, labels and edge index
    dataset = pyg_WikiCS(storage)
    data = dataset[0]
    x = data.x
    y_true = data.y
    edge_index = data.edge_index
    C = dataset.num_classes

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

    return x, y_true, C, edge_index, idx, nodes, x_distorted, A_distorted


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="WikiCS")
    parser.add_argument("--storage", default=osp.join(osp.dirname(osp.realpath(__file__)), ".", "data"),
                        help="Absolute path to store WikiCS dataset")
    parser.add_argument("--split", default=0, help="Which of the 20 splits to use?")
    args = parser.parse_args()
    
    # Create storage directory if it doesn't exist
    if not osp.exists(args.storage):
        print(f"{args.storage} does not exist. Creating {args.storage}...", end="")
        os.mkdir(args.storage)
    print("Done!")

    WikiCS(args.storage, args.split)