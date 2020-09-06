
import argparse

import os
import os.path as osp

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import sys
sys.path.insert(0, osp.join("..", "denoiseRBM"))
from denoiseRBM.distort_X import distort_x
from denoiseRBM.distort_A import distort_a


def ogbn_arxiv(storage):
    """
        ogbn-arxiv citation graph (https://arxiv.org/pdf/2005.00687.pdf) denoising

        Args:
            - storage: Absolute path to store ogbn-arxiv dataset

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
    # Ensure that storage directory exists
    assert osp.exists(storage), ValueError(f"{storage} directory does not exist.")

    # Get node feature matrix, labels and edge index
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    x = data.x
    y_true = data.y
    edge_index = data.edge_index

    # Get indices corresponding to train, validation and test splits
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']
    idx = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }

    # Get the nodes corresponding to train, validation and test splits
    nodes = torch.tensor(range(data.num_nodes))
    train_nodes = nodes[train_idx]
    val_nodes = nodes[val_idx]
    test_nodes = nodes[test_idx]
    nodes = {
        "train_nodes": train_nodes,
        "val_nodes": val_nodes,
        "test_nodes": test_nodes
    }

    # Distort the node feature matrix
    x_path = osp.join(storage, "ogbn-arxiv_x_distorted.pt")
    if not osp.exists(x_path):
        x_distorted = distort_x(x, idx)
        torch.save(x_distorted, x_path)
    else:
        x_distorted = torch.load(x_path)

    # Distort the edge index
    A_path = osp.join(storage, "ogbn-arxiv_A_distorted.pt")
    if not osp.exists(A_path):
        A_distorted = distort_a(edge_index, nodes, idx, "ogbn-arxiv")
        torch.save(A_distorted, A_path)
    else:
        A_distorted = torch.load(A_path)

    return x, y_true, edge_index, idx, nodes, x_distorted, A_distorted


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="ogbn-arxiv")
    parser.add_argument("--storage", default=osp.join(osp.dirname(osp.realpath(__file__)), ".", "data"),
                        help="Absolute path to store ogbn-arxiv dataset")
    args = parser.parse_args()
    
    # Create storage directory if it doesn't exist
    if not osp.exists(args.storage):
        print(f"{args.storage} does not exist. Creating {args.storage}...", end="")
        os.mkdir(args.storage)
    print("Done!")

    ogbn_arxiv(args.storage)