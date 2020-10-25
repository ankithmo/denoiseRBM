
import argparse

import os
import os.path as osp
import sys
here = osp.abspath(osp.dirname(__file__)) # ogbn-arxiv
parent = osp.abspath(osp.dirname(here)) # examples
grandparent = osp.abspath(osp.dirname(parent)) # denoiseRBM
sys.path.insert(0, grandparent)
sys.path.insert(1, parent)
sys.path.insert(2, here)

import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from denoiseRBM.distort_X import distort_x
from denoiseRBM.distort_A import distort_a


@torch.no_grad()
def test(model, x, y_true, idx, evaluator, emb=None, layer=0):
    """
        Testing module for DNN trained on ogbn-arxiv

            Arguments:
                model : DNN
                    Model trained on ogbn-arxiv
                x : torch.tensor of shape (num_nodes, num_node_features)
                    Node feature matrix
                y_true : torch.tensor of shape (num_nodes)
                    True labels
                idx = {
                    "train": list 
                        Boolean list indicating whether or not a node is part 
                        of the training set,
                    "val": list
                        Boolean list indicating whether or not a node is part 
                        of the validation set,
                    "test": list
                        Boolean list indicating whether or not a node is part 
                        of the test set
                    }
                evaluator : ogb.nodeproppred.Evaluator
                    Evaluation metric
                emb: 
                    - if layer == 1: torch.tensor of shape (num_nodes, in_channels)
                    - if layer == 2: torch.tensor of shape (num_nodes, hidden_channels)
                    - if layer == 3: torch.tensor of shape (num_nodes, out_channels)
                        layer's representation
                layer : int
                    Default value = 0
                
            Returns:
                acc = {
                    "train": float
                        Training accuracy
                    "val": float
                        Validation accuracy
                    "test": float
                        Test accuracy
                    }
    """
    model.eval()

    _, out, _ = model(x, emb, layer)
    y_pred = out.argmax(dim=-1, keepdim=True)

    acc = {
        "train": evaluator.eval({
            'y_true': y_true[idx["train"]],
            'y_pred': y_pred[idx["train"]],
            })['acc'],
        "val": evaluator.eval({
            'y_true': y_true[idx["val"]],
            'y_pred': y_pred[idx["val"]],
            })['acc'],
        "test": evaluator.eval({
            'y_true': y_true[idx["test"]],
            'y_pred': y_pred[idx["test"]],
            })['acc']
    }

    return acc


def ogbn_arxiv(storage):
    """
        ogbn-arxiv citation graph (https://arxiv.org/pdf/2005.00687.pdf) denoising

            Arguments:
                storage : os.path
                    Absolute path to store ogbn-arxiv dataset

            Returns:
                x : torch.tensor of shape (num_nodes, num_node_features)
                    Node feature matrix
                y_true : torch.tensor of shape (num_nodes)
                    True node label
                edge_index : torch.tensor of shape (2, num_edges)
                    Edge index
                idx = {
                    "train": list 
                        Boolean list indicating whether or not a node is part 
                        of the training set,
                    "val": list
                        Boolean list indicating whether or not a node is part 
                        of the validation set,
                    "test": list
                        Boolean list indicating whether or not a node is part 
                        of the test set
                }
                nodes = {
                    "train": list
                        Integer list of training nodes,
                    "val": list
                        Integer list of validation nodes,
                    "test": list
                        Integer list of test nodes
                }
                x_distorted = {
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
                    },
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
                A_distorted = {
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
                    },
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
    evaluator = Evaluator(name="ogbn-arxiv")
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
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    # Get the nodes corresponding to train, validation and test splits
    nodes = torch.tensor(range(data.num_nodes))
    train_nodes = nodes[train_idx]
    val_nodes = nodes[val_idx]
    test_nodes = nodes[test_idx]
    nodes = {
        "train": train_nodes,
        "val": val_nodes,
        "test": test_nodes
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

    return x, y_true, edge_index, idx, nodes, x_distorted, A_distorted, evaluator