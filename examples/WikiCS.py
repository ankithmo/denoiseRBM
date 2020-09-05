
import argparse
import os.path as osp

import torch
from torch_geometric.datasets import WikiCS

from distort_X import distort_x

# Arguments
import argparse
parser = argparse.ArgumentParser(description="WikiCS")
parser.add_argument("--storage", default=osp.join(".", "data"), 
                    help="Absolute path to store WikiCS dataset")
parser.add_argument("--split", default=0, help="Which of the 20 splits to use?")
args = parser.parse_args()

def wikiCS():
    """
        WikiCS citation graph (https://arxiv.org/pdf/2007.02901.pdf) denoising

        Returns:
            - nodes = {
                "train_nodes": Integer list of training nodes,
                "val_nodes": Integer list of validation nodes,
                "test_nodes": Integer list of test nodes
              }
            - idx = {
                "train_idx": Boolean list indicating whether or not a node is 
                           part of the training set,
                "val_idx": Boolean list indicating whether or not a node is 
                           part of the validation set,
                "test_idx": Boolean list indicating whether or not a node is 
                           part of the test set
              }
    """
    # Get node feature matrix, labels and edge index
    dataset = WikiCS(args.storage)
    data = dataset[0]
    x = data.x
    y_true = data.y
    edge_index = edge_index

    # Get indices corresponding to train, validation and test splits
    train_idx = data.train_mask[:, args.split]
    val_idx = data.val_mask[:, args.split]
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
    nodes = torch.range(data.num_nodes).to(dtype=torch.int)
    train_nodes = nodes[train_idx]
    val_nodes = nodes[val_idx]
    test_nodes = nodes[test_idx]
    nodes = {
        "train_nodes": train_nodes,
        "val_nodes": val_nodes,
        "test_nodes": test_nodes
    }

    # Distort the node feature matrix
    x_distorted = distort_x(x, idx)

    # Distort the edge index
    A_distorted = distort_A(edge_index, nodes, idx)


if __name__ == "__main__":
    WikiCS()