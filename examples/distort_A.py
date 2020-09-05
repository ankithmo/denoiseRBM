
from tqdm import tqdm
import numpy as np
import torch

def get_distorted_a(edge_index, nodes, idx, prob, dataset, corrupt=True):
    """
        Random distortion of the edge index either through corruption or through blanking out
        - Corruption: Dataset specific corruption of edges
        - Blanking out: Discard edges

        Args:
            - edge_index (2, num_edges): edge index
            - nodes (list): [train_nodes, val_nodes, test_nodes]
            - idx (list): [train_idx, val_idx, test_idx]
            - prob (float): What fraction of the entries in the corresponding edge index must 
                            be distorted
            - dataset (str): For "ogbn-arxiv":
                                - If source node is in the validation set, 
                                  then its respective destination node is a random node from 
                                  either the training or the validation sets.
                                - If source node is in the test set,
                                  then its respective destination node is a random node from
                                  the training, validation or the test sets.
                             For "WikiCS":
                                - If source node is in the validation or test sets,
                                  then its respective destination node is a random node.
            - corrupt (bool): Whether to distort edge index through corruption or blanking out
                                Default value: True (corruption)

        Returns:
            - Percentage of distortion in the validation set
            - Percentage of distortion in the test set
            - Distorted edge index
        
    """
    assert 0 <= prob <= 1, ValueError(f"Expected input in range [0,1], got {prob} instead.")

    train_nodes = nodes["train_nodes"]
    val_nodes = nodes["val_nodes"]
    test_nodes = nodes["test_nodes"]

    train_idx = idx["train_idx"]
    val_idx = idx["val_idx"]
    test_idx = idx["test_idx"]

    distorted_edge_index = []
    val_count = 1e-9
    val_changed = 0.
    test_count = 1e-9
    test_changed = 0.

    pbar = tqdm(total=edge_index.size(1))
    for edge in edge_index.t():
        if edge[0] in val_nodes:
            val_count += 1
            if np.random.rand() < prob:
                val_changed += 1
                if corrupt:
                    dest = np.random.choice(train_idx.numpy()) if dataset == "ogbn-arxiv" else np.random.choice(np.delete(nodes, edge[1]))
                    new_edge = torch.tensor([edge[0], dest])
                    distorted_edge_index.append(new_edge)
            else:
                distorted_edge_index.append(edge)
        elif edge[0] in test_nodes:
            test_count += 1
            if np.random.rand() < prob:
                test_changed += 1
                if corrupt:
                    dest = np.random.choice(np.hstack((train_idx.numpy(), val_idx.numpy()))) if dataset == "ogbn-arxiv" else np.random.choice(np.delete(nodes, edge[1]))
                    new_edge = torch.tensor([edge[0], dest])
                    distorted_edge_index.append(new_edge)
            else:
                distorted_edge_index.append(edge)
        else:
            distorted_edge_index.append(edge)
        pbar.update(1)
    pbar.close()

    return val_changed/val_count, test_changed/test_count, torch.stack(distorted_edge_index).t()

def distort_a_vt(edge_index, nodes, idx, percent):
    """
        Distorts the edge_index corresponding to the validation and the test sets

        Args:
            - edge_index (2, num_edges): edge index
            - nodes (list): [train_nodes, val_nodes, test_nodes]
            - idx (list): [train_idx, val_idx, test_idx]
            - percent (float): What fraction of the edge indices must be distorted

        Returns:

    """
    p_val_c, p_test_c, A_c = get_distorted_a(edge_index, nodes, idx, percent, corrupt=True)
    assert p_val_c in [percent-0.05, percent+0.05], 
        ValueError(f"Expected corruption of {percent} in the edge index corresponding to the validation set, got {p_val_c} instead")
    assert p_test_c in [percent-0.05, percent+0.05],
        ValueError(f"Expected corruption of {percent} in the edge index corresponding to the test set, got {p_test_c} instead")

    p_val_z, p_test_z, A_z = get_distorted_a(edge_index, nodes, idx, percent, corrupt=False)
    assert p_val_z in [percent-0.05, percent+0.05], 
        ValueError(f"Expected blanking out of {percent} in the edge index corresponding to the validation set, got {p_val_z} instead")
    assert p_test_z in [percent-0.05, percent+0.05],
        ValueError(f"Expected blanking out of {percent} in the edge index corresponding to the test set, got {p_test_z} instead")
  
    return A_c, A_z

def distort_a(edge_index, nodes, idx, step=10):
    """
        Creates a dictionary with distortions of the edge index for the validation and test
        sets in increments of `step`

        Args:
            - edge_index (2, num_edges): Edge index
            - nodes (list): [train_nodes, val_nodes, test_nodes]
            - idx (list): [train_idx, val_idx, test_idx]
            - step (int): Step size for the increments

        Returns:
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
    A_c, A_z = {}, {}

    for i in range(0, 101, 10):
        A_c[i], A_z[i] = distort_a_vt(edge_index, nodes, idx, i/100.)

    A_distorted = {
        "A_c": A_c,
        "A_z": A_z
    }

    return A_distorted