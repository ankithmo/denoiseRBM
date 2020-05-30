
import torch
import numpy as np

import os.path as osp
import sys
dRBM_folder = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, dRBM_folder)

import dRBM.utils as utils

###############################################################################
# Node feature matrix
###############################################################################

def get_dubious_x(mat, norm_p, device, noisy=True):
    """
        norm_p: What percentage in [0,1] of the node feature matrix must be made noisy or incomplete
    """
    assert norm_p >= 0 and norm_p <= 1, ValueError(f"Expected input in range [0,1], got {norm_p} instead")
    rand = torch.rand_like(mat)
    ans = mat + rand if noisy else torch.zeros_like(mat)
    return torch.where(rand < norm_p, ans, mat).to(device)


def distort_percent_x(data, p, device):
    """
        p: What percentage in [0, 100] of the node feature matrix must be made dubious
    """
    norm_p = p/100
    x = utils.get_node_feature_matrix(data)

    # Noisy data
    val_x_noisy = get_dubious_x(x["val"], norm_p, device, noisy=True)
    test_x_noisy = get_dubious_x(x["test"], norm_p, device, noisy=True)
    x_noisy = utils.assemble(data.x, data, val_x_noisy, test_x_noisy)

    # Incomplete data
    val_x_incomplete = get_dubious_x(x["val"], norm_p, device, noisy=False)
    test_x_incomplete = get_dubious_x(x["test"], norm_p, device, noisy=False)
    x_incomplete = utils.assemble(data.x, data, val_x_incomplete, test_x_incomplete)
    
    return x_noisy, x_incomplete


###############################################################################
# Adjacency matrix
###############################################################################

def get_dubious_A(split, norm_p, device, noisy=True):
    """
        norm_p: What percentage in [0,1] of the adjacency matrix must be made noisy or incomplete
    """
    assert norm_p >= 0 and norm_p <= 1, ValueError(f"Expected input in range [0,1], got {norm_p} instead.")
    edge_index = []
    for node in split:
        for nbr in split[node]:
            if np.random.rand() < norm_p:
                if noisy:
                    random_nbr = node
                    while random_nbr in [node] + split[node]:
                        random_nbr = np.random.choice(list(split.keys()))
                        edge_index.append(torch.tensor([node, random_nbr]))
            else:
                edge_index.append(torch.tensor([node, nbr]))
    
    if len(edge_index) > 0:
        edge_index = torch.stack(edge_index)
        return edge_index.t().to(device)
    else:
        return None


def distort_percent_A(data, p, device):
    """
        p: What percentage in [0,100] of the adjacency matrix must be made dubious
    """
    norm_p = p/100
    nbrs = utils.get_adj_lists(data)
    others = get_dubious_A(nbrs["other"], 0, device)

    # Noisy data
    val_A_noisy = get_dubious_A(nbrs["val"], norm_p, device, noisy=True)
    test_A_noisy = get_dubious_A(nbrs["test"], norm_p, device, noisy=True)
    A_noisy = torch.cat([others, val_A_noisy, test_A_noisy], dim=1)

    # Incomplete data
    val_A_incomplete = get_dubious_A(nbrs["val"], norm_p, device, noisy=False)
    test_A_incomplete = get_dubious_A(nbrs["test"], norm_p, device, noisy=False)
    A_incomplete = torch.cat([others, val_A_incomplete, test_A_incomplete], dim=1) if norm_p < 1 else others
    
    return A_noisy, A_incomplete