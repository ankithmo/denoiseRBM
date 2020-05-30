
import torch
import numpy as np

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.realpath(__file__)))

import utils as utils

###############################################################################
# Node feature matrix
###############################################################################

def get_dubious_x(mat, percent, device, noisy=True):
    """
        percent: What percentage of the node feature matrix must be made noisy or incomplete
    """
    rand = torch.rand_like(mat)
    ans = mat + rand if noisy else torch.zeros_like(mat)
    return torch.where(rand > percent, mat, ans).to(device)


def distort_percent_x(data, p, device):
    """
        p: What percentage of the node feature matrix must be made dubious
    """
    percent = p/100
    x = utils.get_node_feature_matrix(data)

    # Noisy data
    val_x_noisy = get_dubious_x(x["val"], percent, device, noisy=True)
    test_x_noisy = get_dubious_x(x["test"], percent, device, noisy=True)
    x_noisy = utils.assemble(data.x, data, val_x_noisy, test_x_noisy)

    # Incomplete data
    val_x_incomplete = get_dubious_x(x["val"], percent, device, noisy=False)
    test_x_incomplete = get_dubious_x(x["test"], percent, device, noisy=False)
    x_incomplete = utils.assemble(data.x, data, val_x_incomplete, test_x_incomplete)
    
    return x_noisy, x_incomplete


###############################################################################
# Adjacency matrix
###############################################################################

def get_dubious_A(split, percent, device, noisy=True):
    """
        percent: What percentage of the adjacency matrix must be made noisy or incomplete
    """
    edge_index = []
    for node in split:
        for nbr in split[node]:
            if np.random.rand() < percent:
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
        p: What percentage of the adjacency matrix must be made dubious
    """
    percent = p/100
    nbrs = utils.get_adj_lists(data)
    others = get_dubious_A(nbrs["other"], 0, device)

    # Noisy data
    val_A_noisy = get_dubious_A(nbrs["val"], percent, device, noisy=True)
    test_A_noisy = get_dubious_A(nbrs["test"], percent, device, noisy=True)
    A_noisy = torch.cat([others, val_A_noisy, test_A_noisy], dim=1)

    # Incomplete data
    val_A_incomplete = get_dubious_A(nbrs["val"], percent, device, noisy=False)
    test_A_incomplete = get_dubious_A(nbrs["test"], percent, device, noisy=False)
    A_incomplete = torch.cat([others, val_A_incomplete, test_A_incomplete], dim=1) if percent < 1 else others
    
    return A_noisy, A_incomplete