
import torch
from collections import defaultdict

def get_node_feature_matrix(data):
    x = {
        "train": data.x[data.train_mask],
        "val": data.x[data.val_mask],
        "test": data.x[data.test_mask],
    }
    return x

def get_node_tensors(data):
    nodes = {
        "train": torch.where(data.train_mask)[0],
        "val": torch.where(data.val_mask)[0],
        "test": torch.where(data.test_mask)[0]
    }
    return nodes

def get_adj_lists(data):
    nbrs = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list),
        "other": defaultdict(list)
    }
    nodes = get_node_tensors(data)
    for edge in data.edge_index.t():
        src, dst = edge
        src = src.item()
        dst = dst.item()
        if src in nodes["train"]:
            nbrs["train"][src].append(dst)
        elif src in nodes["val"]:
            nbrs["val"][src].append(dst)
        elif src in nodes["test"]:
            nbrs["test"][src].append(dst)
        else:
            nbrs["other"][src].append(dst)
    return nbrs

def assemble(mat, data, val, test):
    assembled = []

    nodes = get_node_tensors(data)
    for node, vector in enumerate(mat):
        if node in nodes["val"]:
            idx = torch.where(nodes["val"] == node)[0].item()
            assembled.append(val[idx])
        elif node in nodes["test"]:
            idx = torch.where(nodes["test"] == node)[0].item()
            assembled.append(test[idx])
        else:
            assembled.append(vector)
    assembled = torch.stack(assembled)
    assert assembled.size() == mat.size(), \
        ValueError("Assembled tensor does not have the same shape as the original tensor.")
    return assembled