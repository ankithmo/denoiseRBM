
import argparse

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv


class Net(torch.nn.Module):
    def __init__(self, num_in, num_hid, num_out, num_stacks, num_layers, shared_weights, dropout, act):
        super(Net, self).__init__()

        self.conv1 = ARMAConv(num_in, num_hid, num_stacks=num_stacks[0],
                              num_layers=num_layers[0], shared_weights=shared_weights[0], 
                              dropout=dropout[0])

        self.conv2 = ARMAConv(num_hid, num_out, num_stacks=num_stacks[1],
                              num_layers=num_layers[1], shared_weights=shared_weights[1], 
                              dropout=dropout[1],
                              act=act)

    def forward(self, data, emb=None, layer=0):
        z = []

        x = F.dropout(data.x, training=self.training)
        
        x = emb if layer == 1 else self.conv1(x, data.edge_index)
        z.append(None if layer == 1 else x.detach())

        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = emb if layer == 2 else self.conv2(x, data.edge_index)
        z.append(None if layer == 2 else x.detach())

        return z, F.log_softmax(x, dim=1)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    _, out = model(data)
    out = out[data.train_mask]
    F.nll_loss(out, data.y[data.train_mask]).backward()
    optimizer.step()    


def test(model, data, emb=None, layer=0):
    model.eval()
    _, logits = model(data, emb, layer)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def GNN(dataset, checkpoint):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data", dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(num_in=dataset.num_features, num_hid=16, num_out=dataset.num_classes, num_stacks=[3,3], num_layers=[2,2], shared_weights=[True,True], dropout=[0.25,0.25], act=None).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = test_acc = 0
    for epoch in range(1, 401):
        train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    z, _ = model(data)

    checkpoint_dict = {
        "state_dict": model.state_dict(),
        "embeddings": z
    }
    torch.save(checkpoint_dict, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    args = parser.parse_args()
    GNN(dataset=args.dataset, checkpoint=args.checkpoint)