
import argparse

import os.path as osp
from os import mkdir

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv


class Net(torch.nn.Module):
    def __init__(self, num_in, num_hid, num_out):
        super(Net, self).__init__()
        self.lin1 = torch.nn.Linear(num_in, num_hid)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(num_hid, num_out)

    def forward(self, data, emb=None, layer=0):
        z = []

        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))

        x = emb if layer == 1 else self.prop1(x, data.edge_index)
        z.append(None if layer == 1 else x.detach())

        x = emb if layer == 2 else self.prop2(x, data.edge_index)
        z.append(None if layer == 2 else x.detach())

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

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


def main(dataset, checkpoint):
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data", dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net(dataset.num_features, 16, dataset.num_classes).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train(model, optimizer, data)
        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    z, _ = model(data)

    checkpoint = {
        "state_dict": model.state_dict(),
        "embeddings": z
    }
    chkpt_dir = osp.join(osp.dirname(osp.realpath(__file__)), "checkpoints")
    if not osp.exists(chkpt_dir): mkdir(chkpt_dir)
    path = osp.join(chkpt_dir, checkpoint)
    torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset")
    parser.add_argument("--checkpoint", help="Checkpoint path")
    args = parser.parse_args()
    main(dataset=args.dataset, checkpoint=args.checkpoint)