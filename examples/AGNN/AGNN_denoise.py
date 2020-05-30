
import argparse

import os.path as osp
from os import mkdir
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import sys
sys.path.append(osp.dirname(osp.realpath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))

import AGNN
from dRBM.dRBM import RBM
from dRBM.dRBM.utils import assemble
from dRBM.dRBM.denoise import denoise, analysis

def denoise_GNN(dataset, GNN_checkpoint, layer, num_epochs, num_hidden, lr, K, 
         RBM_checkpoint, results):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    path = osp.join(osp.dirname(osp.realpath(__file__)), "data", dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # Model
    model = AGNN.Net(dataset.num_features, 16, dataset.num_classes).to(device)

    # Embeddings
    chkpt = torch.load(GNN_checkpoint)
    model.load_state_dict(chkpt["state_dict"])
    z = chkpt["embeddings"][layer-1].to(device)

    # Verify that the model and the checkpoints work
    train_acc, val_acc, test_acc = AGNN.test(model, data, z, layer)
    print("\nVerify loading:")
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # Train RBM on these embeddings
    z_RBM = RBM.RBM(z.size(1), num_hidden)
    z_RBM = z_RBM.to(device)
    l2 = z_RBM.train_RBM(num_epochs, z[data.train_mask], lr, K, RBM_checkpoint)

    # Check the reconstructions error
    plt.plot(l2)
    plt.xlabel("# epochs")
    plt.ylabel("Reconstruction error")
    plt.show()

    # How good is our trained RBM?
    h, _ = z_RBM.Ph_v(z)
    z_tilde = z_RBM.Pv_h(h)
    train_acc, val_acc, test_acc = AGNN.test(model, data, z_tilde, layer)
    print("\nEvaluate when entire embeddings tensor is reconstructed:")
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # What if we only reconstruct the validation and the test nodes?
    h_val, _ = z_RBM.Ph_v(z[data.val_mask])
    z_val_tilde = z_RBM.Pv_h(h_val)
    h_test, _ = z_RBM.Ph_v(z[data.test_mask])
    z_test_tilde = z_RBM.Pv_h(h_test)
    z_tilde_a = assemble(z, data, z_val_tilde, z_test_tilde)
    train_acc, val_acc, test_acc = AGNN.test(model, data, z_tilde_a, layer)
    print("\nEvaluate when valid and test embedding tensors are reconstructed:")
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # Verify that denoise is working
    ans = denoise(model, data, AGNN.test, layer, z_RBM, device)
    print("\nDenoise results when original node feature matrix and adjacency matrix are passed:")
    print(ans)

    # Denoise analysis
    print("\nPerform denoise analysis:")
    results = analysis(model, data, AGNN.test, layer, z_RBM, device, results)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--GNN_checkpoint", type=str, help="Absolute path to the checkpoint for the GNN")
    parser.add_argument("--layer", type=int, help="Embedding layer on which to train the RBM")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs for the RBM")
    parser.add_argument("--num_hidden", type=int, help="Number of hidden units for the RBM")
    parser.add_argument("--lr", type=float, help="Learning rate for the RBM")
    parser.add_argument("--K", type=int, help="Number of Gibbs sampling")
    parser.add_argument("--RBM_checkpoint", type=str, help="Absolute path to the checkpoint for the RBM")
    parser.add_argument("--results", type=str, help="Absolute path to the location where the dataframe should be saved")
    args = parser.parse_args()
    denoise_GNN(dataset=args.dataset, GNN_checkpoint=args.GNN_checkpoint, layer=args.layer, 
         num_epochs=args.num_epochs, num_hidden=args.num_hidden, lr=args.lr, 
         K=args.K, RBM_checkpoint=args.RBM_checkpoint, results=args.results)