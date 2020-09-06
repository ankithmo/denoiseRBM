
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import os
import os.path as osp

import WikiCS
import DNN


def check_perf(perf_0, perf_1, tol=3):
    """
        Check whether the difference between two performances is within tolerance

        Args:
            - perf_0: First set of performances
            - perf_1: Second set of performances
            - tol: Tolerance limit of the difference between the performances
    """
    for dataset in ["train_acc", "val_acc", "test_acc"]: 
        assert perf_0[dataset]-tol <= perf_1[dataset] <= perf_0[dataset]+tol, \
            ValueError(f"Expected accuracy of {100*perf_0[dataset]:.2f}%, got {100*perf_1[dataset]:.2f}%")


def denoiseRBM(model, optimizer, train_fn, test_fn, num_epochs, model_chkpt_path, MI=False):
    # Move to denoiseRBM once finished
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(1, 1 + num_epochs):
            loss = train_fn(model, x, y_true, train_idx, optimizer)
            acc = test_fn(model, x, y_true, idx)
            pbar.set_description(f"Training model {epoch:02d}: Loss={loss:.4f}, 
                                    Train={100*acc["train_acc"]:.2f}%, 
                                    Valid={100*acc["val_acc"]:.2f}%, 
                                    Test={100*acc["test_acc"]:.2f}%")
            pbar.update(1)
    pbar.close()

    # Save trained model
    if not osp.exists(chkpt_path):
        z, _, _ = model(x)
        chkpt = {
            "state_dict": model.state_dict(),
            "embeddings": z
        }
        torch.save(chkpt, chkpt_path)
    else:
        chkpt = torch.load(chkpt_path)
        model.load_state_dict(chkpt["state_dict"])
        model = chkpt["embeddings"]

    # Check that the accuracies with input and hidden representations are about the same
    acc_z0 = test_fn(model, x, y_true, idx)
    acc_z1 = test_fn(model, x, y_true, idx, z[0], layer=1)
    acc_z2 = test_fn(model, x, y_true, idx, z[1], layer=2)
    acc_z3 = test_fn(model, x, y_true, idx, z[2], layer=3)

    check_accs(acc_z0, acc_z1, 3)
    check_accs(acc_z0, acc_z2, 3)
    check_accs(acc_z0, acc_z3, 3)


def denoise_WikiCS(storage, split, nn_algo, hidden_channels, sigma, device, lr, num_epochs, dropout, MI=False):
    # Get WikiCS noisy dataset
    x, y_true, C, edge_index, idx, nodes, x_distorted, A_distorted = WikiCS.WikiCS(storage, 
                                                                                    split)

    # Move x and y_true to device
    x = x.to(device)
    y_true = y_true.to(device)

    train_idx = idx["train_idx"]
    val_idx = idx["val_idx"]
    test_idx = idx["test_idx"]
    
    # Train model
    model = DNN.DNN(nn_algo, x.size(-1), hidden_channels, C, device, None, 
                    dropout).to(device)
    model.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model_chkpt_path = osp.join(storage, f"WikiCS_{nn_algo}.pt")
    denoiseRBM(model, optimizer, DNN.train, WikiCS.WikiCS_test, num_epochs, 
                model_chkpt_path)

    if MI:
        # Train MLP model for MI estimation
        Z_ = DNN.Z(hidden_channels, sigma)
        MI_model = DNN.DNN(nn_algo, x.size(-1), hidden_channels, C, 
                                device, Z_, dropout).to(device)
        MI_model.reset_parameters()
        MI_optimizer = optim.Adam(MI_model.parameters(), lr=lr)
        MI_model_chkpt_path = osp.join(storage, f"WikiCS_MI_{nn_algo}.pt")
        denoiseRBM(MI_model, MI_optimizer, DNN.train, WikiCS.WikiCS_test, num_epochs, 
                    MI_model_chkpt_path, MI=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Denoise WikiCS")
    parser.add_argument("--storage", default=osp.join(osp.dirname(osp.realpath(__file__)), ".", "results"),
                        help="Absolute path to store the results of denoising WikiCS")
    parser.add_argument("--split", default=0, help="Which of the 20 splits to use?")
    args = parser.parse_args()
    
    # Create storage directory if it doesn't exist
    if not osp.exists(args.storage):
        print(f"{args.storage} does not exist. Creating {args.storage}...", end="")
        os.mkdir(args.storage)
    print("Done!")

    WikiCS(args.storage, args.split)