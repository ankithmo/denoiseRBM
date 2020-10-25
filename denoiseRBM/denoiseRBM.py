
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import os.path as osp
from os import mkdir


def get_perf(model, rbm, x_distorted, x_type, idx, device, layer=None, step=10):
    results = {}
    for perc in tqdm(range(0, 101, step)):
        reqd_x = x_distorted[x_type][perc].to(device)
        z, _, _ = model(reqd_x)

        if layer:
            reqd_z = z[layer]

            h_val, _ = rbm.Ph_v(reqd_z[idx["val"]])
            p_val_recovered, val_recovered = rbm.Pv_h(h_val)

            h_test, _ = rbm.Ph_v(reqd_z[idx["test"]])
            p_test_recovered, test_recovered = rbm.Pv_h(h_test)

            recovered = torch.empty_like(reqd_z)
            recovered[idx["train"]] = reqd_z[idx["train"]]
            recovered[idx["val"]] = val_recovered
            recovered[idx["test"]] = test_recovered
        else:
            h_val, _ = rbm.Ph_v(reqd_x[idx["val"]])
            p_val_recovered, val_recovered = rbm.Pv_h(h_val)

            h_test, _ = rbm.Ph_v(reqd_x[idx["test"]])
            p_test_recovered, test_recovered = rbm.Pv_h(h_test)

            recovered = torch.empty_like(reqd_x)
            recovered[idx["train"]] = reqd_x[idx["train"]]
            recovered[idx["val"]] = val_recovered
            recovered[idx["test"]] = test_recovered

        z, _, _ = model(recovered)
        reco_acc = test(model, recovered, y_true, idx, evaluator)
        print(f"~{x_type}: n_X: {perc}%, Train: {100 * reco_acc['train']:.4f}%, "
                f"Val: {100 * reco_acc['val']:.4f}%, "
                f"Test: {100 * reco_acc['test']:.4f}%\n")

        if not layer:
            for split in reco_acc:
                results[split][perc] = [reco_acc[split], 
                                        recovered[split], 
                                        z[0][split], 
                                        z[1][split], 
                                        z[2][split]
                                        ]
        elif layer == 1:
            for split in reco_acc:
                results[split][perc] = [reco_acc[split], 
                                        recovered[split], 
                                        z[1][split], 
                                        z[2][split]
                                        ]
        elif layer == 2:
            for split in reco_acc:
                results[split][perc] = [reco_acc[split], 
                                        recovered[split], 
                                        z[2][split]
                                        ]
        elif layer == 3:
            for split in reco_acc:
                results[split][perc] = [reco_acc[split], 
                                        recovered[split]
                                        ]

    return results


def denoiseRBM(model, rbm, x, x_distorted, idx, device, chkpt_path, num_hid, 
                num_epochs, lr, CD, batch_size, step, layer=None):
    psi_results = {}

    # Train RBM (if not present)
    rbm_path = osp.join(chkpt_path, f"x_RBM_{num_hid}_{num_epochs}.pt")
    if osp.exists(rbm_path):
        rbm.load_state_dict(torch.load(rbm_path, map_location=device))
    else:
        l2 = rbm.train_RBM(num_epochs, x[idx["train"]].to(device), lr, CD, 
                            batch_size, rbm_path, device)
    plt.plot(l2)
    plt.xlabel("#(epochs)")
    plt.ylabel("L2 error")
    plt.title("d(data, reconstructions)")
    plt.show()

    # Compute Euclidean distance
    h, _ = rbm.Ph_v(x)
    p_tilde, tilde = rbm.Pv_h(h)

    h_val, _ = rbm.Ph_v(x[idx["val"]])
    p_val_tilde, val_tilde = rbm.Pv_h(h_val)

    h_test, _ = rbm.Ph_v(x[idx["test"]])
    p_test_tilde, test_tilde = rbm.Pv_h(h_test)

    print(f"d(data, reconstructions) = {torch.dist(x, x_tilde, 2)}"
            f"d(validation data, reconstructions) = {torch.dist(x[idx['val']], x_val_tilde, 2)}"
            f"d(test data, reconstructions) = {torch.dist(x[idx['test']], x_test_tilde, 2)}")

    # Concatenate the validation and test set reconstructions with the original training data
    x_conc = torch.empty_like(x)
    x_conc[idx["train"]] = x[idx["train"]]
    x_conc[idx["val"]] = x_val_tilde
    x_conc[idx["test"]] = x_test_tilde

    # Compute accuracy between data and reconstructions
    data_acc = test(MLP_model, x, y_true, idx, evaluator)
    print("Data:")
    print(f"Train: {100 * data_acc['train']:.2f}%, "
            f"Valid data: {100 * data_acc['val']:.2f}%, "
            f"Test data: {100 * data_acc['test']:.2f}%")

    reco_acc = test(MLP_model, x_conc, y_true, idx, evaluator)
    print("Reconstructions:")
    print(f"Train: {100 * reco_acc['train']:.2f}%, "
            f"Valid reco: {100 * reco_acc['val']:.2f}%, "
            f"Test reco: {100 * reco_acc['test']:.2f}%")

    # Compute reconstructions
    layer = layer if layer else None
    psi_results["X_c"] = get_perf(model, rbm, x_distorted, "X_c", idx, device, 
                                    layer, step=step)
    psi_results["X_z"] = get_perf(model, rbm, x_distorted, "X_z", idx, device, 
                                    layer, step=step)

    return psi_results