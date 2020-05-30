
import torch
import torch_geometric.data as pyg_data

import numpy as np
import pandas as pd

from tqdm import tqdm
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.realpath(__file__)))

from utils import assemble
from dubious import distort_percent_x, distort_percent_A

def denoise(model, data, test, layer, z_RBM, device):
    """
        Take as input both the node feature matrix and the edge index.
        Two pass solution:
        1. Generate embeddings using these inputs
        2. Pass the denoised embeddings to get final prediction
    """
    ans = [] 
    # train accuracy, val accuracy, test accuracy, 
    # train accuracy, denoised val accuracy, denoised test accuracy

    train_acc, val_acc, test_acc = test(model, data)
    z_list, _ = model(data)
    z = z_list[layer-1].to(device)
    train_acc_v, val_acc_v, test_acc_v = test(model, data, z, layer)
    assert train_acc == train_acc_v, ValueError(f"Train accuracy mismatch. Expected {train_acc}, got {train_acc_v}")
    assert val_acc == val_acc_v, ValueError(f"Validation accuracy mismatch. Expected {val_acc}, got {val_acc_v}")
    assert test_acc == test_acc_v, ValueError(f"Test accuracy mismatch. Expected {test_acc}, got {test_acc_v}")

    ans.append([train_acc, val_acc, test_acc])
    
    h_val, _ = z_RBM.Ph_v(z[data.val_mask])
    z_val_tilde = z_RBM.Pv_h(h_val)
    h_test, _ = z_RBM.Ph_v(z[data.test_mask])
    z_test_tilde = z_RBM.Pv_h(h_test)
    z_tilde_a = assemble(z, data, z_val_tilde, z_test_tilde)
    den_train_acc, den_val_acc, den_test_acc = test(model, data, z_tilde_a, layer)

    ans.append([den_train_acc, den_val_acc, den_test_acc])
    
    return ans

def analysis(model, data, test, layer, z_RBM, device, save_path):
    results = []
    cols = ['train_mask', 'val_mask', 'test_mask', 'y']
    col_names = ["distort A%", "distort X%", "Type", "Train accuracy", "Validation accuracy", 
                 "Test accuracy"]
    col_dtypes = {"A%": float, "X%": float, "Type": str, "Train accuracy": float, 
                  "Validation accuracy": float, "Test accuracy": float}

    pbar = tqdm(total=484)
    for a_p in range(0, 101, 10):
        A_noisy, A_incomplete = distort_percent_A(data, a_p, device)

        for x_p in range(0, 101, 10):
            X_noisy, X_incomplete = distort_percent_x(data, x_p, device)

            # noisy X, noisy A
            new_data = pyg_data.Data(x=X_noisy, edge_index=A_noisy)
            for col in cols:
                new_data[col] = data[col]
            ans = denoise(model, new_data, test, layer, z_RBM, device)
            ans[0] = np.asarray([a_p, x_p, "noisy X, noisy A"] + ans[0])
            ans[1] = np.asarray([a_p, x_p, f"Denoised layer {layer}: noisy X, noisy A"] + ans[1])
            if len(results) == 0:
                results = np.asarray(ans)
            else:
                results = np.vstack((results, np.asarray(ans)))
            pbar.set_description(f"Denoising {a_p}% noisy adjacency matrix and {x_p}% of noisy node feature matrix")
            pbar.update(1)

            # noisy X, incomplete A
            new_data = pyg_data.Data(x=X_noisy, edge_index=A_incomplete)
            for col in cols:
                new_data[col] = data[col]
            ans = denoise(model, new_data, test, layer, z_RBM, device)
            ans[0] = np.asarray([a_p, x_p, "noisy X, incomplete A"] + ans[0])
            ans[1] = np.asarray([a_p, x_p, f"Denoised layer {layer}: noisy X, incomplete A"] + ans[1])
            results = np.vstack((results, np.asarray(ans)))
            pbar.set_description(f"Denoising {a_p}% noisy adjacency matrix and {x_p}% incomplete node feature matrix")
            pbar.update(1)

            # incomplete X, noisy A
            new_data = pyg_data.Data(x=X_incomplete, edge_index=A_noisy)
            for col in cols:
                new_data[col] = data[col]
            ans = denoise(model, new_data, test, layer, z_RBM, device)
            ans[0] = np.asarray([a_p, x_p, "incomplete X, noisy A"] + ans[0])
            ans[1] = np.asarray([a_p, x_p, f"Denoised layer {layer}: incomplete X, noisy A"] + ans[1])
            results = np.vstack((results, np.asarray(ans)))
            pbar.set_description(f"Denoising {a_p}% incomplete adjacency matrix and {x_p}% noisy node feature matrix")
            pbar.update(1)

            # incomplete X, incomplete A
            new_data = pyg_data.Data(x=X_incomplete, edge_index=A_incomplete)
            for col in cols:
                new_data[col] = data[col]
            ans = denoise(model, new_data, test, layer, z_RBM, device)
            ans[0] = np.asarray([a_p, x_p, "incomplete X, incomplete A"] + ans[0])
            ans[1] = np.asarray([a_p, x_p, f"Denoised layer {layer}: incomplete X, incomplete A"] + ans[1])
            results = np.vstack((results, np.asarray(ans)))
            pbar.set_description(f"Denoising {a_p}% incomplete adjacency matrix and {x_p}% incomplete node feature matrix")
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results, columns=col_names)
    df = df.astype(col_dtypes)
    df.to_pickle(save_path)
    print(f"\n{save_path} created!")

    return df