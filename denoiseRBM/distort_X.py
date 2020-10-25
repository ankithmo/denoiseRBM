
import torch
from tqdm import tqdm

def get_distorted_x(x, prob, corrupt=True):
    """
        Distorts the node feature matrix (X) either through corruption or through blanking out
        - Corruption: Add uniform random noise in [0,1] to the node feature matrix
        - Blanking out: Replace entries in the node feature matrix with zeros

            Parameters:
                x : torch.tensor of shape (num_nodes, num_node_features)
                    Node feature matrix
                prob : float
                    What fraction of entries in the node feature matrix must be distorted
                corrupt : bool, optional
                    Whether to distort X through corruption or blanking out
                        Default value: True (corruption)
        
            Returns:
                Distorted node feature matrix : torch.tensor of shape (num_nodes, num_node_features)
    """
    assert 0 <= prob <= 1, ValueError(f"Expected input in range [0,1], got {prob} instead")
    U = torch.rand_like(x)
    x_n = x + U if corrupt else torch.zeros_like(x)
    return torch.where(U < prob, x_n, x)


def distort_x_vt(x_val, x_test, prob):
    """
        Distorts the node feature matrix (X) corresponding to the validation and the 
        test sets

            Parameters:
                x_val : torch.tensor of shape (num_val_nodes, num_node_features)
                    Node feature matrix of the validation nodes
                x_test : torch.tensor of shape (num_test_nodes, num_node_features)
                    Node feature matrix of the test nodes
                prob : float
                    What fraction of the entries in the corresponding node feature matrix must 
                    be distorted

            Returns:
                X_val_c : torch.tensor of shape (num_val_nodes, num_node_features)
                    Corrupted validation node feature matrix
                X_test_c : torch.tensor of shape (num_test_nodes, num_node_features)
                    Corrupted test node feature matrix
                X_val_z : torch.tensor of shape (num_val_nodes, num_node_features)
                    Blanked out validation node feature matrix
                X_test_z : torch.tensor of shape (num_test_nodes, num_node_features)
                    Blanked out test node feature matrix
    """
    num_x_val = x_val.size(0) * x_val.size(1)
    num_x_test = x_test.size(0) * x_test.size(1)

    # Corrupted validation node feature matrix
    X_val_c = get_distorted_x(x_val, prob, corrupt=True)
    X_val_c_p = ((num_x_val - torch.sum(torch.eq(x_val, X_val_c))).item()/num_x_val)
    assert prob-0.05 <= X_val_c_p <= prob+0.05, \
        ValueError(f"Expected corruption of {prob} in the validation node feature matrix, got {X_val_c_p} instead")

    # Corrupted test node feature matrix
    X_test_c = get_distorted_x(x_test, prob, corrupt=True)
    X_test_c_p = ((num_x_test - torch.sum(torch.eq(x_test, X_test_c))).item()/num_x_test)
    assert prob-0.05 <= X_test_c_p <= prob+0.05, \
        ValueError(f"Expected corruption of {prob} in the test node feature matrix, got {X_test_c_p} instead")

    # Blanked out validation node feature matrix
    X_val_z = get_distorted_x(x_val, prob, corrupt=False)
    X_val_z_p = ((num_x_val - torch.sum(torch.eq(x_val, X_val_z))).item()/num_x_val)
    assert prob-0.05 <= X_val_z_p <= prob+0.05, \
        ValueError(f"Expected blanking out of {prob} in the validation node feature matrix, got {X_val_z_p} instead")

    # Blanked out test node feature matrix
    X_test_z = get_distorted_x(x_test, prob, corrupt=False)
    X_test_z_p = ((num_x_test - torch.sum(torch.eq(x_test, X_test_z))).item()/num_x_test)
    assert prob-0.05 <= X_test_z_p <= prob+0.05, \
        ValueError(f"Expected blanking out of {prob} in the test node feature matrix, got {X_test_z_p} instead")

    return X_val_c, X_test_c, X_val_z, X_test_z


def distort_x(x, idx, step=10):
    """
        Creates a dictionary with distortions of the node feature matrix (X) for the validation and test
        sets in increments of `step`

            Parameters:
                x : torch.tensor of shape (num_nodes, num_node_features)
                    Node feature matrix
                idx : list 
                    [train, val, test]
                step : int, optional
                    Step size for the increments of distortion
                        Default: 10

            Returns:
                X_distorted = {
                    "X_c": {
                        0: Concatenation of original train node feature matrix, 
                            0% corrupted (original) validation node feature matrix, 
                            0% corrupted (original) test node feature matrix,
                        0+step: Concatenation of original train node feature matrix, 
                            (0+step)% corrupted (original) validation node feature matrix, 
                            (0+step)% corrupted (original) test node feature matrix,
                        .
                        .
                        .
                        100: Concatenation of original train node feature matrix, 
                            100% corrupted validation node feature matrix, 
                            100% corrupted test node feature matrix
                    },
                    "X_z": {
                        0: Concatenation of original train node feature matrix, 
                            0% blanked out (original) validation node feature matrix, 
                            0% blanked out (original) test node feature matrix,
                        0+step: Concatenation of original train node feature matrix, 
                            (0+step)% blanked out (original) validation node feature matrix, 
                            (0+step)% blanked out (original) test node feature matrix,
                        .
                        .
                        .
                        100: Concatenation of original train node feature matrix, 
                            100% blanked out validation node feature matrix, 
                            100% blanked out test node feature matrix
                    }
                }
    """
    X_c, X_z = {}, {}

    with tqdm(total=100/step) as pbar:
        pbar.set_description("Distorting node feature matrix")
        for i in range(0, 101, step):
            X_val_c, X_test_c, X_val_z, X_test_z = distort_x_vt(x[idx["val"]], 
                                                                x[idx["test"]], 
                                                                i/100.)
            
            X_c[i] = torch.empty_like(x)
            X_c[i][idx["train"]] = x[idx["train"]]
            X_c[i][idx["val"]] = X_val_c
            X_c[i][idx["test"]] = X_test_c

            X_z[i] = torch.empty_like(x)
            X_z[i][idx["train"]] = x[idx["train"]]
            X_z[i][idx["val"]] = X_val_z
            X_z[i][idx["test"]] = X_test_z
            
            pbar.update(1)
    pbar.close()    

    X_distorted = {
        "X_c": X_c,
        "X_z": X_z
    }

    return X_distorted