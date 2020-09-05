
import torch

def get_distorted_x(x, prob, corrupt=True):
    """
        Distorts the node feature matrix (X) either through corruption or through blanking out
        - Corruption: Add uniform random noise to the node feature matrix
        - Blanking out: Replace entries in the node feature matrix with zeros

        Args:
            - x (num_nodes, num_node_features): Node feature matrix
            - prob (float): What fraction of entries in the node feature matrix must be distorted
            - corrupt (bool): Whether to distort X through corruption or blanking out
                              Default value: True (corruption)
        
        Returns:
            - Distorted node feature matrix (num_nodes, num_node_features)
    """
    assert 0 <= prob <= 1, ValueError(f"Expected input in range [0,1], got {prob} instead")
    U = torch.rand_like(x)
    x_n = x + U if corrupt else torch.zeros_like(x)
    return torch.where(U < prob, x_n, x).to(device)


def distort_x_vt(x_val, x_test, prob):
    """
        Distorts the node feature matrix (X) corresponding to the validation and the 
        test sets

        Args:
            - x_val (num_val_nodes, num_node_features): Node feature matrix of the validation nodes
            - x_test (num_test_nodes, num_node_features): Node feature matrix of the test nodes
            - prob (float): What fraction of the entries in the corresponding node feature matrix must 
                            be distorted

        Returns:
            - X_val_c (num_val_nodes, num_node_features): Corrupted validation node feature matrix
            - X_test_c (num_test_nodes, num_node_features): Corrupted test node feature matrix
            - X_val_z (num_val_nodes, num_node_features): Blanked out validation node feature matrix
            - X_test_z (num_test_nodes, num_node_features): Blanked out test node feature matrix
    """
    num_x_val = x_val.size(0) * x_val.size(1)
    num_x_test = x_test.size(0) * x_test.size(1)

    X_val_c = get_distorted_x(x_val, percent, corrupt=True)
    X_val_c_p = (num_x_val - torch.sum(torch.eq(x_val, X_val_c)))/num_x_val
    assert X_val_c_p in [percent-0.05, percent+0.05], 
        ValueError(f"Expected corruption of {percent} in the validation node feature matrix, got {X_val_c_p} instead")

    X_test_c = get_distorted_x(x_test, percent, corrupt=True)
    X_test_c_p = (num_x_test - torch.sum(torch.eq(x_test, X_test_c)))/num_x_test
    assert X_test_c_p in [percent-0.05, percent+0.05],
        ValueError(f"Expected corruption of {percent} in the test node feature matrix, got {X_test_c_p} instead")

    X_val_z = get_distorted_x(x_val, percent, corrupt=False)
    X_val_z_p = (num_x_val - torch.sum(torch.eq(x_val, X_val_z)))/num_x_val
    assert X_val_z_p in [percent-0.05, percent+0.05],
        ValueError(f"Expected blanking out of {percent} in the validation node feature matrix, got {X_val_z_p} instead")

    X_test_z = get_distorted_x(x_test, percent, corrupt=False)
    X_test_z_p = (num_x_test - torch.sum(torch.eq(x_test, X_test_z)))/num_x_test
    assert X_test_z_p in [percent-0.05, percent+0.05],
        ValueError(f"Expected blanking out of {percent} in the test node feature matrix, got {X_test_z_p} instead")

    return X_val_c, X_test_c, X_val_z, X_test_z


def distort_x(x, idx, step=10):
    """
        Creates a dictionary with distortions of the node feature matrix (X) for the validation and test
        sets in increments of `step`

        Args:
            - x (num_nodes, num_node_features): Node feature matrix
            - idx (list): [train_idx, val_idx, test_idx]
            - step (int): Step size for the increments

        Returns:
            - X_distorted = {
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
                }
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

    train_idx = idx["train_idx"]
    val_idx = idx["val_idx"]
    test_idx = idx["test_idx"]

    for i in range(0, 101, step):
        X_val_c, X_test_c, X_val_z, X_test_z = distort_x_vt(x[val_idx], x[test_idx], i/100.)

    X_c[i] = torch.empty_like(x)
    X_c[i][train_idx] = x[train_idx]
    X_c[i][val_idx] = X_val_c
    X_c[i][test_idx] = X_test_c

    X_z[i] = torch.empty_like(x)
    X_z[i][train_idx] = x[train_idx]
    X_z[i][val_idx] = X_val_z
    X_z[i][test_idx] = X_test_z

    X_distorted = {
        "X_c": X_c,
        "X_z": X_z
    }

    return X_distorted