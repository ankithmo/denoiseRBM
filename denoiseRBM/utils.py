
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import multivariate_normal as mvn

from sklearn.manifold import TSNE

import plotly.express as px
from bokeh.palettes import Category20_20, Category20b_20, Accent8


def Z(d, beta):
    """
        Returns a d-dimensional vector drawn from \mathcal{N}(0,\beta^{2}I_{d})

            Arguments:
                d : int
                    Dimensionality
                beta : float
                    Standard deviation for the isotropic Gaussian

            Returns:
                d-dimensional vector
    """
    return mvn.MultivariateNormal(torch.zeros(d), torch.diag(torch.ones(d) * beta**2))


def get_tsne(x, idx, n_components):
    """
        Get 2D t-SNE coordinates for x[idx]

            Arguments:
                x : torch.tensor of shape (num_nodes, num_node_features)
                    Node feature matrix
                idx : torch.tensor
                    train_idx/val_idx/test_idx
                n_components : int
                    Number of t-SNE components
            
            Returns:
                y_emb : torch.tensor of shape (num_nodes, 2)
                    t-SNE embeddings
    """
    return TSNE(n_components=n_components).fit_transform(x[idx].cpu().detach().numpy())

def plot_tsne(x, idx, n_components, y_true):
    projections = get_tsne(x, idx, n_components)
    colormap = np.array(Category20_20 + Category20b_20 + Accent8)
    colors = [colormap[l] for l in y_true[idx]]
    return px.scatter_3d(projections, x=0, y=1, z=2, color=colors)
