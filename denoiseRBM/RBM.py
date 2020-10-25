
import torch
import torch.nn as nn
import torch.distributions as tdist

from torch.utils.data import DataLoader

import os.path as osp
from tqdm import tqdm

class B_RBM(nn.Module):
    """
        Restricted Boltzmann machine with Bernoulli-distributed hidden units

        Variables:
            v_type
            num_v
            num_h
            theta

        Methods:
            init_params
            get_zeros_like_params
            init_adam
            Ph_v
            Pv_h
            CD
            delta
            adam
            train_RBM
    """
    # Adam hyperparams
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8


    def __init__(self, v_type, num_v, num_h):
        """
            Initializer

                Arguments:
                    v_type : str
                        Gaussian or Bernoulli visible units
                    num_v : int
                        Number of visible units
                    num_h : int
                        Number of hidden units
        """
        super(B_RBM, self).__init__()

        assert v_type in ["Gaussian", "Bernoulli"], \
            ValueError(f"Expected vtype to be Gaussian or Bernoulli, got {v_type} instead.")
        self.v_type = v_type

        assert isinstance(num_v, int), \
            ValueError(f"Expected integer number of visible units, got {type(num_v)} instead.")
        self.num_v = num_v

        assert isinstance(num_h, int), \
            ValueError(f"Expected integer number of hidden units, got {type(num_h)} instead.")
        self.num_h = num_h

        self.init_params()
        self.init_adam()


    def init_params(self):
        """
            Initialize parameters of RBM
        """
        self.theta = nn.ParameterDict({
                        "bv": nn.Parameter(torch.zeros(1, self.num_v, dtype=torch.float, requires_grad=False)),
                        "W": nn.Parameter(torch.randn(self.num_v, self.num_h, dtype=torch.float, requires_grad=False) * 0.01),
                        "bh": nn.Parameter(torch.zeros(1, self.num_h, dtype=torch.float, requires_grad=False))
                    })
        if self.v_type == "Gaussian":
            self.theta["sigma"] = nn.Parameter(torch.randn(1, self.num_v, dtype=torch.float,             requires_grad=False))


    def get_zeros_like_params(self):
        """
            Initialize zeros of the same dimensions as the RBM parameters for RBM optimization
        """
        ans = nn.ParameterDict({
            "bv": nn.Parameter(torch.zeros_like(self.theta["bv"], requires_grad=False)),
            "W": nn.Parameter(torch.zeros_like(self.theta["W"], requires_grad=False)),
            "bh": nn.Parameter(torch.zeros_like(self.theta["bh"], requires_grad=False))
          })
        if self.v_type == "Gaussian":
            ans["sigma"] = nn.Parameter(torch.zeros_like(self.theta["sigma"], requires_grad=False))
        return ans


    def init_adam(self):
        """
            Initialize parameters for Adam optimization
        """
        self.dtheta = self.get_zeros_like_params()
        self.first_moments = self.get_zeros_like_params()
        self.second_moments = self.get_zeros_like_params()


    def Ph_v(self, v):
        """
          Compute P(h_{j}=1|v) for all j
          P(h_{j}=1|v) = sigmoid(b_h_{j} + v @ W) if Bernoulli visible units
          P(h_{j}=1|v) = sigmoid(b_h_{j} + v/sigma @ W) if Gaussian visible units

            Arguments:
                v : torch.tensor of shape (num_v)
                    Visible units values

            Returns:
                hid_probs : torch.tensor of shape (num_h)
                    Probabilities of hidden units
                hid_states : torch.tensor of shape (num_h)
                    Sample drawn from hid_probs
        """
        change = v if self.v_type == "Bernoulli" else (v/self.theta["sigma"])
        hid_acts = self.theta["bh"] + change @ self.theta["W"]
        hid_probs = torch.sigmoid(hid_acts)
        hid_states = torch.where(hid_probs > torch.rand_like(hid_probs), 
                                 torch.ones_like(hid_probs), 
                                 torch.zeros_like(hid_probs))
        #hid_states = tdist.Bernoulli(hid_probs).sample()
        return hid_probs, hid_states


    def Pv_h(self, h):
        """
          Compute P(v_{i}|h) for all i
          P(v_{i}=1|h) = sigmoid(b_v_{i} + h @ W.T) if Bernoulli visible units
          P(v_{i}|h) = normal(b_v_{i} + sigma * h @ W.T, sigma) if Gaussian visible units

            Arguments:
                h : torch.tensor of shape (num_h)
                    Hidden units values

            Returns:
                if Bernoulli:
                    vis_probs : torch.tensor of shape (num_v)
                        Probabilities of visible units
                    vis_states : torch.tensor of shape (num_v)
                        Sample drawn from vis_probs
                if Gaussian:
                    vis_distn : torch.tensor of shape (num_v)
                        Probability distribution that visible units follow
                    vis_states : torch.tensor of shape (num_v)
                        Sample drawn from vis_distn
        """
        if self.v_type == "Gaussian":
          mu = self.theta["bv"] + self.theta["sigma"] * (h @ self.theta["W"].t())
          vis_distn = tdist.Normal(mu, self.theta["sigma"])
          vis_states = vis_distn.sample()
          return vis_distn, vis_states
        else:
          vis_acts = self.theta["bv"] + h @ self.theta["W"].t()
          vis_probs = torch.sigmoid(vis_acts)
          vis_states = torch.where(vis_probs > torch.rand_like(vis_probs), 
                                 torch.ones_like(vis_probs), 
                                 torch.zeros_like(vis_probs))
          #vis_states = tdist.Bernoulli(vis_probs).sample()
          return vis_probs, vis_states


    def CD(self, v0, epoch, eta, K=1):
        """
            Contrastive divergence (CD)

                Arguments:
                    v0 : torch.tensor of shape (num_patterns, num_v)
                        Patterns to store
                    epoch : int
                        Current training epoch
                    eta : float
                        Learning rate
                    K : int, optional
                        Number of CD steps
                            Default value = 1

                Returns:
                    L2 : float
                        L2 distance between the original patterns v0 and the reconstructions vK
        """
        batch_size = v0.size(0)

        # Positive phase
        v_data = v0
        pos_hid_probs, pos_hid_states = self.Ph_v(v_data)

        if self.v_type == "Bernoulli":
          pos_delta_bv, pos_delta_W, pos_delta_bh = self.delta(v_data, pos_hid_probs)
        else:
          pos_delta_bv, pos_delta_W, pos_delta_bh, pos_delta_sigma = self.delta(v_data, pos_hid_probs)
        
        # Gibbs sampling
        h_K = pos_hid_states
        if K > 1:
            for _ in range(K-1):
                pv_K, v_K = self.Pv_h(h_K)
                vis = pv_k if self.v_type == "Bernoulli" else v_k
                ph_K, h_K = self.Ph_v(vis)

        # Negative phase
        neg_vis_probs, neg_vis_states = self.Pv_h(h_K)
        neg_vis = neg_vis_probs if self.v_type == "Bernoulli" else neg_vis_states
        neg_hid_probs, neg_hid_states = self.Ph_v(neg_vis)

        if self.v_type == "Bernoulli":
          neg_delta_bv, neg_delta_W, neg_delta_bh = self.delta(neg_vis_probs, neg_hid_probs)
        else:
          neg_delta_bv, neg_delta_W, neg_delta_bh, neg_delta_sigma = self.delta(neg_vis_states, neg_hid_probs)
        
        # Gradients
        self.dtheta["bv"] = nn.Parameter((pos_delta_bv - neg_delta_bv).reshape(self.theta["bv"].size()) / batch_size)
        self.dtheta["W"] = nn.Parameter((pos_delta_W - neg_delta_W).reshape(self.theta["W"].size()) / batch_size)
        self.dtheta["bh"] = nn.Parameter((pos_delta_bh - neg_delta_bh).reshape(self.theta["bh"].size()) / batch_size)
        if self.v_type == "Gaussian":
          self.dtheta["sigma"] = nn.Parameter((pos_delta_sigma - neg_delta_sigma).reshape(self.theta["sigma"].size()) / batch_size)
        
        # Optimization
        self.adam(epoch+1, eta)

        # Reconstruction error
        return torch.dist(v0, neg_vis, 2)


    def delta(self, v, ph):
        """
            Gradient w.r.t parameters

                Arguments:
                    v : torch.tensor of shape (num_patterns, num_v)
                        Patterns (either original or reconstructions)
                    ph : torch.tensor of shape (num_patterns, num_h)
                        Hidden unit probabilities

                Returns:
                    nabla_bv : torch.tensor of shape (num_patterns, num_v)
                        Gradient w.r.t visible bias
                    nabla_W : torch.tensor of shape (num_v, num_h)
                        Gradient w.r.t weight matrix
                    nabla_bh : torch.tensor of shape (num_patterns, num_h)
                        Gradients w.r.t hidden bias
                    nabla_sigma : torch.tensor of shape (num_patterns, num_v)
                        Gradients w.r.t standard deviation vector
        """
        nabla_bh = torch.sum(ph, axis=0)
        if self.v_type == "Bernoulli":
            nabla_bv = torch.sum(v - self.theta["bv"], axis=0)
            nabla_W = v.t() @ ph
            return nabla_bv, nabla_W, nabla_bh
        else:
            nabla_bv = torch.sum((v - self.theta["bv"]) / torch.pow(self.theta["sigma"], 2), axis=0)
            nabla_W = (v / self.theta["sigma"]).t() @ ph
            nabla_sigma = torch.sum(torch.pow(v - self.theta["bv"], 2) \
                                    / torch.pow(self.theta["sigma"], 3) \
                                    - (ph @ self.theta["W"].t()) \
                                    * (v / torch.pow(self.theta["sigma"], 2)), axis=0)
            return nabla_bv, nabla_W, nabla_bh, nabla_sigma


    def adam(self, t, eta):
        """
            Adam optimization (https://arxiv.org/pdf/1412.6980.pdf)

                Arguments:
                    t : int
                        Time step
                    eta : float
                        Learning rate
        """
        for param in self.theta:
            self.first_moments[param] = nn.Parameter(self.beta_1 * self.first_moments[param] \
                                        + (1-self.beta_1) * self.dtheta[param])
            self.second_moments[param] = nn.Parameter(self.beta_2 * self.second_moments[param] \
                                        + (1-self.beta_2) * torch.pow(self.dtheta[param], 2))
            first_corr = self.first_moments[param] / (1 - self.beta_1**t)
            second_corr = self.second_moments[param] / (1 - self.beta_2**t)
            self.theta[param] = nn.Parameter(self.theta[param] + eta * first_corr / (torch.sqrt(second_corr) + self.eps))


    def train_RBM(self, num_epochs, data, eta, K, batch_size, save_chkpt, device):
        """
            Train RBM

                Arguments:
                    num_epochs : int
                        Number of training epochs
                    data : torch.tensor of shape (num_patterns, num_v)
                        Number of patterns to store
                    eta : float
                        Learning rate
                    K : int
                        Number of contrastive divergence steps
                    batch_size : int
                        Batch size
                    save_chkpt : str
                        Absolute path to the file to save the model state dictionary
                    device : torch.device
                        Torch device

                Returns:
                    l2_list: list
                        List of L2 distances between the original patterns and their reconstructions
        """
        self.train()
        data_loader = DataLoader(data, batch_size=batch_size)
        l2_list = []

        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            batch_l2 = []
            for batch in data_loader:
              batch = batch.to(device)
              l2 = self.CD(batch, epoch, eta, K)
              batch_l2.append(l2)
              pbar.set_description(f"Training RBM: L2={l2}")
            batch_l2 = torch.mean(torch.tensor(batch_l2))
            l2_list.append(batch_l2)
            pbar.update(1)
        torch.save(self.state_dict(), save_chkpt)
        pbar.close()
        print(f"\n{save_chkpt} created!")

        return l2_list