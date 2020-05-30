
import torch
import torch.nn as nn
import torch.distributions as tdist

import os.path as osp
from tqdm import tqdm

class RBM(nn.Module):

    # Adam hyperparams
    beta_1 = 0.9
    beta_2 = 0.999
    eps = 1e-8

    def __init__(self, num_v, num_h):
        super(RBM, self).__init__()

        assert isinstance(num_v, int), TypeError(f"Expected integer number of visible units, got {type(num_v)} instead.")
        self.num_v = num_v

        assert isinstance(num_h, int), TypeError(f"Expected integer number of hidden units, got {type(num_h)} instead.")
        self.num_h = num_h

        self.init_params()
        self.init_adam()

    def init_params(self):
        self.theta = nn.ParameterDict({
                        "bv": nn.Parameter(torch.zeros(1, self.num_v, dtype=torch.float, requires_grad=False)),
                        "W": nn.Parameter(torch.randn(self.num_v, self.num_h, dtype=torch.float, requires_grad=False) * 0.01),
                        "bh": nn.Parameter(torch.zeros(1, self.num_h, dtype=torch.float, requires_grad=False)),
                        "sigma": nn.Parameter(torch.randn(1, self.num_v, dtype=torch.float, requires_grad=False))
        })

    def get_zeros_like_params(self):
        return nn.ParameterDict({
            "bv": nn.Parameter(torch.zeros_like(self.theta["bv"], requires_grad=False)),
            "W": nn.Parameter(torch.zeros_like(self.theta["W"], requires_grad=False)),
            "bh": nn.Parameter(torch.zeros_like(self.theta["bh"], requires_grad=False)),
            "sigma": nn.Parameter(torch.zeros_like(self.theta["sigma"], requires_grad=False))
        })

    def init_adam(self):
        self.dtheta = self.get_zeros_like_params()
        self.first_moments = self.get_zeros_like_params()
        self.second_moments = self.get_zeros_like_params()

    def Ph_v(self, v):
        P = torch.sigmoid(self.theta["bh"] + (v/self.theta["sigma"]) @ self.theta["W"])
        P_tilde = tdist.Bernoulli(P).sample()
        return P, P_tilde

    def Pv_h(self, h):
        mu = self.theta["bv"] + self.theta["sigma"] * (h @ self.theta["W"].t())
        tilde = tdist.Normal(mu, self.theta["sigma"]).sample()
        return tilde

    def CD(self, v0, epoch, eta, K=1):
        batch_size = v0.size(0)

        # Positive phase
        v_orig = v0
        ph_orig, h_orig = self.Ph_v(v_orig)

        pos_delta_bv = torch.sum((v_orig - self.theta["bv"]) / torch.pow(self.theta["sigma"], 2), axis=0)
        pos_delta_W = (v_orig / self.theta["sigma"]).t() @ ph_orig
        pos_delta_bh = torch.sum(ph_orig, axis=0)
        pos_delta_sigma = torch.sum(torch.pow(v_orig - self.theta["bv"], 2) \
                                    / torch.pow(self.theta["sigma"], 3) \
                                    - (ph_orig @ self.theta["W"].t()) \
                                    * (v_orig / torch.pow(self.theta["sigma"], 2)), axis=0)

        if K > 1:
            h_K = h_orig

            for _ in range(K-1):
                pv_K = self.Pv_h(h_K)
                ph_K, h_K = self.Ph_v(pv_K)

            pv_K = self.Pv_h(ph_K)
            ph_K, h_orig = self.Ph_v(pv_K)

        # Negative phase
        pv_recon = self.Pv_h(h_orig)
        ph_recon, h_recon = self.Ph_v(pv_recon)

        neg_delta_bv = torch.sum((pv_recon - self.theta["bv"]) / torch.pow(self.theta["sigma"], 2), axis=0)
        neg_delta_W = (pv_recon / self.theta["sigma"]).t() @ ph_recon
        neg_delta_bh = torch.sum(ph_recon, axis=0)
        neg_delta_sigma = torch.sum(torch.pow(pv_recon - self.theta["bv"], 2) \
                                    / torch.pow(self.theta["sigma"], 3) \
                                    - (ph_recon @ self.theta["W"].t()) \
                                    * (pv_recon / torch.pow(self.theta["sigma"], 2)), axis=0)
        
        # Gradients
        self.dtheta["bv"] = nn.Parameter((pos_delta_bv - neg_delta_bv) / batch_size)
        self.dtheta["W"] = nn.Parameter((pos_delta_W - neg_delta_W) / batch_size)
        self.dtheta["bh"] = nn.Parameter((pos_delta_bh - neg_delta_bh) / batch_size)
        self.dtheta["sigma"] = nn.Parameter((pos_delta_sigma - neg_delta_sigma) / batch_size)
        
        # Optimization
        self.adam(epoch+1, eta)

        # Reconstruction error
        return torch.dist(v_orig, pv_recon, 2)

    def adam(self, t, eta):
        for param in self.theta:
            self.first_moments[param] = nn.Parameter(self.beta_1 * self.first_moments[param] \
                                        + (1-self.beta_1) * self.dtheta[param])
            self.second_moments[param] = nn.Parameter(self.beta_2 * self.second_moments[param] \
                                        + (1-self.beta_2) * torch.pow(self.dtheta[param], 2))
            first_corr = self.first_moments[param] / (1 - self.beta_1**t)
            second_corr = self.second_moments[param] / (1 - self.beta_2**t)
            self.theta[param] = nn.Parameter(self.theta[param] + eta * first_corr / (torch.sqrt(second_corr) + self.eps))

    def train_RBM(self, num_epochs, data, eta, K, chkpt):
        self.train()
        l2 = []
        print("\n Training RBM:")
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            l2.append(self.CD(data, epoch, eta, K))
            pbar.update(1)
        torch.save(self.state_dict(), chkpt)
        pbar.close()
        print(f"\n{chkpt} created!")

        return l2