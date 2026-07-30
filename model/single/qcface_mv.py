import math

from sympy import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


def find_optim_val_fnc(u_a=100, l_a=1):
    # f(z, p)
    def _qcfnc(n, p, ua, la, k1):
        return k1 * p * (1/n + n / (ua**2)) + (1 - p) * (1/n + n / (la**2))
    # z*(p)    
    def _optpt(p, ua, la, k1):
        return sqrt(((k1-1)*p + 1) / ((k1 / (ua**2) - 1 / (la**2)) * p + 1 / (la**2)))
    
    # find k
    k = Symbol('k')
    op_k = _optpt(0.5, u_a, l_a, k) - (u_a+l_a)/2
    k_op = float(str(solve(op_k, k)[0]))

    return k_op



class QCFace(nn.Module):
    def __init__(self,
                 embed_size=512,
                 num_classes=1000,
                 scale=64., 
                 ua=100., 
                 la=1,
                 m=0.5):
        super(QCFace, self).__init__()
        self.scale = scale
        self.ua = ua
        self.la = la
        self.k = find_optim_val_fnc(self.ua, self.la)
        
        self.m = m 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi-m)
        self.mm = math.sin(math.pi-m)*m

        self.norm_training_flag = False

        self.eps = 1e-3
        self.mv_weight = 1.12

        self.weight = Parameter(torch.Tensor(embed_size, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def change_training_mode(self, flag):
        self.norm_training_flag = flag

    def _qcfnc(self, n, p, ua, la, k):
        return k * p * (1/(n+self.eps) + n / (ua**2)) + (1 - p) * (1/(n+self.eps) + n / (la**2))
    # z*(p)    
    def _optpt(self, p, ua, la, k):
        return torch.sqrt(((k-1)*p + 1) / ((k / (ua**2) - 1 / (la**2)) * p + 1 / (la**2) + self.eps))

    def get_proxy(self, labels):
        return self.w[labels].clone().detach().transpose(-2, -1)
    
    def forward(self, embeds, labels=None):
        # Get normalize
        weight_norm = F.normalize(self.weight, dim=0) # [embed_dim, num_class]
        embeds_norm = F.normalize(embeds, dim=1) # [N, embed_dim]
        # Get cos(theta)
        cos_theta = torch.mm(embeds_norm, weight_norm) # [N, num_class]
        cos_theta = cos_theta.clamp(-1, 1) # [N, num_class]
        cos_theta_cache = cos_theta.clone().detach()
        # Convert labels --> one_hot 
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # For identification prediction
        if labels is None:
            return cos_theta

        target_cos_theta = cos_theta[torch.arange(0, labels.size(0)), labels].view(-1, 1) # [N, 1]
        if self.is_am:  # AM
            mask = cos_theta > target_cos_theta - self.m # [N, num_classes]
            target_cos_theta_m = torch.where(target_cos_theta > self.m, target_cos_theta - self.m, target_cos_theta) # [N, 1]
        else:  # arcface
            target_sine = torch.sqrt(1.0 - torch.pow(target_cos_theta, 2)) # [N, 1]
            target_cos_theta_m = target_cos_theta * self.cos_m - target_sine * self.sin_m # [N, 1]
            mask = cos_theta > target_cos_theta_m # [N, embed_dim]
            target_cos_theta_m = torch.where(target_cos_theta > 0.0, target_cos_theta_m, target_cos_theta) # [N, 1]
        
        # Process hard samples.
        hard_samples = cos_theta[mask]
        cos_theta[mask] = self.mv_weight * hard_samples + self.mv_weight - 1.0
        cos_theta.scatter_(1, labels.data.view(-1, 1), target_cos_theta_m)
        
        # Get scaled softmax face
        scaled_cos_theta_m = cos_theta * self.scale
        scaled_cos_theta = cos_theta_cache - self.scale

        # Get norm of feature vectors
        norms = torch.norm(embeds, dim=1, keepdim=True) # [N, 1]

        # For the identification training process
        if not self.norm_training_flag: 
            return [scaled_cos_theta, scaled_cos_theta_m], norms, 0, one_hot
        
        # For the norm training process
        with torch.no_grad():
            # Easy probabilities are the probabilities of GT class
            easy_probs = scaled_cos_theta.softmax(dim=1) # [N, embed_size]
            indices = (list(range(embeds.size(0))), labels.tolist())
            easy_probs = easy_probs[indices].unsqueeze(1) # [N, 1]
            # Get z*
            opt_norm = self._optpt(easy_probs, self.ua, self.la, self.k) # [N, 1]
            # Get Lqc*
            opt_norm_loss = self._qcfnc(opt_norm, easy_probs, self.ua, self.la, self.k) # [N, 1]

        # Get norm
        norm_loss = self._qcfnc(norms, easy_probs, self.ua, self.la, self.k) - opt_norm_loss # [N, 1]
        norm_loss = norm_loss.mean()

        return [scaled_cos_theta, scaled_cos_theta_m], norms, norm_loss, one_hot
