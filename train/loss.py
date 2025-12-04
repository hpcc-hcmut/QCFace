import torch
import torch.nn as nn
import torchshard as ts


class DistFaceLoss(nn.Module):
    def __init__(self, loss_model, loss_softmax_calc, loss_g_calc=None):
        super(DistFaceLoss, self).__init__()
        self.loss_model = loss_model
        self.loss_softmax_calc = loss_softmax_calc
        self.loss_g_calc = loss_g_calc if loss_g_calc is not None else lambda x: 0

    def forward(self, input, target, x_norm):
        if len(input)==3:
            cos_theta, cos_theta_m, easy_probs = input # For regularization loss calculation of QCFace
        else:    
            cos_theta, cos_theta_m = input

        # ID loss calculation
        loss_id, one_hot = self.loss_softmax_calc(cos_theta, cos_theta_m, target)

        # Regularization loss calculation
        if len(input)==3:
            loss_g = self.loss_g_calc(x_norm, easy_probs, target)
        else:
            loss_g = 0

        return loss_id, loss_g, one_hot
