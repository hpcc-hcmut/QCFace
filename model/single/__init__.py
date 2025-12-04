import torch
import torch.nn as nn

from model import iresnet
from model.single import (qcface, qcface_curricular, qcface_mv)



class FaceModel(nn.Module):
    def __init__(self, backbone, loss_model, embed_dims, num_classes, phase=None):
        super(FaceModel, self).__init__()
        # build backbone
        if backbone=="ir18":
            self.backbone = iresnet.iresnet18(num_classes=embed_dims)
        elif backbone=="ir50":
            self.backbone = iresnet.iresnet50(num_classes=embed_dims)
        elif backbone=="ir100":
            self.backbone = iresnet.iresnet100(num_classes=embed_dims)
        
        # ni=10572 (CASIA-WebFace) or ni=93431 (MSMV3)
        self.loss_model = loss_model
        if loss_model=="qcface":
            self.head = qcface.QCFace(embed_size=embed_dims, num_classes=num_classes)
            if phase=="norm":
                self.head.change_training_mode(True)
            else:
                self.head.change_training_mode(False)
        
        elif loss_model=="qcface-cur":
            self.head = qcface_curricular.QCFace(embed_size=embed_dims, num_classes=num_classes)
            if phase=="norm":
                self.head.change_training_mode(True)
            else:
                self.head.change_training_mode(False)
        
        elif loss_model=="qcface-mv":
            self.head = qcface_mv.QCFace(embed_size=embed_dims, num_classes=num_classes)
            if phase=="norm":
                self.head.change_training_mode(True)
            else:
                self.head.change_training_mode(False)

    def load_weight(self, path):
        statedict = torch.load(path, map_location="cpu")['state_dict']
        self.load_state_dict(statedict)

    def forward(self, x, target):
        feat = self.backbone(x)
        output, norms, norm_loss, one_hot = self.head(feat, target)
        return output, norms, norm_loss, one_hot



def build_model(args, device):
    model = FaceModel(args.arch, args.loss_model, args.embed_dims, args.num_classes, args.phase)
    model = model.cuda(device)
    return model
