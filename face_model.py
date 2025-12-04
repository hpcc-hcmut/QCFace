from collections import OrderedDict
import torch
from model import iresnet
from model.single import FaceModel, QAFaceModel, TopoFRModel


def load_pretrained_model(weight_path, backbone, loss_model, embed_dims, num_classes, return_head=False):
    # load model and pretrained statedict
    if loss_model=="qaface":
        model = QAFaceModel(backbone, embed_dims, num_classes)
    elif loss_model=="topofr":
        model = TopoFRModel(backbone, loss_model, embed_dims, num_classes)
    else:
        model = FaceModel(backbone, loss_model, embed_dims, num_classes)
    statedict = torch.load(weight_path, map_location="cpu")['state_dict']
    
    # model_statedict = OrderedDict()
    # for k, v in statedict.items():
    #     model_k = k.replace('module.', '')
    #     model_statedict[model_k] = v
    # model.load_state_dict(model_statedict)

    model.load_state_dict(statedict)
    if not return_head:
        model = model.backbone
    model.eval()
    return model

def load_magface_model(weight_path, backbone, embed_dims):
    if backbone=="ir18":
        model = iresnet.iresnet18(num_classes=embed_dims)
    elif backbone=="ir50":
        model = iresnet.iresnet50(num_classes=embed_dims)
    elif backbone=="ir100":
        model = iresnet.iresnet100(num_classes=embed_dims)
    statedict = torch.load(weight_path, map_location="cpu")["state_dict"]
    
    model_statedict = OrderedDict()
    for k, v in statedict.items():
        if k=="module.fc.weight":
            continue
        model_k = k.replace('module.features.', '')
        model_statedict[model_k] = v
    
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def load_facenet_model(weight_path, backbone, embed_dims):
    if backbone=="ir18":
        model = iresnet.iresnet18(num_classes=embed_dims)
    elif backbone=="ir50":
        model = iresnet.iresnet50(num_classes=embed_dims)
    elif backbone=="ir100":
        model = iresnet.iresnet100(num_classes=embed_dims)
    statedict = torch.load(weight_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(statedict)
    model.eval()
    return model
