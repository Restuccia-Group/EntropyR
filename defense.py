import os
from torchvision import datasets
import torch
from torch.nn.functional import interpolate
from torchdistill.common import yaml_util

from sc2bench.analysis import check_if_analyzable
from sc2bench.models.backbone import check_if_updatable

from utils import load_model,  bpp_loss, denoise, patch_variation, forward
from model_wrapper import AuxModule,BPWrapper
from pgd import PGD
from adaptive import RegionAtk, LowFreqAtk
import torchvision.transforms as trn
from torchmetrics.functional.image import image_gradients
from matplotlib import pyplot as plt
import numpy as np
from model_wrapper import AdvWrapper
import argparse

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', type=str, metavar='', default='defense',
                        help = 'specify the mode (attack, defense)')
    parser.add_argument('-atk', '--attack', type=str, metavar='',default='pgd-e',
                        help = 'specify the attack type (pgd-e, freq, region)')
    parser.add_argument('-eps', '--epsilon', type=int, metavar='', default=4,
                        help = 'specify the perturbation (2/255, 4/255, 8/255, 16/255)')
    parser.add_argument('-d', '--device', type=int, metavar='', default=0,
                        help ='specify the gpu device, -1 means cpu')
    
    return parser.parse_args()

def run_experiment():
    args = arguments_parser()
    # set up model
    config = yaml_util.load_yaml_file(os.path.expanduser("configs/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-mshp-beta0.08_from_resnet50.yaml"))

    models_config = config['models']
    student_model_config =\
            models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config.get('ckpt', None)
    student_model = load_model(student_model_config, "cuda", False)

    if check_if_updatable(student_model):
        student_model.update()

    if check_if_analyzable(student_model):
        student_model.activate_analysis()

    student_model.to("cuda")
    if hasattr(student_model, 'use_cpu4compression'):
        student_model.use_cpu4compression()
    student_model.eval()
    student_model.clear_analysis()
    aux_module = AuxModule(student_model,mode='mshp')
    advmodel = AdvWrapper(student_model)
    # set up atk
    if args.attack == 'pgd-e':
        atk = PGD(bpp_loss,eps=args.epsilon)
    elif args.attack == 'freq':
        atk = LowFreqAtk(bpp_loss,eps=args.epsilon)
    elif args.attack == 'region':
        atk = RegionAtk(bpp_loss,eps=args.epsilon) 
    # set up dataset
    test_transform = trn.Compose(
    [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor()])
    test_dataset = datasets.ImageFolder(root="../dataset/ilsvrc2012/val/", transform=test_transform)

    acc = 0
    i = 0

    for x,y in test_dataset:
        i += 1
        x = x[None].to("cuda")
        y = torch.tensor(y)[None].to("cuda")
        if args.attack == 'region':
            _, y_likelihood = forward(student_model,x,False,mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            mask = 1 - y_likelihood.mean(dim=(0,1),keepdim=True).detach().reshape(55,55)
            xadv = atk(aux_module,x,mask=mask)
        else:
            xadv = atk(aux_module,x)
        
        if args.mode == 'attack':
            output = advmodel(xadv).reshape(-1)
        elif args.mode == 'defense':
            _, y_likelihood1 = forward(student_model,xadv,False,mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            mask_expand = interpolate(y_likelihood1.mean(dim=(0,1),keepdim=True),size=(224,224))
            mask_expand = mask_expand.reshape(224,224)
            x_dn = denoise(xadv,100,0.1,0.15,mask_expand)
            output = advmodel(x_dn).reshape(-1)
        
        acc += (torch.argmax(output) == y)
        print("Mode: %s, Samples:%d, Acc:%f"%(args.mode,i,acc/i),end='\r')

    print("Mode: %s, Samples:%d, Acc:%f"%(args.mode,i,acc/i))
    student_model.summarize()

if __name__ == '__main__':
    run_experiment()  