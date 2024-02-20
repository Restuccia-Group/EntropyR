import os
from torchvision import datasets
import torch
from torchdistill.common import yaml_util
from sc2bench.analysis import check_if_analyzable
from sc2bench.models.backbone import check_if_updatable
from utils import load_model, bpp_loss
from model_wrapper import AuxModule, AdvWrapper, BPWrapper
from pgd import PGD
import torchvision.transforms as trn
import argparse

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-md', '--model', type=str, metavar='', default='rn50',
                        help = 'specify the dnn classifier (rn50, rn101, rg64)')
    parser.add_argument('-b', '--beta', type=float, metavar='', default=0.08,
                        help = 'specify the rate-distortion trade-off factor of the dnn model (0.08, 0.16, 0.32, 0.64, 1.28, 2.56)')
    parser.add_argument('-p', '--prior', type=str, metavar='',default='mshp',
                        help = 'specify the prior model (fp, mshp)')
    parser.add_argument('-atk', '--attack', type=str, metavar='', default='pior',
                        help = 'specify the type of attack (prior, clf, noise)')
    parser.add_argument('-eps', '--epsilon', type=int, metavar='',default=4,
                        help = 'specify the perturbation magnitude (2/255, 4/255, 8/255, 16/255, 32/255)')
    parser.add_argument('-d', '--device', type=int, metavar='', default=0,
                        help='specify the gpu device, -1 means cpu')
    
    return parser.parse_args()

def run_experiment():
    args = arguments_parser()
    assert args.model in ['rn50', 'rn101', 'rg64'], "model is not defined"
    assert args.beta in [0.08, 0.16, 0.32, 0.64, 1.28, 2.56], "beta is not defined"
    assert args.prior in ['fp', 'mshp'], "prior is not defined"
    assert args.attack in ['prior', 'clf', 'noise'], "attack is not defined"
    # set up model name
    if args.model == 'rn50':
        modelname = 'resnet50'
    elif args.model == 'rn101':
        modelname = 'resnet101'
    else:
        modelname = 'regnety6.4gf'
    # set up device
    cuda_id = torch.cuda.device_count()
    if args.device == -1 or cuda_id == 0:
        device = "cpu"
    else:
        device = "cuda:%d"%args.device if args.device < cuda_id else "cuda:%d"%(cuda_id-1)
    # load pretrained model
    config_str = "configs/ilsvrc2012/supervised_compression/entropic_student/splitable_%s-%s-beta%.2f_from_%s.yaml"%(modelname,args.prior,args.beta,modelname)
    config = yaml_util.load_yaml_file(os.path.expanduser(config_str))
    models_config = config['models']
    student_model_config =\
            models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config.get('ckpt', None)
    student_model = load_model(student_model_config, device, False)
    if check_if_updatable(student_model):
        student_model.update()
    if check_if_analyzable(student_model):
        student_model.activate_analysis()
    student_model.to(device)
    if hasattr(student_model, 'use_cpu4compression'):
        student_model.use_cpu4compression()
    student_model.eval()

    # set up adversarial attack
    advmodel = AdvWrapper(student_model)
    if args.attack == 'prior':
        atk = PGD(bpp_loss,eps=args.epsilon)
        aux_module = AuxModule(student_model,mode=args.prior)
    elif args.attack == 'clf':
        atk = PGD(eps=args.epsilon)
        if 'resnet' in modelname:
            clf_model = BPWrapper(student_model,md_type='resnet',mode=args.prior)
        else:
            clf_model = BPWrapper(student_model,md_type='regnet',mode=args.prior)

    # set up dataset
    test_transform = trn.Compose(
        [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor()])
    test_dataset = datasets.ImageFolder(root="../dataset/ilsvrc2012/val/", transform=test_transform)

    # run experiment
    acc = 0
    i = 0
    student_model.clear_analysis()
    for x,y in test_dataset:
        i += 1
        x = x[None].to(device)
        y = torch.tensor(y)[None].to(device)
        # clean data
        if args.epsilon == 0.:
            output = advmodel(x).reshape(-1)
        # perturbed data
        else:
            if args.attack == 'prior':
                xadv = atk(aux_module,x)
            elif args.attack == 'clf':
                xadv = atk(clf_model,x,y)
            else:
                noise = torch.randn_like(x).sign()*args.epsilon/255
                xadv = torch.clamp(x+noise,0,1)
            output = advmodel(xadv).reshape(-1)

        acc += (torch.argmax(output) == y)
        print("%s-%s-beta%.2f - Samples:%d, Acc with atk:%f"%(args.model,args.prior,args.beta,i,acc/i),end='\r')
    print("%s-%s-beta%.2f - Samples:%d, Acc with atk:%f"%(args.model,args.prior,args.beta,i,acc/i))
    student_model.summarize()

if __name__ == '__main__':
    run_experiment()  