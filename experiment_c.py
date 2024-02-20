import os
from torchvision import datasets
import torch
import torchvision.transforms as trn
from torchdistill.common import yaml_util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import MetricLogger

from sc2bench.analysis import check_if_analyzable
from sc2bench.models.backbone import check_if_updatable
from utils import load_model
import argparse

def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-md', '--model', type=str, metavar='', default='rn50',
                        help = 'specify the dnn classifier (rn50, rn101, rg64)')
    parser.add_argument('-p', '--prior', type=str, metavar='',default='mshp',
                        help = 'specify the prior model (fp, mshp)')
    parser.add_argument('-b', '--beta', type=float, metavar='', default=0.08,
                        help = 'specify the rate-distortion trade-off factor of the dnn model (0.08, 0.16, 0.32, 0.64, 1.28, 2.56)')
    parser.add_argument('-c', '--corruption', type=str, metavar='',default='defocus_blur',
                        help = 'specify the corruption type (defocus_blur, glass_blur, motion_blur, zoom_blur, contrast, elastic_transform, jpeg_compression, pixelate, gaussian_blur, saturate, spatter, speckle, gaussian_noise, impulse_noise, shot_noise, brightness, fog, frost, snow)')
    parser.add_argument('-s', '--severity', type=int, metavar='', default='1',
                        help = 'specify the severity of corruptions')
    parser.add_argument('-d', '--device', type=int, metavar='', default=0,
                        help='specify the gpu device, -1 means cpu')
    
    return parser.parse_args()

@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device,
             log_freq=1000, title=None, header='Test:'):
    model = model_wo_ddp.to(device)
    if hasattr(model, 'use_cpu4compression'):
        model.use_cpu4compression()

    model.eval()
    analyzable = check_if_analyzable(model_wo_ddp)
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)

        if isinstance(target, torch.Tensor):
            target = target.to(device, non_blocking=True)

        output = model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = len(image)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg

    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return metric_logger.acc1.global_avg

def run_experiment():
    args = arguments_parser()
    assert args.corruption in ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
                               'contrast', 'elastic_transform', 'jpeg_compression', 'pixelate', 
                               'gaussian_blur', 'saturate', 'spatter', 'speckle', 
                               'gaussian_noise', 'impulse_noise', 'shot_noise', 
                               'brightness', 'fog', 'frost', 'snow']
    # set up dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = trn.Compose(
        [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
    if args.severity == 0:
        dir = "../dataset/ilsvrc2012/val/"
    else:
        if args.corruption in ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']:
            dir = '../dataset/imagenet-c/blur/%s/%d'%(args.corruption,args.severity)
        elif args.corruption in ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate']:
            dir = '../dataset/imagenet-c/digital/%s/%d'%(args.corruption,args.severity)
        elif args.corruption in ['gaussian_blur', 'saturate', 'spatter', 'speckle']:
            dir = '../dataset/imagenet-c/extra/%s/%d'%(args.corruption,args.severity)
        elif args.corruption in ['gaussian_noise', 'impulse_noise', 'shot_noise']:
            dir = '../dataset/imagenet-c/noise/%s/%d'%(args.corruption,args.severity)
        else:
            dir = '../dataset/imagenet-c/weather/%s/%d'%(args.corruption,args.severity)
    test_dataset = datasets.ImageFolder(root=dir, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True)
    
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

    acc = evaluate(student_model, test_data_loader, device,
                    log_freq=1000, title='[Student: {}]'.format(student_model_config['name']))
    print("%s-%s-beta%.2f - Corruption:%s, Acc:%f"%(args.model,args.prior,args.beta,args.corruption,acc))

if __name__ == '__main__':
    run_experiment()  