# EntropyR
Official implementation of "Resilience of Entropy Model in Distributed Neural Networks"

## Datasets
Experiments are based on [ImageNet-validation](https://www.image-net.org/) and [ImageNet-C](https://github.com/hendrycks/robustness).

To process ImageNet-validation:

```
mkdir ~/dataset/ilsvrc2013/val
cd ~/dataset/ilsvrc2013/val
wget ${imagenet-val_url}
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
tar -xvf ${imagenet-val_filename}
sh valprep.sh
```

To process ImageNet-C:

```
cd ~/dataset
mkdir imagenet-c
cd imagenet-c
wget ${imagenet-c_url}
tar -xvf ${imagenet-c_filename}
```

## Environment

```
conda env create -n entropy-r --file entropy-r.yml
conda activate entropy-r
```

## Models

We use models from the work [Entropic Student](https://github.com/yoshitomo-matsubara/supervised-compression). 

- Classifier Model
    - ResNet50 (rn50)
    - ResNet101 (rn101)
    - RegNet6.4 (rg64)

- Prior Model
    - Factorized Prior (fp)
    - Mean Square Hyper Prior (mshp)

## Resilience to common corruptions

```
usage: experiment_c.py [-h] [-md] [-p] [-b] [-c] [-s] [-d]

optional arguments:
  -h, --help          show this help message and exit
  -md , --model       specify the dnn classifier (rn50, rn101, rg64) (default:
                      rn50)
  -p , --prior        specify the prior model (fp, mshp) (default: mshp)
  -b , --beta         specify the rate-distortion trade-off factor of the dnn model
                      (0.08, 0.16, 0.32, 0.64, 1.28, 2.56) (default: 0.08)
  -c , --corruption   specify the corruption type (defocus_blur, glass_blur,
                      motion_blur, zoom_blur, contrast, elastic_transform,
                      jpeg_compression, pixelate, gaussian_blur, saturate, spatter,
                      speckle, gaussian_noise, impulse_noise, shot_noise,
                      brightness, fog, frost, snow) (default: defocus_blur)
  -s , --severity     specify the severity of corruptions (default: 1)
  -d , --device       specify the gpu device, -1 means cpu (default: 0)
```

## Resilience to adversarial samples

```
usage: experiment_adv.py [-h] [-md] [-b] [-p] [-atk] [-eps] [-d]

optional arguments:
  -h, --help         show this help message and exit
  -md , --model      specify the dnn classifier (rn50, rn101, rg64) (default: rn50)
  -b , --beta        specify the rate-distortion trade-off factor of the dnn model
                     (0.08, 0.16, 0.32, 0.64, 1.28, 2.56) (default: 0.08)
  -p , --prior       specify the prior model (fp, mshp) (default: mshp)
  -atk , --attack    specify the type of attack (prior, clf, noise) (default: pior)
  -eps , --epsilon   specify the perturbation magnitude (2/255, 4/255, 8/255,
                     16/255, 32/255) (default: 4)
  -d , --device      specify the gpu device, -1 means cpu (default: 0)
```

## Defense

```
usage: defense.py [-h] [-m] [-atk] [-eps] [-d]

optional arguments:
  -h, --help         show this help message and exit
  -m , --mode        specify the mode (attack, defense) (default: defense)
  -atk , --attack    specify the attack type (pgd-e, freq, region) (default: pgd-e)
  -eps , --epsilon   specify the perturbation (2/255, 4/255, 8/255, 16/255)
                     (default: 4)
  -d , --device      specify the gpu device, -1 means cpu (default: 0)
```
