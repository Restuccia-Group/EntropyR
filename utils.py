import torch
from sc2bench.models.registry import load_classification_model
from sc2bench.models.wrapper import get_wrapped_classification_model
from torchmetrics.image import TotalVariation
from torchmetrics.functional.image import image_gradients

def load_model(model_config, device, distributed):
    if 'classification_model' not in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)

def forward(model,x,enable_backprop=True,mean=None,std=None,mode='mshp'):
    if mean is not None and std is not None:
        x = (x-torch.tensor(mean).reshape(1,3,1,1).to(x.device) )/torch.tensor(std).reshape(1,3,1,1).to(x.device)
    aux_module = model.get_aux_module()
    if mode == 'mshp':
        y = aux_module.g_a(x)
        z = aux_module.h_a(y)
        z_hat,z_likelihood = aux_module.entropy_bottleneck(z,training=enable_backprop)
        gaussian_params = aux_module.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihood = aux_module.gaussian_conditional(y,scales_hat,means_hat)
    else:
        y = aux_module.encoder(x)
        y_hat,y_likelihood = aux_module.entropy_bottleneck(y,training=True)
    return y, y_likelihood

def bpp_loss(x,mask=None):
    if mask is not None:
        return (-x.log2()*mask).mean()
    return -x.log2().mean()

def patch_variation(x,p=4):
    b,_,w,h = x.shape
    assert w % p == 0 and h % p == 0
    tv = TotalVariation('none').to(x.device)
    patch = x.reshape(b,3,w//p,p,h//p,p).permute(0,2,4,1,3,5).reshape(-1,3,p,p)
    p_v = tv(patch).reshape(b,w//p,h//p)
    if b == 1:
        p_v = p_v.reshape(w//p,h//p)
    return p_v

def denoise(x,iterations,reg,stepsize,mask):
    x_new = x
    for i in range(iterations):
        g_x,g_y = image_gradients(x_new)     # in the library the image gradient g_x is implemented as x_i - x_i+1
        update = (x - x_new) + reg*(g_x+g_y) # but in the reference the image gradient g_x is defined as x_i+1 - x_i 
        x_new += stepsize*update*mask        # thus we inverse the operation here
    return x_new