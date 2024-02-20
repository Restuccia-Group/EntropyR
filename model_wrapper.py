import torch
from torch import nn
import numpy as np

class AuxModule(nn.Module):
    def __init__(self,model,enable_bp=True,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],mode='mshp') -> None:
        super().__init__()
        self.aux_module = model.get_aux_module()
        self.bp = enable_bp
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
        self.mode = mode
    def forward(self,x):
        device = x.device
        x = (x - self.mean.to(device)) / self.std.to(device)
        if self.mode == 'mshp':
            y = self.aux_module.g_a(x)
            z = self.aux_module.h_a(y)
            z_hat,z_likelihood = self.aux_module.entropy_bottleneck(z,training=self.bp)
            gaussian_params = self.aux_module.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihood = self.aux_module.gaussian_conditional(y,scales_hat,means_hat)
        else:
            y = self.aux_module.encoder(x)
            y_hat,y_likelihood = self.aux_module.entropy_bottleneck(y,training=self.bp)
        return y, y_likelihood

class BPWrapper(nn.Module):
    def __init__(self,model,md_type='resnet',mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],mode='mshp') -> None:
        super().__init__()
        self.aux_module = model.get_aux_module()
        self.type = md_type
        if self.type == 'resnet':
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.avgpool = model.avgpool
            self.fc = model.fc
        else:
            self.s2 = model.s2
            self.s3 = model.s3
            self.s4 = model.s4
            self.head = model.head
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
        self.mode = mode
    def forward(self,x):
        device = x.device
        x = (x - self.mean.to(device)) / self.std.to(device)
        if self.mode == 'mshp':
            y = self.aux_module.g_a(x)
            z = self.aux_module.h_a(y)
            z_hat,z_likelihood = self.aux_module.entropy_bottleneck(z,training=True)
            gaussian_params = self.aux_module.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihood = self.aux_module.gaussian_conditional(y,scales_hat,means_hat)
            x = self.aux_module.g_s(y_hat)
        else:
            y = self.aux_module.encoder(x)
            y_hat,y_likelihood = self.aux_module.entropy_bottleneck(y,training=True)
            x = self.aux_module.decoder(y_hat)
        if self.type == 'resnet':
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.avgpool is None:
                return x

            x = self.avgpool(x)
            if self.fc is None:
                return x

            x = torch.flatten(x, 1)
            return self.fc(x)
        else:
            x = self.s2(x)
            x = self.s3(x)
            x = self.s4(x)
            if self.head is None:
                return x
            return self.head(x)
        
class AdvWrapper(nn.Module):
    def __init__(self,model,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)

    def forward(self,x):
        device = x.device
        x = (x - self.mean.to(device)) / self.std.to(device)
        return self.model(x)