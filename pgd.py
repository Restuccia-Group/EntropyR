import torch
import torch.nn as nn
import numpy as np

class PGD():
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),eps=4,n_iter=20,norm='linf',lr=0.1,clip_min=0.,clip_max=1.,n_sample=16) -> None:
        
        self.loss_fn = loss_fn
        self.eps = eps/255
        self.n_iter = n_iter
        self.norm = norm
        self.lr = lr
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.n_sample = n_sample

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def clip(self,xadv,x):
        if self.norm == 'linf':
            if self.clip_min is not None:
                lb = torch.clamp(x-self.eps,min=self.clip_min) # lower bound
            else:
                lb = x-self.eps
            xadv = torch.max(xadv,lb)
            if self.clip_max is not None:
                ub = torch.clamp(x+self.eps,max=self.clip_max) # upper bound
            else:
                ub = x+self.eps
            xadv = torch.min(xadv,ub)
        else: # projection 
            d = np.prod([*x.shape[1:]])
            delta = xadv - x
            batchsize = delta.size(0)
            deltanorm = torch.norm(delta.view(batchsize,-1),p=2,dim=1)
            scale = np.sqrt(d)*self.eps/deltanorm
            scale[deltanorm<=(np.sqrt(d*self.eps))] = 1
            delta = (delta.transpose(0,-1)*scale).transpose(0,-1).contiguous()
            xadv = x + delta
            if self.clip_min is not None and self.clip_max is not None:
                xadv = torch.clamp(xadv,self.clip_min,self.clip_max)

        return xadv.detach()
    
    def normalize(self,x):
        if self.norm == 'linf':
            x = x.sign()
        else:
            batch_size = x.size(0)
            norm = torch.norm(x.view(batch_size, -1), 2, 1)
            x = (x.transpose(0,-1)/norm).transpose(0,-1).contiguous()
        return x

    def adv_gen(self,forward_fn,x,y=None):
        
        xadv = x.clone().detach()
        forward_fn.eval()
        if y is not None and self.n_sample > 1:
            y = y.unsqueeze(1).repeat(1,self.n_sample).reshape(-1)
        for i in range(self.n_iter):
            if self.n_sample > 1:
                xadv = xadv.unsqueeze(1).repeat(1,self.n_sample,1,1,1).reshape(-1,*x.shape[1:])
            xadv.requires_grad = True
            if y is not None:
                output = forward_fn(xadv) # attack the clf
                loss = self.loss_fn(output,y)
            else:
                _, output = forward_fn(xadv) # attack the prior model
                loss = self.loss_fn(output)
                
            forward_fn.zero_grad()
            loss.backward()

            g = self.normalize(xadv.grad.data)
            xadv = xadv + self.lr*g
            if self.n_sample > 1: # expectation of multiple queries
                xadv = xadv.reshape(-1,self.n_sample,*x.shape[1:]).mean(dim=1)
            xadv = self.clip(xadv,x).detach()

        return xadv