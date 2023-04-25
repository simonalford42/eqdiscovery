import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torchdiffeq import odeint, odeint_adjoint

from equation import *


class Body(nn.Module):
    def __init__(self):
        super().__init__()

        self.acceleration = VectorNetwork({2: 2}, {2: 1},
                                   {"dot-product": (3, (2,2,1),
                                                    lambda x,y: (x*y).sum(-1).unsqueeze(-1)),
                                    "scale-vector": (3, (1,2,2),
                                                     lambda a,v: a*v),
                                    #"add-vector": (3, (2,2,2), lambda x,y: x+y),
                                    # "hat": (3, (2,2),
                                    #                lambda x: x/(((x*x).sum(-1).unsqueeze(-1)**0.5+1e-5))),
                                    "len": (3, (2,1),
                                            lambda x: (x*x).sum(-1).unsqueeze(-1)**0.5),
                                    "id1": (5,(1,1), lambda x: x),
                                    "id2": (5,(2,2), lambda x: x),
                                    #"+": (3, (1,1,1), lambda x,y: x+y),
                                    "*": (2, (1,1,1), lambda x,y: x*y), 
                                    },
                                   2)

    def forward(self, t, state):
        x, v = state
        dx = v

        up = torch.Tensor([[0,1.]]*x.shape[0])
        dv = self.acceleration({2: torch.cat([v.unsqueeze(1), up.unsqueeze(1)],1)})[2].squeeze(1)

        return (dx, dv)

    def integrate(self, x0, v0, dt):

        return odeint_adjoint(self, (x0, v0), dt)

class BodyIntegrator(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.steps = 0
        self.t1 = 500
        self.t2 = 20000
        self.model = Body()
        
    def training_step(self, batch, batch_idx):
        # schedules for regularization and parameter clamping
        self.steps+=1
        _lambda = 0.003 if self.t1 < self.steps < self.t2 else 0.0
        if self.steps >= self.t2: self.model.acceleration.clamp_parameters()

        x0, v0, x, v = batch
        x, x0 = x.squeeze(0), x0.squeeze(0)
        v, v0 = v.squeeze(0), v0.squeeze(0)

        xh, vh = self.model.integrate(x0, v0, torch.Tensor(DragData.TIMES))
        xh, vh = xh.transpose(0,1), vh.transpose(0,1)

        if batch_idx%100 == 0:
            print("\n", self.steps, self.model.acceleration.extract({2:["V", "UP"]}))
        
        # could think of scaling too.
        loss = F.mse_loss(xh, x) + F.mse_loss(vh, v)
        #print(x, y, y_hat, loss)
        regularization = self.model.acceleration.l1()
        self.log('mse_train_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('l1', regularization.item(), on_epoch=True, on_step=False)
        self.log('lambda', _lambda, on_epoch=True, on_step=False)
        return (loss+regularization*_lambda)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x): return self.model(x)
    
class DragData(IterableDataset):
    TIMES = [1, 10]
    def __init__(self, g=0.5, mu=0.05, use_cuda=False):
        self.g, self.mu = g, mu
        super().__init__()

    def __iter__(self):
        
        def f(t, s):
            g = torch.Tensor([0, -self.g])
            mu = -self.mu
            
            x,v = s
            
            norm = (v*v).sum(-1)**0.5
            vh = v/norm.unsqueeze(-1)
            
            dv = g.unsqueeze(0)+mu*(v**2).sum(-1).unsqueeze(-1)*vh

            return v, dv
            
        
        for _ in range(1):
            x = torch.randn(16, 2)
            v = torch.randn(16, 2)
            xs, vs = odeint(f, (x,v), torch.Tensor(DragData.TIMES))
            yield x, v, xs.transpose(0,1), vs.transpose(0,1)

def train():
    loader = DataLoader(
        DragData(),
        batch_size=1,
    #drop_last=True,
    num_workers=12,
    #persistent_workers=True,
    pin_memory=False
    )
    model = BodyIntegrator()
    logger = TensorBoardLogger("lightning_logs", name="eql_ode")
    #early_stop_callback = EarlyStopping(monitor="train_loss", patience=50, verbose=False, mode="min")
    trainer = pl.Trainer(gpus=0, max_epochs=10000,
                                gradient_clip_val=0.5,
                                callbacks=[],
                                logger=logger)
                                #accumulate_grad_batches=4)

    trainer.fit(model, loader)

train()
