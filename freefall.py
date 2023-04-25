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

class FallData(IterableDataset):
    def __init__(self, g=0.5, mu=0.05, use_cuda=False):
        self.g, self.mu = g, mu
        super().__init__()

    def __iter__(self):

        for _ in range(10000):
            x = torch.randn(2)
            v = torch.randn(2)
            g = torch.Tensor([0, -self.g])
            mu = -self.mu

            trajectory = []
            for _ in range(10):
                trajectory.append(torch.concat([x, v], 0))
                norm = (v*v).sum(-1)**0.5
                vh = v/norm
                x, v = x + v, v + g + mu*(v*v).sum(-1)*vh

            yield torch.stack(trajectory)



loader = DataLoader(
    FallData(),
    batch_size=256,
    #drop_last=True,
    num_workers=1,
    #persistent_workers=True,
    pin_memory=False
)



class RegressionModule(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.steps = 0
        self.t1 = 1000
        self.t2 = 20000
        self.model = EquationNetwork(4, 4, 1, 
                                     {"id": (10,lambda x: x),
                                      #"*": (2,lambda x, y: x*y),
                                      "+": (10,lambda x, y: x+y),
                                      #"-": (2,lambda x, y: x-y)
                                     })
        

    
    def training_step(self, batch, batch_idx):
        # schedules for regularization and parameter clamping
        self.steps+=1
        _lambda = 0.001 if self.t1 < self.steps < self.t2 else 0.0
        if self.steps >= self.t2: self.model.clamp_parameters()                

        x, y = batch[:, 3, :], batch[:, 4, :]
        y_hat = self.model(x)

        if batch_idx%1000 == 0:
            print("\n", self.steps, self.model.extract(["x", "y", "vx", "vy"]))
        
        # could think of scaling too.
        loss = F.mse_loss(y_hat, y)
        #print(x, y, y_hat, loss)
        regularization = self.model.l1()
        self.log('mse_train_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('l1', regularization.item(), on_epoch=True, on_step=False)
        self.log('lambda', _lambda, on_epoch=True, on_step=False)
        return (loss+regularization*_lambda)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x): return self.model(x)

class VectorModule(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.steps = 0
        self.t1 = 100
        self.t2 = 200000
        self.model = VectorNetwork({2: 2}, {2: 1},
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
        

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.steps >= self.t2: self.model.clamp_parameters()
        
    
    def training_step(self, batch, batch_idx):
        # schedules for regularization and parameter clamping
        self.steps+=1
        _lambda = 0.001 if self.t1 < self.steps < self.t2 else 0.0
        

        t = random.choice(range(9))
        impulse = batch[:, t+1, 2:]-batch[:, t, 2:]
        v = batch[:, t, 2:].unsqueeze(1)
        up = torch.Tensor([[0,1.]]*batch.shape[0]).unsqueeze(1)
        x = torch.cat([v, up], 1)
        y_hat = self.model({2:x})[2].squeeze(1)
        

        if batch_idx%1000 == 0:
            print("\n", self.steps, self.model.extract({2:["V", "UP"]}))
        
        # could think of scaling too.
        loss = F.mse_loss(y_hat, impulse)
        #print(x, y, y_hat, loss)
        regularization = self.model.l1()
        self.log('mse_train_loss', loss.item(), on_epoch=True, on_step=False)
        self.log('l1', regularization.item(), on_epoch=True, on_step=False)
        self.log('lambda', _lambda, on_epoch=True, on_step=False)
        return (loss+regularization*_lambda)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x): return self.model(x)



def train_regression():
    model = RegressionModule()
    logger = TensorBoardLogger("lightning_logs", name="eql_regression")
    #early_stop_callback = EarlyStopping(monitor="train_loss", patience=50, verbose=False, mode="min")
    trainer = pl.Trainer(gpus=0, max_epochs=10000,
                                gradient_clip_val=0.5,
                                callbacks=[],
                                logger=logger)
                                #accumulate_grad_batches=4)

    trainer.fit(model, loader)

def train_vector():
    model = VectorModule()
    logger = TensorBoardLogger("lightning_logs", name="eql_vector")
    #early_stop_callback = EarlyStopping(monitor="train_loss", patience=50, verbose=False, mode="min")
    trainer = pl.Trainer(gpus=0, max_epochs=10000,
                                gradient_clip_val=0.5,
                                callbacks=[],
                                logger=logger)
                                #accumulate_grad_batches=4)

    trainer.fit(model, loader)

train_vector()
