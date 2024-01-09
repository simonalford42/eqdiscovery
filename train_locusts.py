import os
import argparse
import torch
from torch import optim, nn, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from einops import rearrange
from utils import assert_equal
import locusts
import pickle

IN_DIM = 5
OUT_DIM = 2

class RFF(nn.Module):
    '''
    - normalize position data to be in [0, 1]
    - sample random b1, b2 from N(0, 1).
    - calculate y = b1 x + b2 y
    - use cos(2 pi y), sin(2 pi y) as features
    '''
    def __init__(self, in_dim=2, out_dim=10, sigma=1):
        super().__init__()
        assert out_dim % 2 == 0, "n_features must be even"
        self.b = nn.Parameter(torch.randn((out_dim // 2, in_dim)) / sigma, requires_grad=False)

    def forward(self, x):
        '''
        x: [..., D]
        returns: [..., out_dim]
        '''
        y = x @ self.b.T
        embedding = rearrange([torch.cos(2*np.pi*y), torch.sin(2*np.pi*y)], 'two ... f -> ... (two f)')
        return embedding


class Transformer(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        self.in_dim = hparams['in_dim']
        self.d_model = hparams['d_model']
        self.out_dim = hparams['out_dim']
        self.nhead = hparams['nhead']
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"

        # self.embedding = nn.Linear(self.in_dim, self..d_model)
        self.embedding = RFF(self.in_dim, self.d_model, hparams['rff_scale'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.projection = nn.Linear(self.d_model, self.out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.projection(x)
        # return torch.zeros_like(x) * x
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer

def get_train_val_filenames():
    # val_filenames = [
    #     '05UE20200706',
    #     '15EQ20200329',
    #     '05EQ20200525',
    #     '15UE20191206',
    #     '15UE20191213',
    #     '01EQ20200706',
    #     '15EQ20191209',
    #     '10UE20200331',
    #     '10EQ20200831',
    #     '01UE20200917_2',
    #     '01EQ20201007_2',
    #     '05UE20200708',
    #     '01UE20200918',
    #     '05EQ20200303_reencoded',
    #     '30UE20191206',
    #     '10EQ20200822',
    #     '30EQ20191209',
    #     '10UE20200624',
    # ]

    train_filenames = [
        # '30EQ20191203'
        # '05EQ20200813',
        '01EQ20200706',
    ]
    val_filenames = train_filenames

    # train_filenames = []
    # for filename in os.listdir("Locusts/Data/Tracking"):
    #     if filename.endswith(".mat"):
    #         # convert e.g. 15UE20200509_annotation.mat to 15UE20200509
    #         # to do so, get rid of '_annotation.mat' at the end
    #         f = filename[:-len('_annotation.mat')]
    #         if f not in val_filenames:
    #             train_filenames.append(f)


    # def get_n(filename):
        # return int(filename[:2])

    # n = 5
    # remove all filenames that aren't n=n
    # val_filenames = [f for f in val_filenames if get_n(f) == n]
    # train_filenames = [f for f in train_filenames if get_n(f) == n]

    # for f in val_filenames:
        # assert f not in train_filenames, f"{f} is in both val and train"

    return train_filenames, val_filenames

def get_acceleration(prediction):
    '''
    input: [..., N+2, D]
    output: [..., N, D]
    '''
    N = prediction.shape[-2] - 2
    # the first two rows are the food particles
    return prediction[..., 2:N+2, :]

def make_dataloaders(train_filenames, val_filenames):
    print('Loading data...')
    train_data = {}
    val_data = {}
    max_n = 0

    for f in train_filenames:
        arr, info = locusts.import_data(f)
        train_data[f] = arr, info
        max_n = max(max_n, info['n'])

    for f in val_filenames:
        arr, info = locusts.import_data(f)
        val_data[f] = arr, info
        max_n = max(max_n, info['n'])

    # input data is (T, N, D)
    # we add food particles to make N+2
    # we also add (1) a binary is_food indicator,
    # (2) if the food is high quality or not
    # (3) the radius of the food location
    # so the final dim is D + 3

    def process_data(arr, info):
        T, N, D = arr.shape
        assert_equal(D, 2)

        # make data
        data = np.zeros((T, max_n+2, D+3))
        data[:, 2:N+2, :D] = arr
        data[:, 0, :D] = info['posA']
        data[:, 1, :D] = info['posB']
        data[:, 0:, D] = 1
        data[:, 1, D+1] = info['isA_HQ']
        data[:, 1, D+1] = info['isB_HQ']
        data[:, 0, D+2] = info['radA']
        data[:, 1, D+2] = info['radB']

        # make acceleration targets
        # x = arr
        # v = (1/locusts.DT) * (arr[1:] - arr[:-1])
        # a = (1/locusts.DT) * (v[1:] - v[:-1])
        # pad acceleration targets.
        # acc = np.zeros((T-2, max_n+2, D))
        # acc[:, 2:N+2, :] = a
        # data and acc need to be the same length
        # data = data[:-2]
        # acc = torch.tensor(acc, dtype=torch.float32)

        # instead, have the targets be the next timestep's position
        next_pos = np.zeros((T-1, max_n+2, D))
        next_pos[:, :N, :] = arr[1:]
        data = data[:-1]
        next_pos = torch.tensor(next_pos, dtype=torch.float32)

        # T, N, d
        data = torch.tensor(data, dtype=torch.float32)

        assert_equal(data.shape, (T-1, max_n+2, IN_DIM))
        assert_equal(next_pos.shape, (T-1, max_n+2, OUT_DIM))

        # scale the acceleration targets so that the prediction task is in a typical range
        # acc_scale = 1e4
        # acc = acc * acc_scale
        # assert_equal(data.shape, (T-2, max_n+2, IN_DIM))
        # assert_equal(acc.shape, (T-2, max_n+2, OUT_DIM))

        return data, next_pos
        # return data, acc

    # def embed_timestep(arr):
        # N, D = arr.shape
        # data = np.zeros((N+2, D+3))
        # data[2:N+2, :D] = arr
        # data[0, :D] = info['posA']
        # data[1, :D] = info['posB']
        # data[0:, D] = 1
        # data[1, D+1] = info['isA_HQ']
        # data[1, D+1] = info['isB_HQ']
        # data[0, D+2] = info['radA']
        # data[1, D+2] = info['radB']
        # assert_equal(data.shape, (N+2, IN_DIM))
        # return torch.tensor(data, dtype=torch.float32)

    # check that embed_timestep returns the same thing as process_data
    # for (arr, info) in data.values():
        # assert_equal(embed_timestep(arr[0]), process_data(arr, info)[0][0][:info['n']+2])
    # assert False

    def make_dataset(data):
        inputs, targets = zip(*[process_data(arr, info) for (arr, info) in data.values()])
        inputs, targets = torch.stack(inputs), torch.stack(targets)
        inputs = rearrange(inputs, 'N T nl d -> (N T) nl d')
        targets = rearrange(targets, 'N T nl d -> (N T) nl d')
        return TensorDataset(inputs, targets)

    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)

    # using num_workers that pytorch warned me to use
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=7)
    val_dataloader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=7)

    return train_dataloader, val_dataloader


parser = argparse.ArgumentParser()
parser.add_argument("--slurm_id", default=None, type=int)
parser.add_argument("--slurm_name", default=None, type=str)
parser.add_argument("--no_log", action='store_true')
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--note", type=str, default='')
# parser.add_argument("--steps", type=float, default=None)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--rff_scale", type=float, default=1)
# parser.add_argument("--val_check_interval", type=float, default=0.2)
args = parser.parse_args()


if __name__ == '__main__':
    if args.seed is not None:
        seed_everything(args.seed)

    hparams = vars(args)
    hparams['in_dim'] = IN_DIM
    hparams['out_dim'] = OUT_DIM

    train_filenames, val_filenames = get_train_val_filenames()
    train_dataloader, val_dataloader = make_dataloaders(train_filenames, val_filenames)

    # save dataloaders to a pickle for faster loading later
    # with open(f'dataloaders.pkl', 'wb') as f:
        # pickle.dump((train_dataloader, val_dataloader, train_filenames, val_filenames), f)
    # assert False

    # train_dataloader, val_dataloader, train_filenames, val_filenames = pickle.load(open(f'dataloaders_n=5.pkl', 'rb'))

    hparams['train_filenames'] = train_filenames
    hparams['val_filenames'] = val_filenames

    model = Transformer(hparams)

    logger = WandbLogger(project='locust_transformer', mode='disabled' if args.no_log else 'online')
    logger.log_hyperparams(hparams)
    trainer = Trainer(
        # max_steps=args.steps,
        max_epochs=args.epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        # val_check_interval=args.val_check_interval,
    )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
