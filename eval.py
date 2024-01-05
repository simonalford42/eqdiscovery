import locusts
from train_locusts import Transformer, IN_DIM
import train_locusts
import glob
import torch
from utils import assert_equal
from einops import rearrange
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, init_pos, info, t=1, model_predicts_acceleration=False):
    '''
    input:
        init_pos: [T, N, D]  - can be multiple timesteps because we need at least two timesteps to calculate initial velocity, and might as well handle more than that too
        model_predicts_acceleration: if True, model predicts acceleration, otherwise model predicts position
    returns:
        x: [T+t, N, D]
        v: [T+t-1, N, D]
        a: [T+t-2, N, D]
    '''
    T, N, D = init_pos.shape
    assert T >= 2, 'need at least two time steps to calculate initial velocity'

    model = model.to(DEVICE)

    # should match process_data in train_locusts.py
    def embed_timestep(arr):
        N, D = arr.shape
        data = np.zeros((N+2, D+3))
        data[2:N+2, :D] = arr.detach().cpu().numpy()
        data[0, :D] = info['posA']
        data[1, :D] = info['posB']
        data[0:, D] = 1
        data[1, D+1] = info['isA_HQ']
        data[1, D+1] = info['isB_HQ']
        data[0, D+2] = info['radA']
        data[1, D+2] = info['radB']
        assert_equal(data.shape, (N+2, IN_DIM))
        return torch.tensor(data, dtype=torch.float32)

    x = init_pos
    v = (1/locusts.DT) * (x[1:] - x[:-1])
    a = (1/locusts.DT) * (v[1:] - v[:-1])
    x, v, a = x.to(DEVICE), v.to(DEVICE), a.to(DEVICE)
    x, v, a = list(x), list(v), list(a)

    for i in range(t):
        inp = embed_timestep(x[-1]).to(DEVICE)
        inp = rearrange(inp, 'n d -> () n d')
        inp = inp
        pred = model(inp)[0]
        new_a = train_locusts.get_acceleration(pred)
        new_v = v[-1] + locusts.DT * new_a
        new_x = x[-1] + locusts.DT * new_v
        if model_predicts_acceleration:
            new_a = train_locusts.get_acceleration(pred)
            acc_scale = 1e4
            new_a = new_a / acc_scale
            new_v = v[-1] + locusts.DT * new_a
            new_x = x[-1] + locusts.DT * new_v
            a.append(new_a) # a_i
            v.append(new_v) # v_i+1 = v_i + a_i * dt
        else:
            new_x = pred

        x.append(new_x) # x_i+2 = x_i+1 + v_i+1 * dt

    return torch.stack(x).cpu()


def get_ckpt_path(run_path):
    run_path = run_path[len('simonalford42/'):]
    ckpts = glob.glob(run_path + '/checkpoints/*.ckpt')
    return ckpts[0]


def copy_ckpt(path):
    # move file at path to ~/to_copy/ by doing mv path ~/to_copy/
    import subprocess
    subprocess.run(['mv', path, '~/to_copy/model.ckpt'])



# path = get_ckpt_path('simonalford42/locust_transformer/p01izwoi')
# copy_ckpt(path)
# assert False

path = 'model.ckpt'
model = Transformer.load_from_checkpoint(path)
# validation
# data, info = locusts.import_data('05UE20200708')
# training
data, info = locusts.import_data('05EQ20200207')

data = torch.tensor(data, dtype=torch.float32)
# data: [T, N, D]

data = data[:10000]
x, v, a = predict(model, data[:100], info, t=10000)
print('actual data')
locusts.visualize_data(data.detach().numpy(), info, speedup=25)
print('predictions')
locusts.visualize_data(x.detach().numpy(), info, speedup=25)

# x: [T+t, N, D]

# check that embed_timestep returns the same thing as process_data

