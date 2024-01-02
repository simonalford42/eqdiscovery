import locusts
from train_locusts import Transformer, IN_DIM
import train_locusts
import glob
import torch
from utils import assert_equal
from einops import rearrange
import numpy as np

def predict(model, init_pos, info, t=1):
    '''
    input:
        init_pos: [T, N, D]  - can be multiple timesteps because we need at least two timesteps to calculate initial velocity, and might as well handle more than that too
    returns:
        x: [T+t, N, D]
        v: [T+t-1, N, D]
        a: [T+t-2, N, D]
    '''
    T, N, D = init_pos.shape
    assert T >= 2, 'need at least two time steps to calculate initial velocity'

    model = model.cuda()

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
    x, v, a = x.cuda(), v.cuda(), a.cuda()
    x, v, a = list(x), list(v), list(a)

    for i in range(t):
        inp = embed_timestep(x[-1]).cuda()
        inp = rearrange(inp, 'n d -> () n d')
        inp = inp
        pred = model(inp)[0]
        new_a = train_locusts.get_acceleration(pred)
        new_v = v[-1] + locusts.DT * new_a
        new_x = x[-1] + locusts.DT * new_v

        a.append(new_a) # a_i
        v.append(new_v) # v_i+1 = v_i + a_i * dt
        x.append(new_x) # x_i+2 = x_i+1 + v_i+1 * dt

    return torch.stack(x).cpu(), torch.stack(v).cpu(), torch.stack(a).cpu()


def get_ckpt_path(run_path):
    run_path = run_path[len('simonalford42/'):]
    ckpts = glob.glob(run_path + '/checkpoints/*.ckpt')
    return ckpts[0]


def copy_ckpt(path):
    # move file at path to ~/to_copy/ by doing mv path ~/to_copy/
    import subprocess
    subprocess.run(['mv', path, '~/to_copy/model.ckpt'])



path = get_ckpt_path('simonalford42/locust_transformer/p01izwoi')
copy_ckpt(path)
assert False

model = Transformer.load_from_checkpoint(path)
# validation
# data, info = locusts.import_data('05UE20200708')
# training
data, info = locusts.import_data('05EQ20200207')

data = torch.tensor(data, dtype=torch.float32)
# data: [T, N, D]

for start in range(100, 1000, 100):
    x, v, a = predict(model, data[start-100:start], info, t=100)
    # first show the actual data
    assert_equal(x.shape, data[start-100:start+100].shape)
    locusts.visualize_data(data[start-100:start+100].detach().numpy(), info)
    input('press enter to continue')
    locusts.visualize_data(x.detach().numpy(), info)
    input('press enter to continue')

# x: [T+t, N, D]

# check that embed_timestep returns the same thing as process_data

