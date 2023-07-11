import time
import numpy as np
import torch
import os
import uuid
from collections import namedtuple
import itertools

import matplotlib.pyplot as plt

class Timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            raise ValueError("Timing message should be string function")
        print(f"{message} in {dt:.1f} seconds")


def noisify(data, intensity=0.01):
    x, v, f, a = data
    print(f"std x: {np.std(x):.2f}, std v: {np.std(v):.2f}")
    x2 = x + np.random.normal(scale=np.std(x) * intensity, size=x.shape)
    v2 = v + np.random.normal(scale=np.std(v) * intensity, size=v.shape)
    print(f"std x2: {np.std(x2):.2f}, std v2: {np.std(v2):.2f}")
    return x2, v2, f, a


def assert_equal(a, b):
    if np.ndarray in [type(a), type(b)]:
        assert np.array_equal(a, b), f'a != b: a:{a}, b:{b}'
    elif torch.Tensor in [type(a), type(b)]:
        assert torch.equal(a, b), f'a != b: a:{a}, b:{b}'
    else:
        assert a == b, f'a != b: a:{a}, b:{b}'


def assert_shape(a: torch.Tensor, shape: tuple):
    ''' wherever a has -1 value, shape can be anything '''
    if -1 in shape:
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = a.shape[i]

    assert_equal(a.shape, shape)
