import pandas as pd
import numpy as np
import einops
from utils import assert_shape
from matplotlib import pyplot as plt

def import_data(filename, path='Locusts/Data/Tracking/'):
    '''
		1.1. _tracked.csv
			1.1.1. Number of variables	: 5
			1.1.2. Number of rows		: N*45000 frames (N=number of animals)
			1.1.3. Variable list:
				cnt					: row counter (0 - N*45000-1)
				frame				: corresponding video frame (blocks of N)
				pos_x				: animal's normalized x-position in video frame (blocks of N)
				pos_y				: animal's normalized y-position in video frame (blocks of N)
				id					: current animal ID (N unique entries)

    Returns a (45000, N, 2) numpy array of positions
    '''
    data = pd.read_csv(path + filename, sep=',', header=0, index_col=0).to_numpy()
    data = einops.rearrange(data, '(a b) c -> a b c', a=45000)
    data = data[:, :, 1:3]
    return data

def test_data():
    # three dots. first one moves from (0,0) to (1,1) over 100 frames
    # second one moves from (0,0) to (1,0) over 100 frames
    # third one moves from (0,0) to (0,1) over 100 frames
    data = np.zeros((100, 3, 2))
    data[:, 0, 0] = np.linspace(0, 1, 100)
    data[:, 0, 1] = np.linspace(0, 1, 100)
    data[:, 1, 0] = np.linspace(0, 1, 100)
    data[:, 2, 1] = np.linspace(0, 1, 100)
    return data

def visualize_data(data, speedup=1):
    # watch the termites move around in pygame
    # data is a numpy array of shape (T, N, 2)
    W, H = 600, 600
    fps = 25 * speedup  # 25 is the default fps of the data
    max_fps = 25
    # if we take every Kth frame, that's like a speedup of K
    # so then the new fps needed is fps/K
    # take the smallest K such that fps/K <= max_fps
    # i.e. K >= fps/max_fps
    K = int(np.ceil(fps / max_fps))
    fps = fps / K
    data = data[::K]

    minx, maxx = np.min(data[:, :, 0]), np.max(data[:, :, 0])
    miny, maxy = np.min(data[:, :, 1]), np.max(data[:, :, 1])

    def translate(p):
        x, y = p
        return int(W*(x-minx)/(maxx-minx)), int(H*(y-miny)/(maxy-miny))

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    running = True
    t = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        for i in range(data.shape[1]):
            pygame.draw.circle(screen, (255, 255, 255), translate(data[t, i]), 2)
        pygame.display.flip()
        t += 1
        if t >= data.shape[0]:
            running = False
        clock.tick(fps)
        seconds_elapsed = K * t // fps
        hours = int(seconds_elapsed // 3600)
        minutes = int((seconds_elapsed // 60) % 60)
        seconds = int(seconds_elapsed % 60)

        time_str = f'{hours:02d}:{minutes:02d}:{seconds:02d} ({fps*K/25}x)'
        pygame.display.set_caption(time_str)


def load_locusts(filename, T=500, speedup=10):
    data = import_data(filename)
    # two timesteps get chopped off when calculating acceleration and velocity 😳
    T = T + 2
    return convert_to_xvfa_data(data[0:T*speedup:speedup])

def convert_to_xvfa_data(data):
    # data is a numpy array of shape (T, N, 2)
    # we want to convert it to a numpy array of shape (3, T, N, 2)
    # where the first dimension is x, v, a
    # and x is the position, v is the velocity, and a is the acceleration

    # convert data dtype from object to float
    data = data.astype(float)
    x = data
    v = data[1:] - data[:-1]
    a = v[1:] - v[:-1]
    return x[:len(a)], v[:len(a)], None, a


def detect_segments(data):
    smooth_width = 1
    # data is a (T=45000, N, 2) numpy array of positions
    # calculate velocity for each locust
    vel = data[1:] - data[:-1]
    # when the magnitude of the velocity changes a lot, start/stop a segment
    # (T, N, )
    vel_magnitudes = (vel[:, :, 0]**2 + vel[:, :, 1]**2)**0.5
    # smoothed
    v = []
    if smooth_width:
        # convolve
        for i in range(vel_magnitudes.shape[1]):
            v_i = np.convolve(vel_magnitudes[:, i], np.ones(smooth_width)/smooth_width, mode='same')
            v.append(v_i)

    vel_magnitudes = np.stack(v).T
    return vel_magnitudes


if __name__ == '__main__':
    # first num is number of locusts, ue means unequal food sources, eq equal
    # data = import_data('30UE20191206_tracked.csv')
    # data = import_data('05EQ20200615_tracked.csv')
    # data = import_data('15EQ20191204_tracked.csv')
#

    data = import_data('01EQ20191203_tracked.csv')
    data = data[1000:9000]

    # vel_magnitudes = detect_segments(data)
    # for i in range(data.shape[1]):
    #     plt.plot(vel_magnitudes[:,i])
    # plt.show()
    # visualize_data(data, speedup=10)

    # data = test_data()
    visualize_data(data, speedup=32)
