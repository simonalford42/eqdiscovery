import pandas as pd
import numpy as np
import einops
from utils import assert_shape
from matplotlib import pyplot as plt
import scipy.io
from utils import assert_equal
import random

def import_data(filename, path='Locusts/Data/Tracking/', smoothing=0):
    '''
    filename should be something like '15UE20200509'

		1.1. _tracked.csv
			1.1.1. Number of variables	: 5
			1.1.2. Number of rows		: N*45000 frames (N=number of animals)
			1.1.3. Variable list:
				cnt					: row counter (0 - N*45000-1)
				frame				: corresponding video frame (blocks of N)
				pos_x				: animal's normalized x-position in video frame (blocks of N)
				pos_y				: animal's normalized y-position in video frame (blocks of N)
				id					: current animal ID (N unique entries)

    Returns data, info, where data is a (45000, N, 2) numpy array of positions,
    info has the food positions and other info
    '''
    data_path = path + filename + '_tracked.csv'
    data = pd.read_csv(data_path, sep=',', header=0, index_col=0).to_numpy()
    data = einops.rearrange(data, '(a b) c -> a b c', a=45000)
    data = data[:, :, 1:3]
    data = data.astype(float)

    annotation_path = path + filename + '_annotation.mat'
    annotations = scipy.io.loadmat(annotation_path)

    arena_radius_cm = annotations['Arena'][0,0][0][0][0]
    assert_equal(arena_radius_cm, 45)

    p = annotations['Comment']
    p1, p2 = p[0][0][0], p[1][0][0]
    assert p1 in ['PatchA: HQ', 'PatchB: LQ', 'PatchA: LQ', 'PatchB: HQ']
    assert p2 in ['PatchA: HQ', 'PatchB: LQ', 'PatchA: LQ', 'PatchB: HQ']
    assert 'A' in p1 and 'B' in p2
    is_patchA_HQ = 'HQ' in p1
    is_patchB_HQ = 'HQ' in p2

    patchA_pos = annotations['PatchA'][0][0][0][0]
    patchA_rad = annotations['PatchA'][0][0][1][0][0]
    patchB_pos = annotations['PatchB'][0][0][0][0]
    patchB_rad = annotations['PatchB'][0][0][1][0][0]
    assert_equal(type(patchB_pos), type(patchA_pos), np.ndarray)
    assert_equal(len(patchA_pos), len(patchB_pos), 2)
    assert_equal(type(patchA_rad), type(patchB_rad), np.float64)

    is_valid = 1 in annotations['Valid']
    assert is_valid

    info = {
        'posA': patchA_pos,
        'radA': patchA_rad,
        'isA_HQ': is_patchA_HQ,
        'posB': patchB_pos,
        'radB': patchB_rad,
        'isB_HQ': is_patchB_HQ,
    }
    return smooth_data(data, smoothing=smoothing), info


def smooth_data(data, smoothing=0):
    if smoothing:
        # apply gaussian smoothing to the data to remove noise
        from scipy.ndimage import gaussian_filter1d

        # apply filter to each termite separately
        filtered_data = []
        for i in range(data.shape[1]):
            # filter x and y coordinate separately
            filtered_x = gaussian_filter1d(data[:, i, 0], smoothing, axis=0)
            filtered_y = gaussian_filter1d(data[:, i, 1], smoothing, axis=0)
            filtered_data.append(np.stack([filtered_x, filtered_y], axis=1))

        data = np.stack(filtered_data, axis=1)

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

def visualize_data(data, info=None, speedup=1):
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


    maxv = 1
    minv = -1
    assert np.min(data) >= minv, print(np.min(data), minv)
    assert np.max(data) <= maxv, print(np.max(data), maxv)
    # minx, maxx = np.min(data[:, :, 0]), np.max(data[:, :, 0])
    # miny, maxy = np.min(data[:, :, 1]), np.max(data[:, :, 1])

    def translate(p):
        x, y = p
        return int(W*(x-minv)/(maxv-minv)), int(H*(y-minv)/(maxv-minv))

    # circle rendering
    assert W == H
    r = W//2
    start_time = 10
    # go one full circle over the course of the video
    speed = 2 * np.pi / data.shape[0]

    def circle_pos(t):
        if t < start_time:
            return 0, 0

        return (W/2 + r*np.cos((t-start_time)*speed), H/2 - r*np.sin((t-start_time)*speed))

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    running = True
    t = 0

    def get_color(is_HQ):
        if is_HQ:
            return (120, 0, 0)
        else:
            return (40, 0, 0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))

        # draw the food sources
        if info:
            assert W == H
            rA = translate((info['radA'],0))[0] - translate((0, 0))[0]
            rB = translate((info['radB'],0))[0] - translate((0, 0))[0]

            pygame.draw.circle(screen, get_color(info['isA_HQ']), translate(info['posA']), radius=rA)
            pygame.draw.circle(screen, get_color(info['isB_HQ']), translate(info['posB']), radius=rB)

        # draw the dots moving around
        for i in range(data.shape[1]):
            pygame.draw.circle(screen, (255, 255, 255), translate(data[t, i]), 2)

        # draw the dot moving around in a circle too
        # pygame.draw.circle(screen, (255, 255, 255), circle_pos(t), 2)

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


def add_food_particles(data, info):
    # data is a (T, N, 2) numpy array of positions
    # info contains the position of the food patches via info['posA'] and info['posB']
    # return a (T, N+2, 2) array
    data2 = np.zeros((data.shape[0], data.shape[1]+2, 2))
    data2[:, :-2] = data
    data2[:, -2] = info['posA']
    data2[:, -1] = info['posB']
    # data2[:, -2] = (random.random(), random.random())
    # data2[:, -1] = (random.random(), random.random())
    return data2


def load_locusts(filename, speedup=10, start=0, end=None, smoothing=0):
    data, info = import_data(filename, smoothing=smoothing)
    # we lose two time steps when calculating acceleration
    # so that the total number is still even
    data = data[start:end+2*speedup:speedup]
    data = add_food_particles(data, info)
    return convert_to_xvfa_data(data)

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

    T = len(a)
    x = x[:T]
    v = v[:T]
    a = a[:T]

    assert len(x) == len(v) == len(a) == T, 'x, v, a should all have the same length but len(x) = {}, len(v) = {}, len(a) = {}'.format(len(x), len(v), len(a))
    return x, v, None, a


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


def simulate_random_walks(n=5, T=1000, std=0.00001, seed=1):
    np.random.seed(seed)
    def keep_in_bounds(x, v):
        for p in range(n):
            norm = (x[p, 0] ** 2 + x[p, 1]**2)**0.5
            if norm > 1:
                x[p] = x[p] / norm
                v[p] = 0

        return x, v

    x = [np.zeros((n, 2))]
    v = [np.zeros((n, 2))]
    for t in range(T):
        dv = np.random.normal(size=(n, 2), scale=std)
        v.append(v[-1] + dv)
        x.append(x[-1] + v[-1])
        x[-1], v[-1] = keep_in_bounds(x[-1], v[-1])

    return np.array(x)


if __name__ == '__main__':
    # first num is number of locusts, ue means unequal food sources, eq equal
    # data, info = import_data('30UE20191206_tracked.csv', smoothing=500)

    # data, info = import_data('05EQ20200615_tracked.csv', smoothing=500)
    # data, info = import_data('05UE20200625_tracked.csv', smoothing=50)
    import glob
    # get 10 different n=10 datasets
    # data_files = glob.glob('Locusts/Data/Tracking/05EQ*tracked.csv')[:10]
    # get rid of the directories in front
    # data_files = [f.split('/')[-1] for f in data_files]

    # for data_file in data_files[3:]:
    #     for smoothing in [500]:
    #         data, info = import_data(data_file, smoothing=smoothing)
    #         data = data[:5000]
    #         visualize_data(data, speedup=30)
    #         print('next')

    # data = smooth_data(simulate_random_walks(n=5, T=10000, std=0.0001, seed=3), smoothing=100)
    # data, info = import_data('05UE20200625', smoothing=0)
    # data, info = import_data('05UE20200708', smoothing=0)
    # data, info = import_data('15EQ20191204', smoothing=0)
    # data, info = import_data('15UE20200509', smoothing=0)
    # data, info = import_data('30UE20191206', smoothing=0)

    # data, info = import_data('01EQ20191203', smoothing=0)
    # data, info = import_data('01EQ20191203', smoothing=1000)
    data, info = import_data('01EQ20200917_3', smoothing=0)
    # data = data[3000:6000]
    visualize_data(data, speedup=20)

    # while True:
    #     # get a random file and visualize it
    #     data_files = glob.glob('Locusts/Data/Tracking/05EQ*tracked.csv')
    #     data_file = random.choice(data_files)
    #     # turn the file name into something like 05UE20200625
    #     data_file = data_file.split('/')[-1].split('_')[0]
    #     data, info = import_data(data_file, smoothing=0)

    #     visualize_data(data, info=info, speedup=50)
#

    # data, info = import_data('01EQ20191203', smoothing=1000)
    # data = data[3000:6000]
    # data = data[10000:14000]
    # vel_magnitudes = detect_segments(data)
    # for i in range(data.shape[1]):
    #     plt.plot(vel_magnitudes[:,i])
    # plt.show()
    # visualize_data(data, speedup=10)

    # data = test_data()
