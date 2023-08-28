import numpy as np
import utils
from scipy.spatial.distance import cdist
import pygame
import random

def simulate_step(pos, vel, too_close_radius=3, visible_radius=4, max_vel=1, separation_coeff=1, flocking_coeff=1, alignment_coeff=1, xmin=0, xmax=800, ymin=0, ymax=600):
    distances = cdist(pos, pos)
    n = len(pos)
    x_t, v_t, a_t = [], [], []

    for i in range(n):
        too_close_neighbors = np.where(distances[i] < too_close_radius)[0]
        visible_neighbors = np.where(distances[i] < visible_radius)[0]
        a1, a2, a3 = np.zeros(2), np.zeros(2), np.zeros(2)

        # move away from too close neighbors
        a1 = separation_coeff * (pos[i] - pos[too_close_neighbors]).sum(axis=0)

        # more towards COM of visible neighbors
        a2 = flocking_coeff * (pos[visible_neighbors] - pos[i]).mean(axis=0) / 100

        # make velocity similar to neighbor velocity
        a3 = alignment_coeff * (vel[visible_neighbors] - vel[i]).mean(axis=0) / 8

        new_a = a1 + a2 + a3
        new_v = vel[i] + new_a

        if np.linalg.norm(new_v) > max_vel:
            new_v = new_v / np.linalg.norm(new_v)

        new_x = pos[i] + new_v

        # if out of bounds, wrap
        if new_x[0] < xmin: new_x[0] = xmax
        if new_x[0] > xmax: new_x[0] = xmin
        if new_x[1] < ymin: new_x[1] = ymax
        if new_x[1] > ymax: new_x[1] = ymin

        x_t.append(new_x)
        v_t.append(new_v)
        a_t.append(new_a)

    return np.array(x_t), np.array(v_t), np.array(a_t)


def simulate_boids(n, T=100):
    print('simulating...')
    too_close_radius = 3
    visible_radius = 4
    max_vel = 1

    separation_coeff = .2
    flocking_coeff = .1
    alignment_coeff = 2

    pos = np.random.uniform(-10*n, 10*n, (n, 2))
    vel = np.random.uniform(-1, 1, (n, 2))

    # (3, T, BOIDZ, 2)
    x, v, a = [pos], [vel], []

    for _ in range(T):
        new_x, new_v, new_a = simulate_step(x[-1], v[-1], too_close_radius, visible_radius, max_vel, separation_coeff, flocking_coeff, alignment_coeff)

        x.append(new_x)
        v.append(new_v)
        a.append(new_a)


    print('done simulating')
    assert len(x) == len(v) == len(a) + 1
    return np.array(x[:-1]), np.array(v[:-1]), np.array(a)

def pygame_simulate_boids(n, T=100):
    # Initialize pygame
    pygame.init()

    # Screen dimensions
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Boids Simulation")

    too_close_radius = 3
    visible_radius = 4
    max_vel = 1

    separation_coeff = 1
    flocking_coeff = 1
    alignment_coeff = 1

    pos = np.random.uniform(0, min(width, height), (n, 2))
    vel = np.random.uniform(-1, 1, (n, 2))

    # (3, T, BOIDZ, 2)
    x, v, a = [pos], [vel], []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        new_x, new_v, new_a = simulate_step(x[-1], v[-1], too_close_radius, visible_radius, max_vel, separation_coeff, flocking_coeff, alignment_coeff, 0, width, 0, height)

        x.append(new_x)
        v.append(new_v)
        a.append(new_a)

        for pos in new_x:
            pygame.draw.circle(screen, (255, 255, 255), pos, 2)

        pygame.display.flip()

    pygame.quit()


def animate(x, i=None, overwrite=False, frame_rate=1):
    from matplotlib import pyplot as plt
    import os

    buffer = 10
    minx, maxx = np.min(x[:, :, 0]) - buffer, np.max(x[:, :, 0]) + buffer
    miny, maxy = np.min(x[:, :, 1]) - buffer, np.max(x[:, :, 1]) + buffer
    # actually, just focus on the middle third of the box
    # minx, maxx = minx + (maxx - minx) / 3, maxx - (maxx - minx) / 3

    # remove all images in /tmp/boids
    os.system("rm /tmp/boids_*.png")

    print('animating...')
    for t in range(0, len(x), frame_rate):
        plt.figure()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])

        plt.scatter(x[t, :, 0], x[t, :, 1], label="boids")
        plt.savefig("/tmp/boids_%03d.png"%t)
        plt.close()

    if overwrite:
        path = 'boids_data2/boids.gif'
    elif i is None or i == -1:
        path, i2 = utils.next_unused_path('boids_data2/boids.gif', return_i=True)
        if i == -1 and i2 != i:
            raise ValueError(f'expected i={i}, got i={i2}')
    else:
        path = f'boids_data2/boids_{i}.gif'
    print('converting images to gif...')
    os.system(f"gm convert -delay 5 -loop 0 /tmp/boids_*.png {path}")
    print('animation saved to', path)


def save_and_animate_boids(x, v, a, overwrite=False, frame_rate=1):
    arr = np.stack([x, v, a])
    if overwrite:
        path = 'boids_data2/boids_data.npy'
        i = 0
    else:
        path, i = utils.next_unused_path('boids_data2/boids_data.npy', return_i=True)

    np.save(path, arr)
    print(f'saved to {path}')
    animate(x, i, overwrite, frame_rate)


def load_boids(i=4):
    arr = np.load(f"boids_data2/boids_data_{i}.npy")
    # T = arr.shape[1]
    # assert arr.shape == (3, T, BOIDZ, 2)
    x, v, a = arr
    return x, v, None, a


if __name__ == '__main__':
    T = 10000
    n = 100
    frame_rate = 10
    pygame_simulate_boids(n=n, T=T)
    # x, v, a = simulate_boids(n=n, T=T)
    # save_and_animate_boids(x, v, a, overwrite=True, frame_rate=frame_rate)
