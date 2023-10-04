import numpy as np
import utils
from scipy.spatial.distance import cdist
import pygame

FPS = 30
WIDTH = 800
HEIGHT = 500

MAX_SPEED = 8
FLEE_RADIUS = 25
ALIGN_RADIUS = 100
COHESION_RADIUS = 100
DT = 1/FPS
WRAP = True
CLIP_VEL = False


def wrap(pos):
    if pos[0] > WIDTH:
        pos[0] = 0
    elif pos[0] < 0:
        pos[0] = WIDTH

    if pos[1] > HEIGHT:
        pos[1] = 0
    elif pos[1] < 0:
        pos[1] = HEIGHT

    return pos


def simulate_step(pos, vel):
    n = len(pos)
    x_t, v_t, a_t = [], [], []
    dist = cdist(pos, pos)

    for i in range(n):
        new_a = np.zeros(2)

        for j in range(n):
            if j == i: continue

            # separation
            new_a += (1 if dist[i, j] < FLEE_RADIUS else 0) * (pos[j] - pos[i])

            # alignment
            new_a += (1 if dist[i, j] < ALIGN_RADIUS else 0) * (1/n) * (vel[j] - vel[i])

            # cohesion
            new_a += (1 if dist[i, j] < COHESION_RADIUS else 0) * (1/100) * (1/(n-1)) * (pos[j] - pos[i])

        new_v = vel[i] + new_a * DT

        if CLIP_VEL:
            if np.linalg.norm(new_v) > MAX_SPEED:
                new_v = new_v * (MAX_SPEED / np.linalg.norm(new_v))

        new_x = pos[i] + new_v

        if WRAP:
            new_x = wrap(new_x)

        x_t.append(new_x)
        v_t.append(new_v)
        a_t.append(new_a)

    return np.array(x_t[:len(a_t)]), np.array(v_t[:len(a_t)]), np.array(a_t)


def pygame_simulate_boids(n, save=False, T=None):
    # Initialize pygame
    pygame.init()

    # Screen dimensions
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Boids Simulation")

    pos = np.random.uniform(min(WIDTH, HEIGHT)//4, 3 * min(WIDTH, HEIGHT) // 4, (n, 2))
    vel = np.random.uniform(-MAX_SPEED, MAX_SPEED, (n, 2))

    # (3, T, BOIDZ, 2)
    x, v, a = [pos], [vel], []

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        new_x, new_v, new_a = simulate_step(x[-1], v[-1])

        x.append(new_x)
        v.append(new_v)
        a.append(new_a)

        for pos in new_x:
            pygame.draw.circle(screen, (255, 255, 255), pos, 2)

        pygame.display.flip()
        fps = 30
        clock.tick(fps)

        if len(x) == T and save:
            running = False

    pygame.quit()

    x, v, a = x[:len(a)], v[:len(a)], a[:len(a)]
    x, v, a = np.array(x), np.array(v), np.array(a)

    if save:
        save_and_animate_boids(x, v, a)


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
        # make a circle of radius around each point in x[t, :, :]
        for (x_, y_) in x[t, :, :]:
            circle = plt.Circle((x_, y_), FLEE_RADIUS, color='r', fill=False)
            plt.gca().add_artist(circle)
            circle = plt.Circle((x_, y_), ALIGN_RADIUS, color='b', fill=False)
            plt.gca().add_artist(circle)
            circle = plt.Circle((x_, y_), COHESION_RADIUS, color='g', fill=False)
            plt.gca().add_artist(circle)


        plt.savefig("/tmp/boids_%03d.png"%t)
        plt.close()

    if overwrite:
        path = 'boids_data3/boids.gif'
    elif i is None or i == -1:
        path, i2 = utils.next_unused_path('boids_data3/boids.gif', return_i=True)
        if i == -1 and i2 != i:
            raise ValueError(f'expected i={i}, got i={i2}')
    else:
        path = f'boids_data3/boids_{i}.gif'
    print('converting images to gif...')
    os.system(f"gm convert -delay 5 -loop 0 /tmp/boids_*.png {path}")
    print('animation saved to', path)


def save_and_animate_boids(x, v, a, overwrite=False, frame_rate=1):
    arr = np.stack([x, v, a])
    if overwrite:
        path = 'boids_data3/boids_data.npy'
        i = 0
    else:
        path, i = utils.next_unused_path('boids_data3/boids_data.npy', return_i=True)

    np.save(path, arr)
    print(f'saved to {path}')
    animate(x, i, overwrite, frame_rate)


def load_boids(i=None):
    s = "boids_data3/boids_data" + ('' if i is None else f'_{i}') + '.npy'
    arr = np.load(s)

    n = arr.shape[2]
    print('num boids: ', n)

    # T = arr.shape[1]
    # assert arr.shape == (3, T, BOIDZ, 2)
    x, v, a = arr
    x, v, a = x[:5], v[:5], a[:5]
    return x, v, None, a


if __name__ == '__main__':
    n = 10
    pygame_simulate_boids(n=n, save=False, T=100)
    # x, v, a = simulate_boids(n=n, T=T)
    # save_and_animate_boids(x, v, a, overwrite=True, frame_rate=frame_rate)
