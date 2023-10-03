import numpy as np
import utils
from scipy.spatial.distance import cdist
import pygame
import random

FPS = 30
WIDTH = 900
HEIGHT = 600
MAX_SPEED = 8
FLEE_RADIUS = 43
MAX_FLEE_FORCE = 22
ALIGN_RADIUS = 120
COHESION_RADIUS = 400
DT = 1/FPS

def simulate_step(pos, vel):
    n = len(pos)
    x_t, v_t, a_t = [], [], []


    def length(vec):
        return np.linalg.norm(vec)

    def normalize(vec):
        return vec / length(vec)

    def scale_to_length(vec, length):
        return normalize(vec) * length

    def separation(i, j):
        '''
        Function for Rule 1 - Separation
        Steer to avoid crowding local flockmates
        '''

        steer = np.zeros(2)
        dist = pos[i] - pos[j]
        desired = np.zeros(2)

        if dist[0] != 0 and dist[0] != 0 :
            if length(dist) < FLEE_RADIUS:
                desired = scale_to_length(dist, MAX_SPEED)
            else:
                desired = scale_to_length(vel[i], MAX_SPEED)
        steer = desired - vel[i]
        if length(steer) > MAX_FLEE_FORCE:
            steer = scale_to_length(steer, MAX_FLEE_FORCE)

        return steer

    def alignment(i):
        '''
        Function for Rule 2 - Alignment
        Steer towards the average heading of local flockmates
        '''

        align = np.zeros(2)
        desired = np.zeros(2)
        for j in range(n):
            if j != i:
                if vel[j][0] != 0 and vel[j][1] != 0:
                    if length(pos[i] - pos[j]) < ALIGN_RADIUS:
                        desired += normalize(vel[j]) * MAX_SPEED

        align = desired - vel[i]
        align =  align // n

        if length(align) > MAX_SPEED:
            align = scale_to_length(align, MAX_SPEED)

        return align

    def cohesion(i):
        '''
        Function for Rule 3 - Cohesion
        Steer to move towards the average position (center of mass) of local flockmates
        '''
        cohes = np.zeros(2)
        average_location = np.zeros(2)
        for j in range(n):
            if j != i:
                dist = pos[i] - pos[j]
                if length(dist) < COHESION_RADIUS:
                    average_location += pos[j]

        average_location = average_location/(n-1)
        cohes = average_location - pos[i]
        cohes = scale_to_length(cohes, MAX_SPEED)
        return cohes


    for i in range(n):
        new_a = np.zeros(2)

        for j in range(n):
            if j != i:
                new_a += separation(i, j)

        new_a += alignment(i)
        new_a += cohesion(i)

        new_v = vel[i] + new_a * DT

        if len(new_v) > MAX_SPEED:
            new_v = scale_to_length(new_v, MAX_SPEED)

        new_x = pos[i] + new_v

        if new_x[0] > WIDTH:
            new_x[0] = 0
        elif new_x[0] < 0:
            new_x[0] = WIDTH

        if new_x[1] > HEIGHT:
            new_x[1] = 0
        elif new_x[1] < 0:
            new_x[1] = HEIGHT

        x_t.append(new_x)
        v_t.append(new_v)
        a_t.append(new_a)

    return np.array(x_t), np.array(v_t), np.array(a_t)


def pygame_simulate_boids(n):
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
    n = 15
    pygame_simulate_boids(n=n)
    # x, v, a = simulate_boids(n=n, T=T)
    # save_and_animate_boids(x, v, a, overwrite=True, frame_rate=frame_rate)
