import pygame
import random
import math
import numpy as np


class Boid:

    def __init__(self, x, y, dx=0, dy=0):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([dx, dy], dtype=float)

    def distance(self, other):
        return np.linalg.norm(self.position - other.position)

class Flock:

    def __init__(self, num_boids, width, height):
        self.boids = [Boid(random.randint(0, width),
                           random.randint(0, height),
                           random.random(),
                           random.random())
                      for _ in range(num_boids)]

    def rule1(self, boid):
        pcJ = np.array([0.0, 0.0])
        for bj in self.boids:
            if boid != bj:
                pcJ += bj.position
        pcJ /= len(self.boids) - 1
        return (pcJ - boid.position) * 0.01

    def rule2(self, boid):
        c = np.array([0.0, 0.0])
        for bj in self.boids:
            if boid != bj:
                if boid.distance(bj) < 100:
                    c -= (bj.position - boid.position)
        return c

    def rule3(self, boid):
        pvJ = np.array([0.0, 0.0])
        for bj in self.boids:
            if boid != bj:
                pvJ += bj.velocity
        pvJ /= len(self.boids) - 1
        return (pvJ - boid.velocity) * 0.125


def compute_new_positions(flock):
    limit = np.array([1.0, 1.0])
    for boid in flock.boids:
        v1 = flock.rule1(boid)
        v2 = flock.rule2(boid)
        v3 = flock.rule3(boid)
        boid.velocity = boid.velocity + v1 + v2 + v3
        boid.velocity = (boid.velocity/np.linalg.norm(boid.velocity)) * limit
        boid.position = boid.position + boid.velocity


def render_flock(screen, flock):
    for boid in flock.boids:
        pygame.draw.circle(screen, (255, 255, 255), boid.position.astype(int), 2)


def main():
    width, height = 800, 600
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((width, height))
    flock = Flock(100, width, height)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))
        compute_new_positions(flock)
        render_flock(screen, flock)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
