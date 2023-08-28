import pygame
import random
import math

# Initialize pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Boids Simulation")

# Colors
white = (255, 255, 255)

# Boid class
class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)


    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def update(self, boids):
        # Rule 1: Boids move towards the center of mass of nearby birds
        center_x = sum([boid.x for boid in boids if self.distance(boid) < 50]) / len(boids)
        center_y = sum([boid.y for boid in boids if self.distance(boid) < 50]) / len(boids)
        self.vx += (center_x - self.x) * 0.01
        self.vy += (center_y - self.y) * 0.01

        # Rule 2: Boids avoid getting too close to each other
        for boid in boids:
            if boid != self:
                distance = self.distance(boid)
                if distance < 25:
                    self.vx -= (boid.x - self.x) * 0.1 / distance
                    self.vy -= (boid.y - self.y) * 0.1 / distance

        # Rule 3: Boids try to match the velocity of nearby boids
        avg_vx = sum([boid.vx for boid in boids if self.distance(boid) < 50]) / len(boids)
        avg_vy = sum([boid.vy for boid in boids if self.distance(boid) < 50]) / len(boids)
        self.vx += (avg_vx - self.vx) * 0.05
        self.vy += (avg_vy - self.vy) * 0.05

        # Update position
        self.x += self.vx
        self.y += self.vy

        # Wrap around screen edges
        if self.x > width:
            self.x -= width
        elif self.x < 0:
            self.x += width
        if self.y > height:
            self.y -= height
        elif self.y < 0:
            self.y += height

    def draw(self):
        pygame.draw.circle(screen, white, (int(self.x), int(self.y)), 5)

# Create boids
num_boids = 50
boids = [Boid(random.uniform(0, width), random.uniform(0, height)) for _ in range(num_boids)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for boid in boids:
        boid.update(boids)
        boid.draw()

    pygame.display.flip()

pygame.quit()
