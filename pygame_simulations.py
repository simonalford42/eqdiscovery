import pygame
import math

def simulate_circle():
    # Parameter Initialization
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
    BACKGROUND_COLOR = (0, 0, 0)
    DOT_COLOR = (255, 255, 255)
    DOT_RADIUS = 10
    CIRCLE_RADIUS = 200
    FPS = 60
    DURATION = 30  # Duration in seconds

    # Pygame Initialization
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # Circle Path Initialization
    circle_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    dot_speed = 2 * math.pi / (FPS * DURATION)  # speed in radians/frame

    def draw_dot(angle):
        x = circle_center[0] + CIRCLE_RADIUS * math.cos(angle)
        y = circle_center[1] + CIRCLE_RADIUS * math.sin(angle)
        pygame.draw.circle(screen, DOT_COLOR, (int(x), int(y)), DOT_RADIUS)

    def main():
        angle = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(BACKGROUND_COLOR)
            draw_dot(angle)
            angle += dot_speed

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

    main()

if __name__ == '__main__':
    simulate_circle()
