import pygame
import sys
import math

WIDTH, HEIGHT = 800, 600
FPS = 60

ACCELERATION = 0.2
FRICTION = 0.05
ROTATION_SPEED = 3
MAX_SPEED = 5


class Kart:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.image_original = pygame.Surface((50, 30), pygame.SRCALPHA)
        pygame.draw.polygon(self.image_original, (255, 0, 0), [(0, 0), (50, 15), (0, 30)])
        self.image = self.image_original
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y -= self.speed * math.sin(math.radians(self.angle))
        self.speed *= (1 - FRICTION)
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def accelerate(self, forward=True):
        delta = ACCELERATION if forward else -ACCELERATION
        self.speed = max(-MAX_SPEED, min(MAX_SPEED, self.speed + delta))

    def rotate(self, direction):
        self.angle = (self.angle + ROTATION_SPEED * direction) % 360
        self.image = pygame.transform.rotate(self.image_original, self.angle)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


def draw_track(surface):
    outer_rect = pygame.Rect(50, 50, WIDTH - 100, HEIGHT - 100)
    inner_rect = pygame.Rect(200, 150, WIDTH - 400, HEIGHT - 300)
    pygame.draw.rect(surface, (100, 100, 100), outer_rect)
    pygame.draw.rect(surface, (0, 150, 0), inner_rect)



def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption('Go Kart')

    kart = Kart(WIDTH // 2, HEIGHT // 2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            kart.accelerate(True)
        if keys[pygame.K_DOWN]:
            kart.accelerate(False)
        if keys[pygame.K_LEFT]:
            kart.rotate(1)
        if keys[pygame.K_RIGHT]:
            kart.rotate(-1)

        kart.update()

        screen.fill((0, 150, 0))
        draw_track(screen)
        kart.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
