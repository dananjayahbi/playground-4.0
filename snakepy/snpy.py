import pygame
import random

# Constants
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
FPS = 15  # Adjust for speed (higher FPS, faster)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()


class Snake:
    def __init__(self):
        self.body = []
        self.direction = 0  # 0: right, 1: up, 2: left, 3: down
        self.grow_count = 0 # Track how many segments to grow

    def add_segment(self):
        self.body.append((self.body[-1][0] + self.direction_change[0], self.body[-1][1] + self.direction_change[1]))
        self.grow_count = 0 # Reset growth counter

    def move(self):
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction_change[0], head_y + self.direction_change[1])
        self.body.insert(0, new_head)  # Add new head

        if self.grow_count > 0: # Only remove tail if we've grown
            self.body.pop()
        else:
            pass # Don't pop the tail if not growing


    def change_direction(self, direction):
        if direction == 0 and self.direction != 2:  # Right
            self.direction_change = (1, 0)
        elif direction == 1 and self.direction != 3: # Up
            self.direction_change = (0, -1)
        elif direction == 2 and self.direction != 0: # Left
            self.direction_change = (-1, 0)
        elif direction == 3 and self.direction != 1: # Down
            self.direction_change = (0, 1)
        else:
            pass

    def draw(self, screen):
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))


class Food:
    def __init__(self, snake):
        self.x = 0
        self.y = 0
        self.snake = snake
        self.generate()

    def generate(self):
        while True:
            self.x = random.randint(0, (WIDTH // GRID_SIZE) - 1)
            self.y = random.randint(0, (HEIGHT // GRID_SIZE) - 1)
            if not self.check_collision():
                break

    def check_collision(self):
        for segment in self.snake.body:
            if segment[0] == self.x and segment[1] == self.y:
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))


# Game Initialization
snake = Snake()
food = Food(snake)
snake.direction_change = (1, 0)  # Initial direction: right

def game_over():
    font = pygame.font.Font(None, 50)
    text = font.render("Game Over", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.wait(2000)  # Wait for 2 seconds before quitting
    pygame.quit()
    exit()


def main():
    global snake, food

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    snake.change_direction(0)
                elif event.key == pygame.K_UP:
                    snake.change_direction(1)
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(2)
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(3)

        # Game Logic
        snake.move()

        # Check for collision with walls
        if (
            snake.body[0][0] < 0
            or snake.body[0][0] >= (WIDTH // GRID_SIZE)
            or snake.body[0][1] < 0
            or snake.body[0][1] >= (HEIGHT // GRID_SIZE)
        ):
            game_over()

        # Check for collision with self
        for i in range(1, len(snake.body)):  # Start from index 1 to avoid checking the head against itself
            if snake.body[0][0] == snake.body[i][0] and snake.body[0][1] == snake.body[i][1]:
                game_over()

        # Check for food consumption
        head_x, head_y = snake.body[0]
        if head_x == food.x and head_y == food.y:
            snake.grow_count = 5 # Grow by 5 segments
            food.generate()


        # Drawing
        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    exit()


if __name__ == "__main__":
    main()
