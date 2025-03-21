# Pygame snake game with a GUI
import sys, time  # Import time
import pygame
import os  # Import os for clearing the screen


# Constants (you can adjust these!)
BLOCK_size = 20
SPEED = 1  # Adjust for game speed

FPS = 60 
SLEET = 1 / FPS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()

# Set window attributes
screen_width = 10 * BLOCK_size
screen_height = 10 * BLOCK_size
screen =pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")


# Load image
snake_img = pygame.image.load("snake_image.png") #Change to your snake image path

# Initialize snake
snake_list = []
snake_img = pygame.transform.scale(snake_img, (BLOCK_size, BLOCK_size)) #Scale the image

snake_head_coords = (BLOCK_size, BLOCK_size)
snake_block_coords = (BLOCK_size, BLOCK_size)

# Initialize food
food_coords = (os.getpid() * BLOCK_size, os.getpid() * BLOCK_size)  #Example:  Random coordinate generation


def draw_snake(snake_list, block_size):
    for x, y in snake_list:
        pygame.draw.rectangle(screen, RED, [(x, y), (x + block_size, y + block_size)])  # Draw each block


def redraw_game_deluxe(snake_list, food_coords):
    screen.fill(BLACK) 
    draw_snake(snake_list, BLOCK_size)
    pygame.draw.rect(screen, GREEN, food_coords)

    if snake_list and food_coords == snake_list[0]: #Check if Snake eats food
        food_coords = (os.getpid() * BLOCK_size, os.getpid() * BLOCK_size)    
      
    pygame.display.update()


def main():
    game = True 
    snake_x = BLOCK_SIZE
    snake_y = BLOCK_size 

    snake_x_change = 0
    snake_y_change = 0 


    clock = pygame.time.Clock() 

    while game:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.type == pygame.K_RIGHT and snake_x_change == 0:
                    snake_x_change = BLOCK_size
                    snake_y_change = 0
                if event.type == pygame.K_UP and snake_y_change == 0:
                    snake_x_code = 0
                    snake_y_change = - BLOCK_size
                if event.type == pygame.K_LEFT and snake_x_change == 0:
                    snake_x_change = - BLOCK_size
                    snake_y_change = 0
                if event.type == pygame.K_DOWN and snake_y_change == 0:
                    snake_x_code = 0
                    snake_y_change = BLOCK_size  
               
        # Game Logic
        snake_x += snake_x_change  #Update snake's coordinates.
        snake_y += snake_y_change  
          
        # Check for collision/bounds
        if snake_x >= screen_width or snake_x < 0 or snake_y >= screen_height or snake_y < 0:
            game = False 
            
        #Food generation logic - generates a new food
        if snake_x == food_coords[0] and snake_y == food_coords[1]:
            food_coords = (os.getpid()*BLOCK_size, os.getpid() * BLOCK_size)  
 


        snake_list = list([(snake_x, snake_y)])
        
        if snake_list and snake_list[0][0] == food_coords[0] and snake_list[0][1] == food_coords[1]:
          
            game = False #Win Condition

        # Draw Everything.
        redraw_game_deluxe(snake_list, food_coords)
        pygame.time.tick(FPS)  


    pygame.quit()
    sys.exit()

if __name__=='__main__' :
    main()