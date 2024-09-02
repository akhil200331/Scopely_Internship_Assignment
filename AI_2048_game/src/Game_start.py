import time
from src.initializing_grid import initialize_grid
from src.Displaying_the_grid import print_grid
from src.Movements_of_the_tile import move_down,move_left,move_right,move_up
from src.Adding_the_tile_after_Movement import add_new_tile
from src.Game_Over import is_game_over
from src.Evaluating_score import evaluate_state
from src.best_move import choose_best_move
# from src.Game_start import play_game

def play_game(grid_size):
    grid = initialize_grid(grid_size)
    print(grid)
    while True:
        # print_grid(grid)
        # AI chooses the best move
        time.sleep(5)
        move = choose_best_move(grid)
        print("Move:", move)
        time.sleep(5)
        if move == "left":
            if move_left(grid):
                print("State after move but before adding new tile:")
                print_grid(grid)
                add_new_tile(grid)
                time.sleep(5)
        elif move == "right":
            if move_right(grid):
                print("State after move but before adding new tile:")
                print_grid(grid)
                add_new_tile(grid)
                time.sleep(5)
        elif move == "up":
            if move_up(grid):
                print("State after move but before adding new tile:")
                print_grid(grid)
                add_new_tile(grid)
                time.sleep(5)
        elif move == "down":
            if move_down(grid):
                print("State after move but before adding new tile:")
                print_grid(grid)
                add_new_tile(grid)
                time.sleep(5)
        print("State after adding new tile:")
        print_grid(grid)
        time.sleep(5)
        if is_game_over(grid):
            print_grid(grid)
            if 256 in grid:
                print("Congratulations! You won!")
            else:
                print("Game over!")
            break