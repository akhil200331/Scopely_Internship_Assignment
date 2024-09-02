from src.Evaluating_score import evaluate_state
from src.Movements_of_the_tile import move_down,move_left,move_right,move_up
from src.Adding_the_tile_after_Movement import add_new_tile

def choose_best_move(grid):
    best_move = None
    best_score = 0
    grid1=grid.copy()
    for move in ["left", "right", "up", "down"]:
        simulated_grid = [row[:] for row in grid1]
        if move == "left":
            if move_left(simulated_grid):
                add_new_tile(simulated_grid)
        elif move == "right":
            if move_right(simulated_grid):
                add_new_tile(simulated_grid)
        elif move == "up":
            if move_up(simulated_grid):
                add_new_tile(simulated_grid)
        elif move == "down":
            if move_down(simulated_grid):
                add_new_tile(simulated_grid)
        score = evaluate_state(simulated_grid)
        if score[0] > best_score:
            best_move = move
            best_score = score[0]
            grid=simulated_grid
    return best_move