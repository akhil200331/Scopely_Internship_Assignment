import random
# grid initialising with grid_size 4*4
def initialize_grid(grid_size):
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    # Add two random tiles to the grid at first.
    for i in range(2):
        row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        while grid[row][col] != 0:
            row, col = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        grid[row][col] = random.choice([2, 4])
    return grid
