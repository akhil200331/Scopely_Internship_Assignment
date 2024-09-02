import random
def add_new_tile(grid):
    empty_cells = []
    for row in range(len(grid)):
        for col in range(len(grid)):
            if grid[row][col] == 0:
                empty_cells.append((row, col))
    if empty_cells:
        row, col = random.choice(empty_cells)
        grid[row][col] = random.choice([2, 4])