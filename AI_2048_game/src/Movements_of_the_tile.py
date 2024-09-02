def move_left(grid):
    changed = False
    for row in range(len(grid)):
        for col in range(1, len(grid)):
            if grid[row][col] != 0 and grid[row][col - 1] == 0:
                grid[row][col - 1] = grid[row][col]
                grid[row][col] = 0
                changed = True
            elif grid[row][col] != 0 and grid[row][col - 1] == grid[row][col]:
                grid[row][col - 1] *= 2
                grid[row][col] = 0
                changed = True
    return changed

def move_right(grid):
    changed = False
    for row in range(len(grid)):
        for col in range(len(grid) - 2, -1, -1):
            if grid[row][col] != 0 and grid[row][col + 1] == 0:
                grid[row][col + 1] = grid[row][col]
                grid[row][col] = 0
                changed = True
            elif grid[row][col] != 0 and grid[row][col + 1] == grid[row][col]:
                grid[row][col + 1] *= 2
                grid[row][col] = 0
                changed = True
    return changed

def move_up(grid):
    changed = False
    for col in range(len(grid)):
        for row in range(1, len(grid)):
            if grid[row][col] != 0 and grid[row - 1][col] == 0:
                grid[row - 1][col] = grid[row][col]
                grid[row][col] = 0
                changed = True
            elif grid[row][col] != 0 and grid[row - 1][col] == grid[row][col]:
                grid[row - 1][col] *= 2
                grid[row][col] = 0
                changed = True
    return changed

def move_down(grid):
    changed = False
    for col in range(len(grid)):
        for row in range(len(grid) - 2, -1, -1):
            if grid[row][col] != 0 and grid[row + 1][col] == 0:
                grid[row + 1][col] = grid[row][col]
                grid[row][col] = 0
                changed = True
            elif grid[row][col] != 0 and grid[row + 1][col] == grid[row][col]:
                grid[row + 1][col] *= 2
                grid[row][col] = 0
                changed = True
    return changed