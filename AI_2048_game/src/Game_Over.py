def is_game_over(grid):
    for row in range(len(grid)):
        for col in range(len(grid)):
            if grid[row][col] == 0:
                return False
            elif col < len(grid) - 1 and grid[row][col] == grid[row][col + 1]:
                return False
            elif row < len(grid) - 1 and grid[row][col] == grid[row + 1][col]:
                return False
    return True