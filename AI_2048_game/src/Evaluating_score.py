def evaluate_state(grid):
    highest_tile = 0
    empty_cells = 0
    smoothness = 0
    for row in range(len(grid)):
        for col in range(len(grid)):
            if grid[row][col] != 0:
                highest_tile = max(highest_tile, grid[row][col])
            else:
                empty_cells += 1
            if col < len(grid) - 1:
                smoothness += abs(grid[row][col] - grid[row][col + 1])
    return highest_tile, empty_cells, smoothness