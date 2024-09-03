# AI 2048 Game

# main_Game_2048.py
  1. grid_size=4*4
  2. start the game (play_game)
      1. First best possible by Agent (Left,Right,Up,Down)
      2. Until Game is over we play the game

# Adding_the_tile_after_Movement.py
    1. The add_new_tile function places a random tile (2 or 4) in an empty cell of the given grid.


# best_move.py
    1. The choose_best_move function simulates all possible moves (left, right, up, down) on a grid, evaluates the resulting grid states, and returns the move that produces the highest score, updating the grid accordingly.

# Displaying_the_grid.py
    1. Printing the grid

# Evaluating_score.py
    1. The evaluate_state function calculates and returns the highest tile value, the number of empty cells, and the smoothness (difference between adjacent tiles) in the grid.

# Game_Over.py
    1. The is_game_over function checks if the game is over by determining if there are no empty cells and no possible adjacent tile merges in any direction. If no moves are possible, it returns True; otherwise, it returns False.

# Game_start.py
    1. The play_game function runs the 2048 game, where the AI continuously chooses the best move for the grid. After each move, the grid's state is displayed before and after adding a new tile. The game continues until no more moves are possible, at which point it checks if the player has won (by reaching 256) or if the game is over.
    
# intializing_grid.py
    1. The initialize_grid function creates a grid_size x grid_size grid filled with zeros and randomly places two tiles (either 2 or 4) in empty positions at the start of the game.

# Movements_of_the_tile.py
    1. The move_left, move_right, move_up, and move_down functions implement tile movement and merging for the 2048 game.
    2. They shift tiles in the specified direction, merge adjacent tiles with the same value, and return whether any changes were made to the grid.



# Gen AI (ChatGPT)  has assisted in:
1. Designing the Code: Provided guidance on structuring and implementing game mechanics efficiently.
2. Choosing Optimal Moves: Helped in developing algorithms to select the best moves for the AI player.
3. Handling Game States: Aided in managing scenarios where no valid moves are possible, ensuring proper game-over handling and smooth gameplay.



# Future Enhancement
 1. We can train the agent by generating large datasets automatically, enabling us to leverage the latest reinforcement learning techniques.
 2. Reinforcement learning techniques help the agent learn by rewarding or penalizing moves, allowing it to improve its strategy over time.
