"""
Testing various AIs with various lookup-table functions on tic-tac-toe
"""
# %%
import numpy as np 


# %%
"""
Hashes out all possible board trajectories, then grades them.
Working backwards from all finished states (with either 0, 0.5, or 1 grade), for any non-finished
state, assigns maximum value of all possible future moves if my turn, minimum value if
opponent's turn.
The function assumes that it's "symbol"'s turn.
Speed: O(n!) where n is # of slots -> ~O(3^n) with dynamic programming to make a full lookup table
Simplicity: idk, took ~an hour for me to code up
"""
utilities = {}
def utility(board: np.array, symbol, opp_symbol) -> int:
    #print(board)
    init_score = grade_board(board)
    if init_score != -1:
        return init_score
    
    all_traj = []
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                board_copy = np.copy(board)
                board_copy[row][col] = symbol
                all_traj.append(board_copy)
    
    """
    grades = []
    for traj in all_traj:
        if not (traj.tobytes() in utilities.keys()):
            utilities[traj.tobytes()] = utility(traj, opp_symbol, symbol)
        grades.append(utilities[traj.tobytes()])
    """
    grades = [utility(traj, opp_symbol, symbol) for traj in all_traj]
    #print(all_traj)
    #print(grades)
    if symbol in grades:
        return symbol
    if 0.5 in grades:
        return 0.5
    return opp_symbol

"""
Grades a board -- either the winner, 0.5 if draw, or -1 if not finished
(given one player is 1, the other is 2)
Inspiration: https://gist.github.com/qianguigui1104/edb3b11b33c78e5894aad7908c773353 
"""
def grade_board(board: np.array):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] and board[row][0] != 0:
            return board[row][0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != 0:
            return board[0][col]
        
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        return board[0][2]
    
    full = True
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                full = False
    if full:
        return 0.5
    return -1

def hash(board, symbol) -> int:
    out = 0
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for row in range(3):
        for col in range(3):
            out += primes[3*row + col] * board[row][col]
    out += 29 * symbol
    return

print(utility(np.zeros((3, 3)), 1, 2)) #expected: 0.5
print(utility(np.array([[1, 0, 1], [2, 2, 0], [1, 0, 2]]), 1, 2)) 
#X _ X | O O _ | X _ O; expected: 1
print(utility(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]]), 1, 2))
#X _ _ | _ O _ | _ _ _; expected: 0.5
"""
Is this a utility function model or a value model? The recursion part might make it a value model,
but we're also effectively using brute-force here (the only "heuristic" here is end-game score).
"""

# %%
"""
All heuristics; see https://blog.ostermiller.org/tic-tac-toe-strategy/ for an example
Pseudocode:
def act(board, symbol = 'X', opp_symbol = 'O'):
    
"""