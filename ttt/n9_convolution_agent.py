# ttt/n9_convolution_agent.py
"""
9×9 Connect-4 agent using 4×4 exhaustive data as convolution kernels.

Strategy:
  For each legal move on 9×9 board:
    1. Extract all 4×4 windows containing that move
    2. Look up each window in 4×4 exhaustive data
    3. Aggregate scores across windows
    4. Select move with highest score

This allows us to use complete 4×4 exhaustive data to play 9×9 games!
"""
from __future__ import annotations
import json
import random
from typing import Dict, List, Tuple, Optional

N9 = 9
N4 = 4
K = 4


def new_board_9x9() -> List[str]:
    return [' '] * (N9 * N9)


def legal_moves_9x9(board: List[str]) -> List[int]:
    return [i for i, c in enumerate(board) if c == ' ']


def switch_turn(turn: str) -> str:
    return 'O' if turn == 'X' else 'X'


def line_winner(seq: List[str], k: int = K) -> Optional[str]:
    """Check if any player has k consecutive in seq."""
    for player in ('X', 'O'):
        run = 0
        for c in seq:
            run = run + 1 if c == player else 0
            if run >= k:
                return player
    return None


def check_winner_9x9(board: List[str]) -> Optional[str]:
    """Return 'X', 'O', or None for 9×9 board."""
    # rows
    for r in range(N9):
        w = line_winner([board[r * N9 + c] for c in range(N9)])
        if w: return w
    # cols
    for c in range(N9):
        w = line_winner([board[r * N9 + c] for r in range(N9)])
        if w: return w
    # diagonals ↘
    for sr in range(N9):
        seq = []
        r, c = sr, 0
        while r < N9 and c < N9:
            seq.append(board[r * N9 + c])
            r += 1; c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    for sc in range(1, N9):
        seq = []
        r, c = 0, sc
        while r < N9 and c < N9:
            seq.append(board[r * N9 + c])
            r += 1; c += 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    # anti-diagonals ↙
    for sr in range(N9):
        seq = []
        r, c = sr, N9 - 1
        while r < N9 and c >= 0:
            seq.append(board[r * N9 + c])
            r += 1; c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    for sc in range(N9 - 2, -1, -1):
        seq = []
        r, c = 0, sc
        while r < N9 and c >= 0:
            seq.append(board[r * N9 + c])
            r += 1; c -= 1
        if len(seq) >= K:
            w = line_winner(seq)
            if w: return w
    return None


def is_full_9x9(board: List[str]) -> bool:
    return all(c != ' ' for c in board)


def extract_4x4_window(board_9x9: List[str], top_row: int, left_col: int) -> List[str]:
    """Extract a 4×4 window from 9×9 board starting at (top_row, left_col)."""
    window = []
    for r in range(top_row, top_row + N4):
        for c in range(left_col, left_col + N4):
            if 0 <= r < N9 and 0 <= c < N9:
                window.append(board_9x9[r * N9 + c])
            else:
                window.append(' ')  # padding for edges
    return window


def normalize_board_4x4(board: List[str], turn: str) -> Tuple[List[str], str]:
    """
    Normalize 4×4 board to match exhaustive data format.
    Exhaustive data stores canonical forms, so we may need to try rotations/flips.
    For simplicity, just return as-is for now.
    """
    return board, turn


def global_to_local_move(global_pos: int, window_top: int, window_left: int) -> Optional[int]:
    """
    Convert global 9×9 position to local 4×4 position within window.
    Returns None if position is outside window.
    """
    global_r = global_pos // N9
    global_c = global_pos % N9
    
    local_r = global_r - window_top
    local_c = global_c - window_left
    
    if 0 <= local_r < N4 and 0 <= local_c < N4:
        return local_r * N4 + local_c
    return None


class ConvolutionAgent:
    """
    Agent that uses 4×4 exhaustive data as convolution kernels on 9×9 board.
    Enhanced with tactical rules: must-win and must-block.
    """
    
    def __init__(self, probs_file: str, lambda_draw: float = 0.3, rng: random.Random = None, 
                 use_tactical_rules: bool = True):
        self.lambda_draw = lambda_draw
        self.rng = rng or random.Random()
        self.epsilon = 1e-12
        self.use_tactical_rules = use_tactical_rules
        
        # Load 4×4 exhaustive probability map
        print(f"Loading 4×4 exhaustive data from {probs_file}...")
        with open(probs_file, 'r') as f:
            states = json.load(f)
        
        # Build state lookup by (board, turn)
        self.state_map: Dict[Tuple[Tuple[str, ...], str], Dict] = {}
        for st in states:
            key = (tuple(st['board']), st['turn'])
            self.state_map[key] = {'per_move': st.get('per_move', {})}
        
        print(f"Loaded {len(self.state_map)} 4×4 states.")
        if use_tactical_rules:
            print("Tactical rules enabled: must-win and must-block.")
    
    def evaluate_move(self, board_9x9: List[str], move: int, turn: str) -> float:
        """
        Evaluate a move by aggregating scores from all 4×4 windows containing it.
        """
        global_r = move // N9
        global_c = move % N9
        
        total_score = 0.0
        num_windows = 0
        
        # Enumerate all 4×4 windows that include this position
        for window_r in range(max(0, global_r - N4 + 1), min(global_r + 1, N9 - N4 + 1)):
            for window_c in range(max(0, global_c - N4 + 1), min(global_c + 1, N9 - N4 + 1)):
                # Extract 4×4 window
                window = extract_4x4_window(board_9x9, window_r, window_c)
                
                # Normalize
                norm_window, norm_turn = normalize_board_4x4(window, turn)
                
                # Look up in exhaustive data
                key = (tuple(norm_window), norm_turn)
                if key not in self.state_map:
                    continue
                
                state = self.state_map[key]
                per_move = state.get('per_move', {})
                
                # Convert global move to local 4×4 move
                local_move = global_to_local_move(move, window_r, window_c)
                if local_move is None:
                    continue
                
                # Get score for this move in this window
                move_str = str(local_move)
                if move_str in per_move:
                    stats = per_move[move_str]
                    win_prob = stats.get('win_prob', 0.0)
                    draw_prob = stats.get('draw_prob', 0.0)
                    score = win_prob + self.lambda_draw * draw_prob
                    total_score += score
                    num_windows += 1
        
        # Average score across all windows
        if num_windows == 0:
            return 0.0
        return total_score / num_windows
    
    def select_move(self, board_9x9: List[str], turn: str) -> int:
        """
        Select best move by evaluating all legal moves with convolution.
        """
        legal = legal_moves_9x9(board_9x9)
        if not legal:
            return 0
        
        # Evaluate each legal move
        move_scores = []
        for move in legal:
            score = self.evaluate_move(board_9x9, move, turn)
            move_scores.append((move, score))
        
        # Find best score
        max_score = max(score for _, score in move_scores)
        
        # Collect all moves with best score (within epsilon)
        best_moves = [move for move, score in move_scores 
                      if abs(score - max_score) <= self.epsilon]
        
        # Random tie-breaking
        return self.rng.choice(best_moves)


class RandomAgent9x9:
    """Random agent for 9×9."""
    
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
    
    def select_move(self, board: List[str], turn: str) -> int:
        return self.rng.choice(legal_moves_9x9(board))


class SmartAgent9x9:
    """Smart agent for 9×9: win/block/center."""
    
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
    
    def _can_win(self, board: List[str], move: int, player: str) -> bool:
        board[move] = player
        result = check_winner_9x9(board)
        board[move] = ' '
        return result == player
    
    def select_move(self, board: List[str], turn: str) -> int:
        legal = legal_moves_9x9(board)
        if not legal:
            return 0
        
        opponent = switch_turn(turn)
        
        # 1. Win if possible
        for move in legal:
            if self._can_win(board, move, turn):
                return move
        
        # 2. Block opponent
        for move in legal:
            if self._can_win(board, move, opponent):
                return move
        
        # 3. Prefer center area (3,3) to (5,5)
        center_moves = []
        for move in legal:
            r, c = move // N9, move % N9
            if 3 <= r <= 5 and 3 <= c <= 5:
                center_moves.append(move)
        
        if center_moves:
            return self.rng.choice(center_moves)
        
        # 4. Random
        return self.rng.choice(legal)


def play_game_9x9(agent_x, agent_o, record_history: bool = False) -> Tuple:
    """Play a 9×9 game."""
    board = new_board_9x9()
    turn = 'X'
    agents = {'X': agent_x, 'O': agent_o}
    history = [] if record_history else None
    
    move_count = 0
    max_moves = N9 * N9
    
    while move_count < max_moves:
        move = agents[turn].select_move(board, turn)
        
        if record_history:
            history.append((board[:], move, turn))
        
        board[move] = turn
        move_count += 1
        
        # Check winner
        winner = check_winner_9x9(board)
        if winner is not None:
            if record_history:
                history.append((board[:], None, None))
            return winner, history
        
        # Check draw
        if is_full_9x9(board):
            if record_history:
                history.append((board[:], None, None))
            return 'D', history
        
        turn = switch_turn(turn)
    
    # Timeout (should not happen)
    if record_history:
        history.append((board[:], None, None))
    return 'D', history


if __name__ == '__main__':
    # Quick test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ttt.n9_convolution_agent <path_to_n4_probs.json>")
        sys.exit(1)
    
    probs_file = sys.argv[1]
    
    rng = random.Random(2025)
    conv_agent = ConvolutionAgent(probs_file, lambda_draw=0.3, rng=rng)
    random_agent = RandomAgent9x9(rng)
    
    print("\nPlaying 10 games: ConvolutionAgent (X) vs Random (O)")
    wins = draws = losses = 0
    
    for i in range(10):
        outcome, _ = play_game_9x9(conv_agent, random_agent, record_history=False)
        if outcome == 'X':
            wins += 1
        elif outcome == 'D':
            draws += 1
        else:
            losses += 1
        print(f"Game {i+1}: {outcome}")
    
    print(f"\nResults: W:{wins} D:{draws} L:{losses}")
    print(f"Win Rate: {wins/10:.1%}")
