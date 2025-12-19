"""
9×9 Pattern-based Agent using 4×4 Exhaustive Lookup
Uses exhaustive 4×4 data to evaluate local patterns in 9×9 game
"""

import numpy as np
from typing import Tuple, Optional, List
from battle_agents.n4_exhaustive_agent_db import ExhaustiveAgentDB
from battle_agents.path_resolver import get_database_path


class N9ExhaustivePatternAgent:
    """
    9×9 agent using exhaustive 4×4 lookup.
    
    Strategy:
    1. Extract all 4×4 windows
    2. Query exhaustive DB
    3. Aggregate scores
    4. Prefer moves strong in 4×4 lookup
    """
    
    def __init__(self, db_file: str = None):
        """
        Initialize agent with 4×4 DB.
        
        Args:
            db_file: path to DB (None = auto)
        """
        print("Loading 9×9 Exhaustive Pattern Agent...")
        
        # Auto path resolve
        if db_file is None:
            db_file = get_database_path()
        # Load JSON win/draw/loss probs
        from battle_agents.path_resolver import get_path_resolver
        resolver = get_path_resolver()
        probs_json = resolver.get_probs_json()
        self.exhaustive_agent = ExhaustiveAgentDB(db_file, probs_json=probs_json)
        self.N = 9
        self.K = 4
        self.window_size = 4
        
    def extract_4x4_windows(self, board: list) -> List[Tuple[int, int, list]]:
        """
        Extract all 4×4 windows from 9×9.
        
        Args:
            board: list of 81 cells
            
        Returns:
            List of (row, col, window_board)
        """
        windows = []
        
        for start_r in range(self.N - self.window_size + 1):
            for start_c in range(self.N - self.window_size + 1):
                window = []
                for r in range(start_r, start_r + self.window_size):
                    for c in range(start_c, start_c + self.window_size):
                        idx = r * self.N + c
                        window.append(board[idx])
                
                windows.append((start_r, start_c, window))
        
        return windows
    
    def evaluate_move_in_windows(self, board: list, move: int, turn: str) -> dict:
        """
        Evaluate a move using 4×4 windows that include it.
        
        Three views:
        1. full board
        2. my stones only
        3. opponent stones only
        """
        move_r = move // self.N
        move_c = move % self.N
        opponent = 'O' if turn == 'X' else 'X'

        perspectives = {
            'full': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0},
            'my_pieces': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0},
            'opp_pieces': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0}
        }
        trivial_count = 0
        trivial_0 = 0
        trivial_1 = 0
        trivial_2 = 0
        max_window_stones = 0

        for start_r in range(max(0, move_r - self.window_size + 1),
                             min(move_r + 1, self.N - self.window_size + 1)):
            for start_c in range(max(0, move_c - self.window_size + 1),
                                 min(move_c + 1, self.N - self.window_size + 1)):
                window_full = []
                window_my = []
                window_opp = []

                for r in range(start_r, start_r + self.window_size):
                    for c in range(start_c, start_c + self.window_size):
                        idx = r * self.N + c
                        cell = board[idx]
                        window_full.append(cell)
                        window_my.append(' ' if cell == opponent else cell)
                        window_opp.append(' ' if cell == turn else cell)

                window_move_r = move_r - start_r
                window_move_c = move_c - start_c
                window_move = window_move_r * self.window_size + window_move_c

                for perspective, window in [
                    ('full', window_full),
                    ('my_pieces', window_my),
                    ('opp_pieces', window_opp)
                ]:
                    if perspective == 'full':
                        stones = sum(1 for x in window if x != ' ')
                        # Count trivial windows
                        if stones == 0:
                            trivial_0 += 1
                            trivial_count += 1
                        elif stones == 1:
                            trivial_1 += 1
                            trivial_count += 1
                        elif stones == 2:
                            trivial_2 += 1
                            trivial_count += 1
                        if stones > max_window_stones:
                            max_window_stones = stones
                    move_probs = self.exhaustive_agent.get_move_probs(window, turn) or {}
                    if str(window_move) in move_probs:
                        probs = move_probs[str(window_move)]
                        wp = probs.get('win_prob', probs.get('p_win', 0.0))
                        dp = probs.get('draw_prob', probs.get('p_draw', 0.0))
                        lp = probs.get('loss_prob', probs.get('p_loss', 0.0))
                        perspectives[perspective]['win'] += wp
                        perspectives[perspective]['draw'] += dp
                        perspectives[perspective]['loss'] += lp
                        perspectives[perspective]['count'] += 1
                        best_move = self.exhaustive_agent.get_move(window, turn)
                        if best_move == window_move:
                            perspectives[perspective]['best_count'] += 1

                result = {}
        for persp_name, persp_data in perspectives.items():
            if persp_data['count'] > 0:
                result[f'{persp_name}_avg_win'] = persp_data['win'] / persp_data['count']
                result[f'{persp_name}_avg_draw'] = persp_data['draw'] / persp_data['count']
                result[f'{persp_name}_avg_loss'] = persp_data['loss'] / persp_data['count']
                result[f'{persp_name}_best_count'] = persp_data['best_count']
                result[f'{persp_name}_window_count'] = persp_data['count']
            else:
                result[f'{persp_name}_avg_win'] = 0.0
                result[f'{persp_name}_avg_draw'] = 1.0
                result[f'{persp_name}_avg_loss'] = 0.0
                result[f'{persp_name}_best_count'] = 0
                result[f'{persp_name}_window_count'] = 0
        result['trivial_window_count'] = trivial_count
        result['trivial_0_count'] = trivial_0
        result['trivial_1_count'] = trivial_1
        result['trivial_2_count'] = trivial_2
        result['max_window_stones'] = max_window_stones

        # Tactical checks
        N, K = self.N, self.K

        # Opp immediate wins before our move
        legal_now = [i for i, v in enumerate(board) if v == ' ']
        opponent = 'O' if turn == 'X' else 'X'
        opp_wins = set()
        for mv in legal_now:
            board[mv] = opponent
            if check_winner(board, N, K) == opponent:
                opp_wins.add(mv)
            board[mv] = ' '
        result['opp_has_immediate'] = 1 if len(opp_wins) > 0 else 0
        result['blocks_opp_immediate'] = 1 if move in opp_wins else 0

        # Our immediate win
        board_tmp = board[:]
        board_tmp[move] = turn
        result['immediate_win'] = 1 if check_winner(board_tmp, N, K) == turn else 0

        # Check open threats
        def creates_open_run(board_after: list, m: int, color: str, need: int):
            r0, c0 = divmod(m, N)
            dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
            L = (K - 1) + need
            def at_edge(r, c):
                return r == 0 or r == N - 1 or c == 0 or c == N - 1
            for dr, dc in dirs:
                cells = []
                coords = []
                r, c = r0, c0
                while 0 <= r - dr < N and 0 <= c - dc < N:
                    r -= dr
                    c -= dc
                rr, cc = r, c
                while 0 <= rr < N and 0 <= cc < N:
                    coords.append((rr, cc))
                    cells.append(board_after[rr * N + cc])
                    rr += dr
                    cc += dc
                for i in range(0, len(cells) - L + 1):
                    seg = cells[i:i + L]
                    left_open = (i - 1 >= 0 and cells[i - 1] == ' ')
                    right_open = (i + L < len(cells) and cells[i + L] == ' ')
                    if not (left_open and right_open):
                        continue
                    cnt_color = sum(1 for x in seg if x == color)
                    cnt_empty = sum(1 for x in seg if x == ' ')
                    cnt_opp = sum(1 for x in seg if x == ('O' if color == 'X' else 'X'))
                    if cnt_opp > 0:
                        continue
                    seg_coords = coords[i:i + L]
                    if not any(rrcc[0] * N + rrcc[1] == m for rrcc in seg_coords):
                        continue
                    touches_border = any(at_edge(r, c) for r, c in seg_coords)
                    left_coord = coords[i - 1] if i - 1 >= 0 else None
                    right_coord = coords[i + L] if i + L < len(coords) else None
                    left_extend_ok = left_coord is not None and board_after[left_coord[0] * N + left_coord[1]] == ' '
                    right_extend_ok = right_coord is not None and board_after[right_coord[0] * N + right_coord[1]] == ' '
                    edge_trapped = False
                    if touches_border:
                        if need == 1 and not (left_extend_ok or right_extend_ok):
                            edge_trapped = True
                        if need == 2 and not (left_extend_ok and right_extend_ok):
                            edge_trapped = True
                    if need == 1 and cnt_color >= K - 1 and cnt_empty >= 1:
                        return True, edge_trapped
                    if need == 2 and cnt_color >= K - 2 and cnt_empty >= 2:
                        return True, edge_trapped
            return False, False

        board_after = board[:]
        board_after[move] = turn
        ok1, trapped1 = creates_open_run(board_after, move, turn, need=1)
        ok2, trapped2 = creates_open_run(board_after, move, turn, need=2)
        result['creates_open_k1'] = 1 if ok1 else 0
        result['creates_open_k2'] = 1 if ok2 else 0
        result['open_k1_edge_trapped'] = 1 if (ok1 and trapped1) else 0
        result['open_k2_edge_trapped'] = 1 if (ok2 and trapped2) else 0

        # Count double threats
        legal_after = [i for i, v in enumerate(board_after) if v == ' ']
        dt = 0
        for nm in legal_after:
            board_after[nm] = turn
            if check_winner(board_after, N, K) == turn:
                dt += 1
            board_after[nm] = ' '
        result['double_threat_count'] = dt

        return result
    
    def check_immediate_threats(self, board: list, turn: str, legal_moves: list, verbose: bool = False) -> Optional[int]:
        """
        Check immediate win or block.
        
        Returns:
            move or None
        """
        opponent = 'O' if turn == 'X' else 'X'
        
        # Our immediate win
        for move in legal_moves:
            board[move] = turn
            if check_winner(board, self.N, self.K) == turn:
                board[move] = ' '
                if verbose:
                    print(f"✓ Winning move: {move}")
                return move
            board[move] = ' '
        
        # Opp immediate win → block
        for move in legal_moves:
            board[move] = opponent
            if check_winner(board, self.N, self.K) == opponent:
                board[move] = ' '
                if verbose:
                    print(f"✗ Block opponent: {move}")
                return move
            board[move] = ' '
        
        return None
    
    def get_move(self, board: list, turn: str, verbose: bool = False) -> Optional[int]:
        """
        Select best move.
        
        Steps:
        1. Immediate win/block
        2. Score using 3 perspectives
        """
        legal_moves = [i for i in range(81) if board[i] == ' ']
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 81:
            return 40  # center
        
        # Step 1
        critical_move = self.check_immediate_threats(board, turn, legal_moves, verbose)
        if critical_move is not None:
            return critical_move
        
        # Step 2: evaluate moves
        move_scores = {}
        
        for move in legal_moves:
            scores = self.evaluate_move_in_windows(board, move, turn)
            
            score = (
                scores['full_avg_win'] * 100.0 +
                scores['full_best_count'] * 10.0 +
                -scores['full_avg_loss'] * 100.0 +
                scores['my_pieces_avg_win'] * 20.0 +
                scores['my_pieces_best_count'] * 2.0 +
                -scores['opp_pieces_avg_win'] * 50.0 +
                scores['opp_pieces_best_count'] * 5.0 +
                scores['full_window_count'] * 0.5
            )
            
            move_scores[move] = {
                'score': score,
                'details': scores
            }
        
        best_move = max(move_scores.items(), key=lambda x: x[1]['score'])[0]
        
        if verbose:
            print(f"\nMove eval (top 5):")
            sorted_moves = sorted(move_scores.items(), key=lambda x: -x[1]['score'])[:5]
            for move, data in sorted_moves:
                r, c = move // self.N, move % self.N
                d = data['details']
                print(f"  Pos {move} ({r},{c}): score={data['score']:.2f}")
                print(f"    Full: win={d['full_avg_win']:.3f}, best={d['full_best_count']}, windows={d['full_window_count']}")
                print(f"    Mine: win={d['my_pieces_avg_win']:.3f}, best={d['my_pieces_best_count']}")
                print(f"    Opp : win={d['opp_pieces_avg_win']:.3f}, best={d['opp_pieces_best_count']}")
        
        return best_move


def check_winner(board: list, N: int = 9, K: int = 4) -> Optional[str]:
    """Check winner (X/O/None)."""
    def check_line(seq):
        if len(seq) < K:
            return None
        for i in range(len(seq) - K + 1):
            window = seq[i:i+K]
            if all(x == 'X' for x in window):
                return 'X'
            if all(x == 'O' for x in window):
                return 'O'
        return None
    
    # rows
    for r in range(N):
        row = [board[r*N + c] for c in range(N)]
        w = check_line(row)
        if w:
            return w
    
    # cols
    for c in range(N):
        col = [board[r*N + c] for r in range(N)]
        w = check_line(col)
        if w:
            return w
    
    # diag ↘
    for start_r in range(N - K + 1):
        for start_c in range(N - K + 1):
            seq = []
            r, c = start_r, start_c
            while r < N and c < N:
                seq.append(board[r*N + c])
                r += 1
                c += 1
            w = check_line(seq)
            if w:
                return w
    
    # diag ↙
    for start_r in range(N - K + 1):
        for start_c in range(K - 1, N):
            seq = []
            r, c = start_r, start_c
            while r < N and c >= 0:
                seq.append(board[r*N + c])
                r += 1
                c -= 1
            w = check_line(seq)
            if w:
                return w
    
    return None


def print_board(board: list, N: int = 9):
    """Print 9×9 board."""
    print("\n  ", end="")
    for c in range(N):
        print(f"{c} ", end="")
    print()
    
    for r in range(N):
        print(f"{r} ", end="")
        for c in range(N):
            idx = r * N + c
            cell = board[idx]
            print(f"{cell if cell != ' ' else '·'} ", end="")
        print()
    print()


def play_game_vs_random(agent: N9ExhaustivePatternAgent, 
                       agent_plays: str = 'X',
                       verbose: bool = True) -> str:
    """
    Play Exhaustive Agent vs random.
    
    Returns:
        'X', 'O', or 'draw'
    """
    import random
    
    board = [' '] * 81
    turn = 'X'
    
    if verbose:
        print("="*60)
        print(f"Game: Exhaustive Agent ({agent_plays}) vs Random")
        print("="*60)
        print_board(board)
    
    move_count = 0
    
    while True:
        winner = check_winner(board)
        if winner:
            if verbose:
                print(f"{'='*60}")
                print(f"Winner: {winner}")
                print(f"{'='*60}")
                print_board(board)
            return winner
        
        if ' ' not in board:
            if verbose:
                print("Draw")
            return 'draw'
        
        # choose move
        if turn == agent_plays:
            move = agent.get_move(board, turn, verbose=False)
            player_name = "Exhaustive Agent"
        else:
            legal = [i for i in range(81) if board[i] == ' ']
            move = random.choice(legal)
            player_name = "Random"
        
        board[move] = turn
        move_count += 1
        
        if verbose:
            r, c = move // 9, move % 9
            print(f"Move {move_count}: {turn} ({player_name}) at ({r},{c})")
            print_board(board)
        
        turn = 'O' if turn == 'X' else 'X'


def demo():
    """Run 5 demo games."""
    agent = N9ExhaustivePatternAgent()
    
    print("\n" + "="*60)
    print("Demo: Exhaustive Agent vs Random (5 games)")
    print("="*60)
    
    results = {'X': 0, 'O': 0, 'draw': 0}
    
    for game_num in range(5):
        print(f"\n{'='*60}")
        print(f"Game {game_num + 1}/5")
        print(f"{'='*60}")
        
        agent_plays = 'X' if game_num % 2 == 0 else 'O'
        result = play_game_vs_random(agent, agent_plays, verbose=False)
        results[result] += 1
        
        print(f"Result: {result}")
        if result == agent_plays:
            print("✓ Agent wins")
        elif result == 'draw':
            print("= Draw")
        else:
            print("✗ Random wins")
    
    print("\n" + "="*60)
    print("Final Results:")
    print("="*60)
    print(f"Wins:  {results.get('X', 0) + results.get('O', 0) - results.get('draw', 0)}")
    print(f"Draws: {results['draw']}")
    print(f"Losses: (see above)")


if __name__ == '__main__':
    demo()
