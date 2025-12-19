"""
9×9 Pattern-based Agent using 4×4 Exhaustive Lookup
Uses exhaustive 4×4 data to evaluate local patterns in 9×9 game
"""

import numpy as np
from typing import Tuple, Optional, List
from ttt.n4_exhaustive_agent_db import ExhaustiveAgentDB
from ttt.path_resolver import get_database_path


class N9ExhaustivePatternAgent:
    """
    Agent for 9×9 4-in-a-row that uses exhaustive 4×4 lookup

    Strategy:
    1. Extract all 4×4 windows from the 9×9 board
    2. For each window, query exhaustive database for evaluation
    3. Aggregate scores from all windows to select best move
    4. Prioritize moves that appear in high-value positions in 4×4 lookup
    """

    def __init__(self, db_file: str = None):
        """
        Initialize agent with exhaustive 4×4 database

        Args:
            db_file: Path to exhaustive database (None = auto-detect)
        """
        print("Loading 9×9 Exhaustive Pattern Agent...")

        # Smart path resolution
        if db_file is None:
            db_file = get_database_path()

        self.exhaustive_agent = ExhaustiveAgentDB(db_file)
        self.N = 9
        self.K = 4
        self.window_size = 4

    def extract_4x4_windows(self, board: list) -> List[Tuple[int, int, list]]:
        """
        Extract all 4×4 windows from 9×9 board

        Args:
            board: 9×9 board as list of 81 elements

        Returns:
            List of (start_row, start_col, window_board) tuples
        """
        windows = []

        for start_r in range(self.N - self.window_size + 1):  # 0-5
            for start_c in range(self.N - self.window_size + 1):  # 0-5
                window = []
                for r in range(start_r, start_r + self.window_size):
                    for c in range(start_c, start_c + self.window_size):
                        idx = r * self.N + c
                        window.append(board[idx])

                windows.append((start_r, start_c, window))

        return windows

    def evaluate_move_in_windows(self, board: list, move: int, turn: str) -> dict:
        """
        Evaluate a move by checking all 4×4 windows that contain it.
        Uses THREE perspectives:
        1. Full board (actual position)
        2. Only my pieces (opponent stones treated as empty)
        3. Only opponent pieces (my stones treated as empty)

        Args:
            board: 9×9 board
            move: Move position (0-80)
            turn: 'X' or 'O'

        Returns:
            Dict with aggregated scores from all three perspectives
        """
        move_r = move // self.N
        move_c = move % self.N
        opponent = 'O' if turn == 'X' else 'X'

        # Scores for three perspectives
        perspectives = {
            'full': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0},
            'my_pieces': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0},
            'opp_pieces': {'win': 0.0, 'draw': 0.0, 'loss': 0.0, 'count': 0, 'best_count': 0}
        }

        # Check all windows that contain this move
        for start_r in range(max(0, move_r - self.window_size + 1),
                             min(move_r + 1, self.N - self.window_size + 1)):
            for start_c in range(max(0, move_c - self.window_size + 1),
                                 min(move_c + 1, self.N - self.window_size + 1)):
                # Extract window
                window_full = []
                window_my = []
                window_opp = []

                for r in range(start_r, start_r + self.window_size):
                    for c in range(start_c, start_c + self.window_size):
                        idx = r * self.N + c
                        cell = board[idx]

                        # Full board
                        window_full.append(cell)

                        # Only my pieces (opponent becomes empty)
                        if cell == opponent:
                            window_my.append(' ')
                        else:
                            window_my.append(cell)

                        # Only opponent pieces (my pieces become empty)
                        if cell == turn:
                            window_opp.append(' ')
                        else:
                            window_opp.append(cell)

                # Map 9×9 move to 4×4 window position
                window_move_r = move_r - start_r
                window_move_c = move_c - start_c
                window_move = window_move_r * self.window_size + window_move_c

                # Evaluate from all three perspectives
                for perspective, window in [
                    ('full', window_full),
                    ('my_pieces', window_my),
                    ('opp_pieces', window_opp)
                ]:
                    # Query exhaustive database
                    move_probs = self.exhaustive_agent.get_move_probs(window, turn)

                    if str(window_move) in move_probs:
                        probs = move_probs[str(window_move)]
                        perspectives[perspective]['win'] += probs['p_win']
                        perspectives[perspective]['draw'] += probs['p_draw']
                        perspectives[perspective]['loss'] += probs['p_loss']
                        perspectives[perspective]['count'] += 1

                        # Check if this is best move in this window
                        best_move = self.exhaustive_agent.get_move(window, turn)
                        if best_move == window_move:
                            perspectives[perspective]['best_count'] += 1

        # Aggregate results
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

        return result

    def check_immediate_threats(self, board: list, turn: str, legal_moves: list,
                                verbose: bool = False) -> Optional[int]:
        """
        Check immediate tactical threats:
        1. Can we win in one move?
        2. Does the opponent have a one-move win we must block?

        Args:
            board: 9×9 board
            turn: 'X' or 'O'
            legal_moves: List of legal move positions
            verbose: Print debug info

        Returns:
            Critical move if found, otherwise None
        """
        opponent = 'O' if turn == 'X' else 'X'

        # 1. Check if we can win in one move
        for move in legal_moves:
            board[move] = turn
            if check_winner(board, self.N, self.K) == turn:
                board[move] = ' '
                if verbose:
                    print(f"✓ Winning move found: {move}")
                return move
            board[move] = ' '

        # 2. Check if opponent can win (must block)
        for move in legal_moves:
            board[move] = opponent
            if check_winner(board, self.N, self.K) == opponent:
                board[move] = ' '
                if verbose:
                    print(f"✗ Blocking opponent's winning move: {move}")
                return move
            board[move] = ' '

        return None

    def get_move(self, board: list, turn: str, verbose: bool = False) -> Optional[int]:
        """
        Get best move for 9×9 board using exhaustive 4×4 lookup

        Strategy:
        1. Check immediate win/block (highest priority)
        2. Use three-perspective pattern analysis

        Args:
            board: 9×9 board (81 elements)
            turn: 'X' or 'O'
            verbose: Print debug info

        Returns:
            Best move position (0-80)
        """
        legal_moves = [i for i in range(81) if board[i] == ' ']

        if not legal_moves:
            return None

        if len(legal_moves) == 81:
            # First move: center
            return 40  # Center of 9×9

        # PRIORITY 1: Check immediate threats (win or block)
        critical_move = self.check_immediate_threats(board, turn, legal_moves, verbose)
        if critical_move is not None:
            return critical_move

        # Evaluate all legal moves
        move_scores = {}

        for move in legal_moves:
            scores = self.evaluate_move_in_windows(board, move, turn)

            # Scoring using THREE perspectives:
            # 1. Full board - most important (real threats/opportunities)
            # 2. My pieces only - offensive potential
            # 3. Opponent pieces only - defensive awareness
            score = (
                # Full board perspective (highest weight)
                scores['full_avg_win'] * 100.0 +           # true local win rate
                scores['full_best_count'] * 10.0 +         # best move count in full view
                -scores['full_avg_loss'] * 100.0 +         # avoid high loss rate

                # My pieces perspective (offense)
                scores['my_pieces_avg_win'] * 20.0 +       # own connection potential
                scores['my_pieces_best_count'] * 2.0 +     # best moves w.r.t. own patterns

                # Opponent pieces perspective (defense)
                -scores['opp_pieces_avg_win'] * 50.0 +     # penalize strong opponent patterns
                scores['opp_pieces_best_count'] * 5.0 +    # disrupting key opponent positions

                # Coverage bonus
                scores['full_window_count'] * 0.5          # more windows = more robust estimate
            )

            move_scores[move] = {
                'score': score,
                'details': scores
            }

        # Select best move
        best_move = max(move_scores.items(), key=lambda x: x[1]['score'])[0]

        if verbose:
            print(f"\nMove evaluation (top 5) - THREE perspectives:")
            sorted_moves = sorted(move_scores.items(), key=lambda x: -x[1]['score'])[:5]
            for move, data in sorted_moves:
                r, c = move // self.N, move % self.N
                d = data['details']
                print(f"  Pos {move} ({r},{c}): score={data['score']:.2f}")
                print(f"    Full view:      win={d['full_avg_win']:.3f}, "
                      f"best={d['full_best_count']}, windows={d['full_window_count']}")
                print(f"    Self-only view: win={d['my_pieces_avg_win']:.3f}, "
                      f"best={d['my_pieces_best_count']}")
                print(f"    Opp-only view:  win={d['opp_pieces_avg_win']:.3f}, "
                      f"best={d['opp_pieces_best_count']}")

        return best_move


def check_winner(board: list, N: int = 9, K: int = 4) -> Optional[str]:
    """Check if there's a winner"""
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

    # Check rows
    for r in range(N):
        row = [board[r*N + c] for c in range(N)]
        w = check_line(row)
        if w:
            return w

    # Check columns
    for c in range(N):
        col = [board[r*N + c] for r in range(N)]
        w = check_line(col)
        if w:
            return w

    # Check diagonals (top-left to bottom-right)
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

    # Check diagonals (top-right to bottom-left)
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
    """Print 9×9 board"""
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
    Play game: Exhaustive Pattern Agent vs Random

    Args:
        agent: The exhaustive pattern agent
        agent_plays: 'X' or 'O'
        verbose: Print game progress

    Returns:
        Winner: 'X', 'O', or 'draw'
    """
    import random

    board = [' '] * 81
    turn = 'X'

    if verbose:
        print("="*60)
        print(f"Game: Exhaustive Pattern Agent ({agent_plays}) vs Random")
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
                print("Game ended in a draw")
            return 'draw'

        # Get move
        if turn == agent_plays:
            move = agent.get_move(board, turn, verbose=False)
            player_name = "Exhaustive Agent"
        else:
            legal = [i for i in range(81) if board[i] == ' ']
            move = random.choice(legal)
            player_name = "Random"

        # Make move
        board[move] = turn
        move_count += 1

        if verbose:
            r, c = move // 9, move % 9
            print(f"Move {move_count}: {turn} ({player_name}) plays at ({r},{c})")
            print_board(board)

        # Switch turn
        turn = 'O' if turn == 'X' else 'X'


def demo():
    """Demo: Play games"""
    agent = N9ExhaustivePatternAgent()

    print("\n" + "="*60)
    print("Demo: Exhaustive Pattern Agent vs Random (5 games)")
    print("="*60)

    results = {'X': 0, 'O': 0, 'draw': 0}

    for game_num in range(5):
        print(f"\n{'='*60}")
        print(f"Game {game_num + 1}/5")
        print(f"{'='*60}")

        # Alternate who goes first
        agent_plays = 'X' if game_num % 2 == 0 else 'O'
        result = play_game_vs_random(agent, agent_plays, verbose=False)
        results[result] += 1

        print(f"Result: {result}")
        if result == agent_plays:
            print("✓ Exhaustive Agent wins!")
        elif result == 'draw':
            print("= Draw")
        else:
            print("✗ Random wins")

    print("\n" + "="*60)
    print("Final Results (Exhaustive Agent):")
    print("="*60)
    print(f"Wins:  {results.get('X', 0) + results.get('O', 0) - results.get('draw', 0)} "
          f"(counting agent color)")
    print(f"Draws: {results['draw']}")
    print(f"Losses: (check individual game results)")


if __name__ == '__main__':
    demo()
