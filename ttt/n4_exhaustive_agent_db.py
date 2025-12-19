"""
4×4 Exhaustive Lookup Agent (SQLite version)
Uses SQLite database for memory-efficient lookup
"""

import json
import sqlite3
from typing import Tuple, Optional


class ExhaustiveAgentDB:
    """Agent that uses SQLite database for exhaustive lookup"""
    
    def __init__(self, db_file: str = 'data/n4_full/n4_exhaustive.db'):
        """
        Initialize agent with database
        
        Args:
            db_file: Path to the SQLite database file
        """
        print(f"Opening database {db_file}...")
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        
        # Check database size
        self.cursor.execute("SELECT COUNT(*) FROM states")
        count = self.cursor.fetchone()[0]
        print(f"Database loaded: {count:,} states")
    
    def __del__(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def get_move(self, board: list, turn: str) -> Optional[int]:
        """
        Get best move from database
        
        Args:
            board: List of 16 elements, each ' ', 'X', or 'O'
            turn: 'X' or 'O'
            
        Returns:
            Best move index (0-15), or None if no legal moves
        """
        board_str = ''.join(board)
        
        self.cursor.execute(
            'SELECT best_move FROM states WHERE board = ? AND turn = ?',
            (board_str, turn)
        )
        
        result = self.cursor.fetchone()
        if result is None:
            print(f"WARNING: Board state not found in database!")
            print(f"Board: {board_str}, Turn: {turn}")
            return None
        
        best_move = result[0]
        return best_move if best_move is not None else None
    
    def get_move_probs(self, board: list, turn: str) -> dict:
        """
        Get probability distribution over all moves
        
        Args:
            board: List of 16 elements, each ' ', 'X', or 'O'
            turn: 'X' or 'O'
            
        Returns:
            Dict mapping move index to win/draw/loss probabilities
        """
        board_str = ''.join(board)
        
        self.cursor.execute(
            'SELECT per_move FROM states WHERE board = ? AND turn = ?',
            (board_str, turn)
        )
        
        result = self.cursor.fetchone()
        if result is None:
            return {}
        
        return json.loads(result[0])
    
    def get_probability_map(self, board: list, turn: str) -> dict:
        """
        Get probability map for visualization
        
        Args:
            board: List of 16 elements, each ' ', 'X', or 'O'
            turn: 'X' or 'O'
            
        Returns:
            Dict mapping move index to selection probability
        """
        board_str = ''.join(board)
        
        self.cursor.execute(
            'SELECT probability_map FROM states WHERE board = ? AND turn = ?',
            (board_str, turn)
        )
        
        result = self.cursor.fetchone()
        if result is None:
            return {}
        
        return json.loads(result[0])


def print_board(board: list):
    """Print 4×4 board"""
    print("\n  0  1  2  3")
    for r in range(4):
        row_str = f"{r} "
        for c in range(4):
            idx = r * 4 + c
            cell = board[idx]
            row_str += f"{cell if cell != ' ' else '·'}  "
        print(row_str)
    print()


def check_winner(board: list, K: int = 4) -> Optional[str]:
    """Check if there's a winner"""
    N = 4
    
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
        seq = []
        r, c = start_r, 0
        while r < N and c < N:
            seq.append(board[r*N + c])
            r += 1
            c += 1
        w = check_line(seq)
        if w:
            return w
    
    for start_c in range(1, N - K + 1):
        seq = []
        r, c = 0, start_c
        while r < N and c < N:
            seq.append(board[r*N + c])
            r += 1
            c += 1
        w = check_line(seq)
        if w:
            return w
    
    # Check diagonals (top-right to bottom-left)
    for start_r in range(N - K + 1):
        seq = []
        r, c = start_r, N - 1
        while r < N and c >= 0:
            seq.append(board[r*N + c])
            r += 1
            c -= 1
        w = check_line(seq)
        if w:
            return w
    
    for start_c in range(K - 1, N - 1):
        seq = []
        r, c = 0, start_c
        while r < N and c >= 0:
            seq.append(board[r*N + c])
            r += 1
            c -= 1
        w = check_line(seq)
        if w:
            return w
    
    return None


def is_full(board: list) -> bool:
    """Check if board is full"""
    return ' ' not in board


def play_game(agent_x: ExhaustiveAgentDB, agent_o: ExhaustiveAgentDB, 
              verbose: bool = True) -> str:
    """
    Play a game between two agents
    
    Args:
        agent_x: Agent playing X
        agent_o: Agent playing O
        verbose: Print game progress
        
    Returns:
        Winner: 'X', 'O', or 'draw'
    """
    board = [' '] * 16
    turn = 'X'
    
    if verbose:
        print("="*50)
        print("Starting game: Exhaustive Agent vs Exhaustive Agent")
        print("="*50)
        print_board(board)
    
    move_count = 0
    
    while True:
        winner = check_winner(board)
        if winner:
            if verbose:
                print(f"{'='*50}")
                print(f"Winner: {winner}")
                print(f"{'='*50}")
                print_board(board)
            return winner
        
        if is_full(board):
            if verbose:
                print(f"{'='*50}")
                print("Game ended in a draw")
                print(f"{'='*50}")
                print_board(board)
            return 'draw'
        
        # Get move from agent
        agent = agent_x if turn == 'X' else agent_o
        move = agent.get_move(board, turn)
        
        if move is None:
            if verbose:
                print(f"No legal moves for {turn}. Draw.")
            return 'draw'
        
        # Get move probabilities for display
        move_probs = agent.get_move_probs(board, turn)
        
        # Make move
        board[move] = turn
        move_count += 1
        
        if verbose:
            print(f"Move {move_count}: {turn} plays at position {move}")
            if str(move) in move_probs:
                probs = move_probs[str(move)]
                print(f"  Win: {probs['p_win']:.3f}, Draw: {probs['p_draw']:.3f}, Loss: {probs['p_loss']:.3f}")
            print_board(board)
        
        # Switch turn
        turn = 'O' if turn == 'X' else 'X'


def demo():
    """Demo: Play a game"""
    print("Loading exhaustive agent (database version)...")
    agent = ExhaustiveAgentDB()
    
    print("\n" + "="*50)
    print("Playing Exhaustive Agent vs Exhaustive Agent")
    print("="*50)
    
    result = play_game(agent, agent, verbose=True)
    
    print(f"\nFinal result: {result}")
    
    # Show statistics for empty board
    print("\n" + "="*50)
    print("Starting position analysis (empty board):")
    print("="*50)
    
    board = [' '] * 16
    move_probs = agent.get_move_probs(board, 'X')
    prob_map = agent.get_probability_map(board, 'X')
    best_move = agent.get_move(board, 'X')
    
    print(f"\nBest move: {best_move}")
    print(f"\nMove probabilities for best move:")
    
    if str(best_move) in move_probs:
        probs = move_probs[str(best_move)]
        print(f"  Position {best_move}: Win={probs['p_win']:.3f}, Draw={probs['p_draw']:.3f}, Loss={probs['p_loss']:.3f}")
    
    print(f"\nProbability map (top 5 positions):")
    for pos, prob in sorted(prob_map.items(), key=lambda x: -x[1])[:5]:
        print(f"  Position {pos}: {prob:.4f}")
    
    print(f"\nAll moves from empty board:")
    for pos in sorted(move_probs.keys(), key=lambda x: int(x)):
        probs = move_probs[pos]
        print(f"  Pos {pos}: Win={probs['p_win']:.3f}, Draw={probs['p_draw']:.3f}, Loss={probs['p_loss']:.3f}")


if __name__ == '__main__':
    demo()
