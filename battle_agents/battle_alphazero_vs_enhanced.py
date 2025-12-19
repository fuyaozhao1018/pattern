#!/usr/bin/env python3
"""
Battle: AlphaZero (trained model) vs Enhanced Pattern Agent

This script loads your trained AlphaZero model and tests it against
the enhanced exhaustive pattern agent.

Usage:
    python ttt/battle_alphazero_vs_enhanced.py --model path/to/model.pth --games 100

Requirements:
    - Trained AlphaZero model (.pth file)
    - 4×4 exhaustive database (data/n4_full/n4_exhaustive.db)
"""
import sys
import os
import argparse
import time
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


import torch
import numpy as np
from battle_agents.enhanced_alphazero_counter import EnhancedAlphaZeroCounter
from battle_agents.common import check_winner, print_board

# Import AlphaZero model components
try:
    from model import NeuralNetwork
    from mcts import MonteCarloTreeSearch
    from value_policy_function import ValuePolicyNetwork
    from game import TicTacToe
except ImportError as e:
    print(f"Error importing AlphaZero components: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class AlphaZeroAgent:
    """Wrapper for trained AlphaZero model"""
    
    def __init__(self, model_path: str, board_size: int = 9, win_length: int = 4, 
                 num_simulations: int = 200):
        """
        Initialize AlphaZero agent with trained model
        
        Args:
            model_path: Path to trained model .pt file
            board_size: Size of board (default 9 for 9×9)
            win_length: Pieces in a row to win (default 4)
            num_simulations: MCTS simulations per move (default 200)
        """
        self.N = board_size
        self.K = win_length
        self.num_simulations = num_simulations
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero using device: {self.device}")
        
        # Initialize game
        self.game = TicTacToe()
        
        # Load model using ValuePolicyNetwork
        print(f"Loading AlphaZero model from {model_path}...")
        try:
            vpn = ValuePolicyNetwork(model_path)
            self.policy_value_network = vpn.get_vp
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        # Initialize MCTS
        self.mcts = MonteCarloTreeSearch(self.game, self.policy_value_network)
        
        print(f"AlphaZero ready (MCTS simulations: {num_simulations})")
    
    def get_move(self, board: list, turn: str, verbose: bool = False) -> int:
        """
        Get AlphaZero's move using MCTS
        
        Args:
            board: Current board state (list of 81 elements, 'X'/'O'/' ')
            turn: Current player ('X' or 'O')
            verbose: Print debug info
            
        Returns:
            Selected move position (0-80)
        """
        # Convert from 'X'/'O'/' ' format to 1/-1/0 format (AlphaZero's format)
        state = np.zeros(self.N * self.N, dtype=np.float32)
        for i in range(self.N * self.N):
            if board[i] == 'X':
                state[i] = 1
            elif board[i] == 'O':
                state[i] = -1
            else:
                state[i] = 0
        
        # Determine current player (1 for X, -1 for O)
        player = 1 if turn == 'X' else -1
        
        # Convert to canonical form (current player's perspective)
        canonical_state = state * player
        
        # Run MCTS to get visit counts
        # Create a root node for the current state
        root_node = self.mcts.init_root_node()
        root_node.set_state(canonical_state)
        
        # Run simulations
        # IMPORTANT: MCTS expects canonical state (current player is 1)
        # But run_simulation takes 'player' argument to handle perspective switching internally
        # However, looking at mcts.py:
        # root_state = root_node.state
        # next_player = -1 * player
        # value, action_probs = self.policy_value_network(root_state, player)
        
        # If we pass canonical_state (where current player is 1), we should pass player=1 to run_simulation
        # Because canonical_state already reflects the perspective of the current player.
        
        root_node = self.mcts.run_simulation(root_node, num_simulations=self.num_simulations, player=1)
        
        # Get action probabilities from visit counts
        action_probs = np.zeros(self.N * self.N)
        for k, v in root_node.children.items():
            action_probs[k] = v.total_visits_N
            
        visit_counts = action_probs
        
        # Get legal moves
        legal_moves = [i for i in range(self.N * self.N) if board[i] == ' ']
        
        if verbose:
            # Show top 5 moves by visit count
            visit_list = [(i, visit_counts[i]) for i in legal_moves if visit_counts[i] > 0]
            visit_list.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nAlphaZero MCTS top 5 moves:")
            for move, count in visit_list[:5]:
                r, c = move // self.N, move % self.N
                print(f"  Pos {move} ({r},{c}): visits={int(count)}")
        
        # Select move with highest visit count
        best_move = max(legal_moves, key=lambda m: visit_counts[m])
        
        if verbose:
            r, c = best_move // self.N, best_move % self.N
            print(f"🤖 AlphaZero selected: {best_move} ({r},{c}), visits={int(visit_counts[best_move])}")
        
        return best_move


def play_game(alphazero: AlphaZeroAgent, enhanced: EnhancedAlphaZeroCounter,
              alphazero_first: bool, verbose: bool = False) -> str:
    """
    Play one game between AlphaZero and Enhanced agent
    
    Args:
        alphazero: AlphaZero agent
        enhanced: Enhanced pattern agent
        alphazero_first: True if AlphaZero plays first (X), False if second (O)
        verbose: Print game moves
        
    Returns:
        Winner: 'alphazero', 'enhanced', or 'draw'
        Game History: A string detailing the game's progression
    """
    board = [' '] * (alphazero.N * alphazero.N)
    N = alphazero.N
    K = alphazero.K
    game_history = []
    
    # Determine who plays X and O
    if alphazero_first:
        x_player = alphazero
        o_player = enhanced
        x_name = "AlphaZero"
        o_name = "Enhanced"
    else:
        x_player = enhanced
        o_player = alphazero
        x_name = "Enhanced"
        o_name = "AlphaZero"
    
    current_player = 'X'
    move_count = 0
    
    header = f"Game Start: {x_name}=X, {o_name}=O\n{'='*60}\n"
    if verbose:
        print(f"\n{'='*60}")
        print(f"Game Start: {x_name}=X, {o_name}=O")
        print(f"{'='*60}\n")
    game_history.append(header)
    
    while True:
        # Get current agent
        if current_player == 'X':
            agent = x_player
            agent_name = x_name
        else:
            agent = o_player
            agent_name = o_name
        
        # Get move
        move = agent.get_move(board, current_player, verbose=verbose)
        
        # Make move
        board[move] = current_player
        move_count += 1
        
        r, c = move // N, move % N
        move_info = f"{agent_name} ({current_player}) plays: {move} ({r},{c})\n"
        board_str = get_board_string(board, N)
        game_history.append(move_info)
        game_history.append(board_str + "\n")

        if verbose:
            print(f"{agent_name} plays: {move} ({r},{c})")
            print_board(board, N)
            print()
        
        # Check winner
        winner = check_winner(board, N, K)
        if winner:
            winner_name = x_name if winner == 'X' else o_name
            result = f"Winner: {winner_name} ({winner})\n"
            game_history.append(result)
            if verbose:
                print(f"Winner: {winner_name} ({winner})")
            return ('alphazero' if winner_name == 'AlphaZero' else 'enhanced'), "".join(game_history)
        
        # Check for draw
        if move_count == N * N:
            result = "Result: Draw\n"
            game_history.append(result)
            if verbose:
                print("Draw!")
            return 'draw', "".join(game_history)
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'

def get_board_string(board, n):
    """Returns the board as a formatted string."""
    s = "  " + " ".join([f"{i:<2}" for i in range(n)]) + "\n"
    for r in range(n):
        s += f"{r:<2}" + " ".join([f"{board[r*n+c]:<2}" for c in range(n)]) + "\n"
    return s

def main():
    """Main function to run the battle"""
    parser = argparse.ArgumentParser(description="Battle AlphaZero vs Enhanced Pattern Agent")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained AlphaZero model (.pt file, NOT .pth)')
    parser.add_argument('--games', type=int, default=500,
                       help='Number of games to play (default: 500)')
    parser.add_argument('--size', type=int, default=9,
                       help='Board size (default: 9 for 9×9)')
    parser.add_argument('--connect', type=int, default=4,
                       help='Pieces in a row to win (default: 4)')
    parser.add_argument('--mcts-sims', type=int, default=200,
                       help='MCTS simulations per move for AlphaZero (default: 200)')
    parser.add_argument('--db', type=str, default=None,
                       help='Path to exhaustive database (optional, default: data/n4_full/n4_exhaustive.db)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output for debugging')
    
    args = parser.parse_args()
    
    # --- Initialize Agents ---
    print("Initializing agents...")
    
    # AlphaZero agent
    alphazero_agent = AlphaZeroAgent(
        model_path=args.model,
        board_size=args.size,
        win_length=args.connect,
        num_simulations=args.mcts_sims
    )
    print("✓ AlphaZero agent initialized")
    
    # Enhanced pattern agent
    enhanced_agent = EnhancedAlphaZeroCounter(
        board_size=args.size,
        db_file=args.db
    )
    print("✓ Enhanced agent initialized")
    
    # --- Battle Loop ---
    start_time = time.time()
    
    results = {'alphazero': 0, 'enhanced': 0, 'draw': 0}
    won_games_history = []
    lost_games_history = []

    for game_num in range(1, args.games + 1):
        # Alternate who goes first
        alphazero_starts = (game_num % 2 == 1)
        
        print(f"--- Game {game_num}/{args.games} (AlphaZero starts: {alphazero_starts}) ---")
        
        winner, game_history = play_game(alphazero_agent, enhanced_agent, alphazero_starts, verbose=args.verbose)
        
        if winner == 'alphazero':
            results['alphazero'] += 1
            lost_games_history.append(f"--- Enhanced Lost Game #{len(lost_games_history) + 1} ---\n{game_history}")
        elif winner == 'enhanced':
            results['enhanced'] += 1
            won_games_history.append(f"--- Enhanced Won Game #{len(won_games_history) + 1} ---\n{game_history}")
        else:
            results['draw'] += 1
            
        # Print progress
        print(f"Game {game_num} result: {winner.upper()} wins")
        print(f"Current Score -> AlphaZero: {results['alphazero']}, Enhanced: {results['enhanced']}, Draw: {results['draw']}")
        print("-" * 30)

    end_time = time.time()
    
    # --- Save Game Records ---
    output_dir = Path(__file__).parent
    
    # Save all lost games
    if lost_games_history:
        lost_games_file = output_dir / "enhanced_lost_games.txt"
        with open(lost_games_file, 'w') as f:
            f.write("\n\n".join(lost_games_history))
        print(f"\nSaved {len(lost_games_history)} lost games to {lost_games_file}")

    # Save a random sample of won games
    if won_games_history:
        won_games_sample_file = output_dir / "enhanced_won_games_sample.txt"
        sample_size = min(5, len(won_games_history))
        won_sample = random.sample(won_games_history, sample_size)
        with open(won_games_sample_file, 'w') as f:
            f.write("\n\n".join(won_sample))
        print(f"Saved a random sample of {sample_size} won games to {won_games_sample_file}")

    # --- Final Results ---
    print("\n" + "="*40)
    print("           BATTLE RESULTS")
    print("="*40)
    results = {
        'alphazero': 0,
        'enhanced': 0,
        'draw': 0
    }
    alphazero_first_wins = 0
    alphazero_second_wins = 0
    enhanced_first_wins = 0
    enhanced_second_wins = 0
    
    # Statistics
    for i in range(args.games):
        # Alternate who goes first
        alphazero_first = (i % 2 == 0)
        
        # Show verbose output for first 2 games only
        verbose = (i < 2)
        
        # Play game
        winner, _ = play_game(alphazero_agent, enhanced_agent, alphazero_first, verbose)
        
        # Update statistics
        results[winner] += 1
        if winner == 'alphazero':
            if alphazero_first:
                alphazero_first_wins += 1
            else:
                alphazero_second_wins += 1
        elif winner == 'enhanced':
            if not alphazero_first:
                enhanced_first_wins += 1
            else:
                enhanced_second_wins += 1
    
    print(f"  - AlphaZero Wins: {results['alphazero']} ({results['alphazero']/args.games*100:.1f}%)")
    print(f"  - Enhanced Wins: {results['enhanced']} ({results['enhanced']/args.games*100:.1f}%)")
    print(f"  - Draws:         {results['draw']} ({results['draw']/args.games*100:.1f}%)")
    print(f"Total games: {args.games}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("="*40)

if __name__ == "__main__":
    main()
