#!/usr/bin/env python3
"""
Quick test: Strategic Convolution Agent vs Minimax (SmartAgent)
"""
import json
import random
from pathlib import Path
from ttt.n9_strategic_convolution_agent import StrategicConvolutionAgent
from ttt.n9_convolution_agent import SmartAgent9x9, play_game_9x9


def main():
    print("="*80)
    print("Strategic Convolution Agent vs Minimax (SmartAgent)")
    print("="*80)
    
    rng = random.Random(2025)
    
    # Load strategic agent
    strategic_agent = StrategicConvolutionAgent(
        'data/n4_dual/n4_exhaustive_probs.json',
        lambda_draw=0.3,
        rng=rng
    )
    
    # Create smart agent (must-win/must-block + center heuristic)
    smart_agent = SmartAgent9x9(rng=rng)
    
    print("\nTesting 200 games:")
    print("  100 games: Strategic (X) vs Smart (O)")
    print("  100 games: Smart (X) vs Strategic (O)")
    print()
    
    # Track loss games
    loss_games = []
    game_id = 0
    
    # Strategic as X
    wins_as_x = draws_as_x = losses_as_x = 0
    print("Playing as X (first player)...")
    for i in range(100):
        outcome, history = play_game_9x9(strategic_agent, smart_agent, record_history=True)
        game_id += 1
        
        if outcome == 'X':
            wins_as_x += 1
        elif outcome == 'D':
            draws_as_x += 1
        else:
            losses_as_x += 1
            # Record loss
            loss_games.append({
                'game_id': game_id,
                'strategic_agent_played_as': 'X',
                'opponent': 'smart_agent',
                'winner': outcome,
                'outcome': 'loss',
                'lambda_draw': 0.3,
                'history': history
            })
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100 games: W:{wins_as_x} D:{draws_as_x} L:{losses_as_x}")
    
    # Strategic as O
    wins_as_o = draws_as_o = losses_as_o = 0
    print("\nPlaying as O (second player)...")
    for i in range(100):
        outcome, history = play_game_9x9(smart_agent, strategic_agent, record_history=True)
        game_id += 1
        
        if outcome == 'O':
            wins_as_o += 1
        elif outcome == 'D':
            draws_as_o += 1
        else:
            losses_as_o += 1
            # Record loss
            loss_games.append({
                'game_id': game_id,
                'strategic_agent_played_as': 'O',
                'opponent': 'smart_agent',
                'winner': outcome,
                'outcome': 'loss',
                'lambda_draw': 0.3,
                'history': history
            })
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100 games: W:{wins_as_o} D:{draws_as_o} L:{losses_as_o}")
    
    # Summary
    total_wins = wins_as_x + wins_as_o
    total_draws = draws_as_x + draws_as_o
    total_losses = losses_as_x + losses_as_o
    total_games = 200
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"As X (first):  W:{wins_as_x}/100 D:{draws_as_x}/100 L:{losses_as_x}/100 ({wins_as_x/100*100:.1f}% WR)")
    print(f"As O (second): W:{wins_as_o}/100 D:{draws_as_o}/100 L:{losses_as_o}/100 ({wins_as_o/100*100:.1f}% WR)")
    print("-"*80)
    print(f"Total:         W:{total_wins}/200 D:{total_draws}/200 L:{total_losses}/200 ({total_wins/200*100:.1f}% WR)")
    print("="*80)
    
    # Save loss games
    if loss_games:
        out_dir = Path('out/runs')
        out_dir.mkdir(parents=True, exist_ok=True)
        loss_file = out_dir / 'n9_strategic_vs_smart_losses.json'
        
        loss_data = {
            'lambda_draw': 0.3,
            'total_losses': len(loss_games),
            'opponent': 'smart_agent',
            'loss_games': loss_games
        }
        
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        print(f"\nLoss games saved to: {loss_file}")
        print(f"Total losses: {len(loss_games)}")
if __name__ == '__main__':
    main()
