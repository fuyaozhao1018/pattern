"""
Convert n4_dual exhaustive JSON to SQLite database
"""

import json
import sqlite3
import time


def convert_dual_to_db(json_file: str, db_file: str):
    """
    Convert n4_dual JSON to SQLite database
    
    Args:
        json_file: Path to n4_exhaustive_probs.json
        db_file: Path to output SQLite database
    """
    print(f"Creating database {db_file} from {json_file}...")
    
    # Create database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS states (
            board TEXT NOT NULL,
            turn TEXT NOT NULL,
            per_move TEXT,
            best_move INTEGER,
            probability_map TEXT,
            PRIMARY KEY (board, turn)
        )
    ''')
    
    # Create index for faster lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_board_turn ON states(board, turn)')
    
    print("Reading JSON file...")
    start = time.time()
    
    count = 0
    batch = []
    batch_size = 10000
    
    with open(json_file, 'r') as f:
        # Skip opening bracket
        line = f.readline()
        if line.strip() != '[':
            raise ValueError("Expected JSON array")
        
        for line in f:
            line = line.strip()
            if line in [']', '']:
                continue
            
            # Remove trailing comma
            if line.endswith(','):
                line = line[:-1]
            
            try:
                state = json.loads(line)
            except:
                continue
            
            # Extract data
            state_id = state.get('id', '')
            board_list = state.get('board', [])
            board = ''.join(board_list)  # Convert list to string
            turn = state.get('turn', 'X')
            
            # Get per_move data
            per_move_data = state.get('per_move', {})
            per_move = json.dumps(per_move_data)
            
            # Calculate best move (highest win rate)
            best_move = None
            best_score = -1
            
            for move_str, stats in per_move_data.items():
                move = int(move_str)
                wins = stats.get('wins', 0)
                draws = stats.get('draws', 0)
                losses = stats.get('losses', 0)
                total = wins + draws + losses
                
                if total > 0:
                    # Simple win rate
                    score = wins / total
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            # Create probability map (optional, can be empty)
            probability_map = json.dumps({})
            
            batch.append((
                board,
                turn,
                per_move,
                best_move,
                probability_map
            ))
            
            count += 1
            
            if len(batch) >= batch_size:
                cursor.executemany(
                    'INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)',
                    batch
                )
                batch = []
                
                if count % 100000 == 0:
                    conn.commit()
                    elapsed = time.time() - start
                    rate = count / elapsed
                    print(f"  Processed {count:,} states ({rate:.0f} states/s)")
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                'INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)',
                batch
            )
    
    conn.commit()
    elapsed = time.time() - start
    
    print(f"\nDatabase created successfully!")
    print(f"Total states: {count:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Rate: {count/elapsed:.0f} states/s")
    
    # Get database size
    cursor.execute("SELECT COUNT(*) FROM states")
    db_count = cursor.fetchone()[0]
    print(f"Database contains: {db_count:,} states")
    
    conn.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        db_file = sys.argv[2] if len(sys.argv) > 2 else 'data/n4_dual/n4_exhaustive_dual.db'
    else:
        json_file = 'data/n4_dual/n4_exhaustive_probs.json'
        db_file = 'data/n4_dual/n4_exhaustive_dual.db'
    
    print("=" * 60)
    print("N4 Dual Exhaustive JSON → SQLite Database Converter")
    print("=" * 60)
    
    convert_dual_to_db(json_file, db_file)
