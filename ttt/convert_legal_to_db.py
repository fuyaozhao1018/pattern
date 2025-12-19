"""
Convert legal 4x4 exhaustive JSON to SQLite database
"""

import json
import sqlite3
import time
import sys


def convert_legal_json_to_db(states_json: str, best_json: str, db_file: str):
    """
    Convert legal JSON files to SQLite database
    
    Args:
        states_json: Path to n4_exhaustive_states.json (contains board states)
        best_json: Path to n4_exhaustive_best.json (contains best moves)
        db_file: Path to output SQLite database
    """
    print(f"Creating legal database {db_file}...")
    print(f"  States file: {states_json}")
    print(f"  Best moves file: {best_json}")
    
    # Create database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create table (same schema as illegal version for compatibility)
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
    
    print("\n[1/2] Loading best moves...")
    best_moves = {}
    with open(best_json, 'r') as f:
        best_moves = json.load(f)
    print(f"  Loaded {len(best_moves):,} state best moves")
    
    print("\n[2/2] Processing states and inserting to database...")
    start = time.time()
    
    count = 0
    batch = []
    batch_size = 10000
    
    with open(states_json, 'r') as f:
        # Skip opening bracket
        line = f.readline()
        if line.strip() != '[':
            raise ValueError("Expected JSON array")
        
        print("  Processing states line by line...")
        
        for line in f:
            line = line.strip()
            if line in [']', '']:
                continue
            
            # Remove trailing comma
            if line.endswith(','):
                line = line[:-1]
            
            try:
                state_data = json.loads(line)
            except:
                continue
            
            # Get state info
            state_id = state_data.get('id', '')
            board_list = state_data.get('board', [])
            board = ''.join(board_list)  # Convert list to string
            turn = state_data.get('turn', 'X')
            
            # Get best move from best_moves dict
            best_move_list = best_moves.get(state_id, [])
            best_move = best_move_list[0] if best_move_list else None
            
            # Get per_move if exists
            per_move_data = state_data.get('per_move', {})
            per_move = json.dumps(per_move_data)
            
            # For legal version, we might not have probability_map
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
                    print(f"  Progress: {count:,} - {rate:.0f} states/s")
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                'INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)',
                batch
            )
    
    conn.commit()
    elapsed = time.time() - start
    
    print(f"\n✓ Database created successfully!")
    print(f"  Total states: {count:,}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Rate: {count/elapsed:.0f} states/s")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM states")
    db_count = cursor.fetchone()[0]
    print(f"  Database contains: {db_count:,} states")
    
    conn.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        states_file = sys.argv[1]
        best_file = sys.argv[2] if len(sys.argv) > 2 else states_file.replace('_states', '_best')
        db_file = sys.argv[3] if len(sys.argv) > 3 else 'data/n4_legal/n4_exhaustive_legal.db'
    else:
        # Default paths
        states_file = 'data/n4_exhaustive_states.json'
        best_file = 'data/n4_exhaustive_best.json'
        db_file = 'data/n4_legal/n4_exhaustive_legal.db'
    
    print("=" * 60)
    print("Legal 4x4 Exhaustive JSON → SQLite Database Converter")
    print("=" * 60)
    
    convert_legal_json_to_db(states_file, best_file, db_file)
