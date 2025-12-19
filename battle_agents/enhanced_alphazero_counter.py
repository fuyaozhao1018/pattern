"""
Enhanced Agent to Counter AlphaZero-based Connect Five AI

Key Strategies:
1. Immediate threat detection (win/block)
2. Multi-step forcing moves (create unavoidable threats)
3. Pattern-based evaluation with aggressive weights
4. Defensive formation detection
"""

import sys
from typing import Optional, List, Tuple, Dict
from battle_agents.n9_exhaustive_pattern_agent import N9ExhaustivePatternAgent


class EnhancedAlphaZeroCounter(N9ExhaustivePatternAgent):
    """
    Enhanced agent specifically designed to counter AlphaZero-based AI
    
    """
    
    def __init__(self, db_file: str = None, board_size: int = 15):
        """
        Initialize enhanced agent
        
        Args:
            db_file: Path to exhaustive 4x4 database (None = auto-detect)
            board_size: Board size (9, 11, 13, 15, 19 for standard boards)
        """
        super().__init__(db_file)
        self.N = board_size
        self.K = 4  # Connect-4
        
        # Aggressive scoring weights
        self.weights = {
            'immediate_win': 1000000,
            'immediate_block': 500000,
            'forcing_threat': 100000,  # Double threat, open-3
            'defensive_formation': 50000,
            'pattern_full': 100.0,
            'pattern_defense': 80.0,  # Increased defensive weight
            'pattern_offense': 40.0,
            'center_bonus': 5.0,
            'opp_next_win': 300.0,     # penalty weight for opponent best next-win probability
            'trivial_penalty': 2.0,    # penalty per trivial 4x4 window (empty or single stone)
            'edge_penalty': 0.5        # small penalty for edge/corner proximity
        }

    # --- Probability-driven helpers ---
    def _opponent_winning_moves(self, board: list, opponent: str) -> set:
        """Return set of moves where opponent wins immediately if they play now."""
        legal = [i for i, v in enumerate(board) if v == ' ']
        wins = set()
        for mv in legal:
            board[mv] = opponent
            if self._check_winner_at(board, mv, opponent):
                wins.add(mv)
            board[mv] = ' '
        return wins

    def _estimate_opp_next_best_win_prob(self, board_after: list, opponent: str) -> float:
        """
        After we play, estimate opponent's best next-move win probability using the 4x4 DB.
        We take the maximum full_avg_win across opponent's legal replies.
        """
        legal_opp = [i for i, v in enumerate(board_after) if v == ' ']
        if not legal_opp:
            return 0.0
        best = 0.0
        for omv in legal_opp:
            evals = self.evaluate_move_in_windows(board_after, omv, opponent)
            best = max(best, evals.get('full_avg_win', 0.0))
            if best >= 1.0:
                return 1.0
        return best

    def get_move(self, board: list, turn: str, verbose: bool = False) -> int:
        """
        Enhanced move selection with multiple priority levels
        
        Args:
            board: Board as list (size N*N)
            turn: Current player ('X' or 'O')
            verbose: Print detailed analysis
            
        Returns:
            Best move index
        """
        # If it's the very first move on an empty board, always play center
        if all(cell == ' ' for cell in board):
            center = (self.N // 2) * self.N + (self.N // 2)
            if verbose:
                print(f"[Opening] Empty board. {turn} plays center {center}")
            return center

        # Legal moves
        legal_moves = [i for i, cell in enumerate(board) if cell == ' ']
        if not legal_moves:
            return -1

        opponent = 'O' if turn == 'X' else 'X'
        move_scores = self._evaluate_all_moves(board, turn, legal_moves, verbose)
        if not move_scores:
            return legal_moves[0]
        # Resolve best move; break ties by closest to center
        best_score = max(move_scores.values())
        center = (self.N // 2) * self.N + (self.N // 2)
        tied_moves = [m for m, s in move_scores.items() if s == best_score]
        if len(tied_moves) == 1:
            best_move = tied_moves[0]
        else:
            # second priority: pick move with smallest Manhattan distance to center
            cr, cc = divmod(center, self.N)
            def manhattan(idx: int) -> int:
                r, c = divmod(idx, self.N)
                return abs(r - cr) + abs(c - cc)
            best_move = min(tied_moves, key=manhattan)
        if verbose:
            print(f"\n🎯 Selected move (scored): {best_move} (score: {move_scores[best_move]:.2f})")
        return best_move

    def check_immediate_threats(self, board: list, turn: str, K: int = 4) -> Optional[int]:
        """
        Parameterized immediate threat detection with K-in-a-row.
        Priority:
          1) Win-in-1 (make K)
          2) Block opponent win-in-1
          3) Create open-(K-1) (both ends open) or block opponent's
          4) Create open-(K-2) or block opponent's
        Returns a move index or None.
        """
        N = self.N
        opponent = 'O' if turn == 'X' else 'X'
        legal = [i for i, v in enumerate(board) if v == ' ']

        def wins_at(m: int, color: str) -> bool:
            r, c = divmod(m, N)
            dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in dirs:
                cnt = 1
                i = 1
                while i < K:
                    nr, nc = r + i * dr, c + i * dc
                    if 0 <= nr < N and 0 <= nc < N and board[nr * N + nc] == color:
                        cnt += 1
                        i += 1
                    else:
                        break
                i = 1
                while i < K:
                    nr, nc = r - i * dr, c - i * dc
                    if 0 <= nr < N and 0 <= nc < N and board[nr * N + nc] == color:
                        cnt += 1
                        i += 1
                    else:
                        break
                if cnt >= K:
                    return True
            return False

        for m in legal:
            board[m] = turn
            win = wins_at(m, turn)
            board[m] = ' '
            if win:
                return m

        for m in legal:
            board[m] = opponent
            win = wins_at(m, opponent)
            board[m] = ' '
            if win:
                return m

        def at_board_edge(idx: int) -> bool:
            r, c = divmod(idx, N)
            return r == 0 or r == N - 1 or c == 0 or c == N - 1

        def creates_open_run(m: int, color: str, need: int) -> bool:
            r0, c0 = divmod(m, N)
            dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
            L = (K - 1) + need
            for dr, dc in dirs:
                cells = []
                r, c = r0, c0
                while 0 <= r - dr < N and 0 <= c - dc < N:
                    r -= dr
                    c -= dc
                coords = []
                rr, cc = r, c
                while 0 <= rr < N and 0 <= cc < N:
                    coords.append((rr, cc))
                    cells.append(board[rr * N + cc])
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
                    if not any(coords[i + j][0] * N + coords[i + j][1] == m for j in range(L)):
                        continue
                    # Edge dead-end: if the segment touches a border and future extension is impossible,
                    # avoid considering this as a good open run (even if current ends are open).
                    seg_coords = coords[i:i + L]
                    touches_border = any(r == 0 or r == N - 1 or c == 0 or c == N - 1 for r, c in seg_coords)
                    if touches_border:
                        # If need==1 (aiming for K-1), we still require at least one interior empty beyond ends
                        # to allow follow-up placement; otherwise it becomes a trapped 3 at edge.
                        # Check the immediate cells beyond the ends are inside board and empty.
                        left_coord = coords[i - 1] if i - 1 >= 0 else None
                        right_coord = coords[i + L] if i + L < len(coords) else None
                        left_extend_ok = left_coord is not None and board[left_coord[0] * N + left_coord[1]] == ' '
                        right_extend_ok = right_coord is not None and board[right_coord[0] * N + right_coord[1]] == ' '
                        if need == 1 and not (left_extend_ok or right_extend_ok):
                            continue
                        if need == 2 and not (left_extend_ok and right_extend_ok):
                            # For K-2 creation we prefer both sides to be extendable
                            continue
                    if need == 1 and cnt_color >= K - 1 and cnt_empty >= 1:
                        return True
                    if need == 2 and cnt_color >= K - 2 and cnt_empty >= 2:
                        return True
            return False

        def collect_open_segments(color: str, need: int):
            segments = []
            dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
            L = (K - 1) + need
            for r0 in range(N):
                for c0 in range(N):
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
                            cells.append(board[rr * N + cc])
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
                            if need == 1 and not (cnt_color >= K - 1 and cnt_empty >= 1):
                                continue
                            if need == 2 and not (cnt_color >= K - 2 and cnt_empty >= 2):
                                continue
                            seg_coords = coords[i:i + L]
                            segments.append(seg_coords)
            return segments

        def pick_block_for_segments(segments: list) -> Optional[int]:
            legal_set = {i for i, v in enumerate(board) if v == ' '}
            for seg in segments:
                ends = []
                if seg:
                    ends = [seg[0], seg[-1]]
                candidates = []
                for r, c in ends:
                    idx = r * N + c
                    if idx in legal_set:
                        candidates.append(idx)
                if candidates:
                    return candidates[0]
                for r, c in seg:
                    idx = r * N + c
                    if idx in legal_set:
                        return idx
            return None

        # Block opponent open-(K-1) first
        opp_open_k1 = collect_open_segments(opponent, need=1)
        if opp_open_k1 and hasattr(self, 'debug') and self.debug:
            print(f"[Threat] Opponent open-(K-1) segments: {len(opp_open_k1)}")
        blk = pick_block_for_segments(opp_open_k1)
        if blk is not None:
            if hasattr(self, 'debug') and self.debug:
                print(f"[Block] Blocking open-(K-1) at {blk}")
            return blk
        # Then create our open-(K-1)
        for m in legal:
            board[m] = turn
            # Avoid selecting dead-end edge creations where future extension is blocked
            if creates_open_run(m, turn, need=1):
                board[m] = ' '
                # Prefer non-edge moves when multiple are possible
                if not at_board_edge(m):
                    return m
                # If only edge options exist, still return
                edge_candidate = m
                board[m] = ' '
                return edge_candidate
            board[m] = ' '

        # Block opponent open-(K-2)
        opp_open_k2 = collect_open_segments(opponent, need=2)
        blk2 = pick_block_for_segments(opp_open_k2)
        if blk2 is not None:
            if hasattr(self, 'debug') and self.debug:
                print(f"[Block] Blocking open-(K-2) at {blk2}")
            return blk2
        # Then create our open-(K-2)
        for m in legal:
            board[m] = turn
            if creates_open_run(m, turn, need=2):
                board[m] = ' '
                if not at_board_edge(m):
                    return m
                edge_candidate2 = m
                board[m] = ' '
                return edge_candidate2
            board[m] = ' '

        return None

    def _evaluate_all_moves(self, board: list, turn: str, 
                           legal_moves: List[int], verbose: bool = False,
                           opp_immediate: set = None) -> Dict[int, float]:
        """
        Probability-driven unified scoring. All priorities are folded into one score.
        """
        opponent = 'O' if turn == 'X' else 'X'

        scores = {}
        for move in legal_moves:
            # Simulate our move
            board[move] = turn

            # 1) 4×4 evaluation with integrated tactical flags
            window_eval = self.evaluate_move_in_windows(board, move, turn)
            full_avg_win   = window_eval.get('full_avg_win', 0.0)
            full_avg_loss  = window_eval.get('full_avg_loss', 0.0)
            full_best_cnt  = window_eval.get('full_best_count', 0)
            my_avg_win     = window_eval.get('my_pieces_avg_win', 0.0)
            my_best_cnt    = window_eval.get('my_pieces_best_count', 0)
            opp_avg_win    = window_eval.get('opp_pieces_avg_win', 0.0)
            opp_best_cnt   = window_eval.get('opp_pieces_best_count', 0)
            full_win_count = window_eval.get('full_window_count', 0)
            trivial_windows = window_eval.get('trivial_window_count', 0)
            trivial_0 = window_eval.get('trivial_0_count', None)
            trivial_1 = window_eval.get('trivial_1_count', None)
            trivial_2 = window_eval.get('trivial_2_count', None)
            max_window_stones = window_eval.get('max_window_stones', 0)
            immediate_win_flag = window_eval.get('immediate_win', 0)
            opp_has_immediate = window_eval.get('opp_has_immediate', 0)
            blocks_opp_immediate = window_eval.get('blocks_opp_immediate', 0)
            creates_open_k1 = window_eval.get('creates_open_k1', 0)
            creates_open_k2 = window_eval.get('creates_open_k2', 0)
            open_k1_edge_trapped = window_eval.get('open_k1_edge_trapped', 0)
            open_k2_edge_trapped = window_eval.get('open_k2_edge_trapped', 0)
            double_threat_count = window_eval.get('double_threat_count', 0)

            # 2) Immediate win/block via flags
            immediate_win = 1 if immediate_win_flag else 0

            # 4) Opponent best next-win probability after we play
            opp_next_best = self._estimate_opp_next_best_win_prob(board, opponent)

            # 5) Compose score
            s = 0.0
            w = self.weights
            # Hard priorities as big weights
            if immediate_win:
                s += w['immediate_win']
            if opp_has_immediate:
                if blocks_opp_immediate:
                    s += w['immediate_block']
                else:
                    s -= w['immediate_block']

            # Pattern-driven terms
            s += full_avg_win * w['pattern_full']
            s -= full_avg_loss * w['pattern_full']
            s += full_best_cnt * 10.0

            s += my_avg_win * w['pattern_offense']
            s += my_best_cnt * 2.0

            s -= opp_avg_win * w['pattern_defense']
            s += opp_best_cnt * 5.0

            s += full_win_count * 0.5
            # Penalize trivial windows. If we have per-level counts, apply graded penalties:
            # 0-stone > 1-stone > 2-stone. Else fallback to aggregate penalty.
            if trivial_0 is not None:
                s -= (trivial_0 * (w['trivial_penalty'] * 1.5)
                      + trivial_1 * (w['trivial_penalty'] * 1.2)
                      + trivial_2 * (w['trivial_penalty'] * 1.0))
            else:
                s -= trivial_windows * w['trivial_penalty']

            # Forcing threat bonuses using integrated counts
            if double_threat_count >= 2:
                s += (w['forcing_threat'] * 1.2)
            elif double_threat_count == 1:
                s += (w['forcing_threat'] * 0.6)
            # Favor creating open-(K-1)/(K-2) unless edge-trapped
            if creates_open_k1 and not open_k1_edge_trapped:
                s += (w['forcing_threat'] * 0.4)
            if creates_open_k2 and not open_k2_edge_trapped:
                s += (w['forcing_threat'] * 0.2)

            # Penalize lines that allow opponent strong next reply
            s -= opp_next_best * w['opp_next_win']

            # Center bias: bonus on exact center and slight tie-break via distance
            center = (self.N // 2) * self.N + (self.N // 2)
            if move == center:
                s += w['center_bonus']
            # Add tiny preference for being closer to center (second priority for ties)
            cr, cc = divmod(center, self.N)
            r, c = divmod(move, self.N)
            manhattan_dist = abs(r - cr) + abs(c - cc)
            s += -manhattan_dist * 0.0005

            # Small edge/corner penalty: discourage choosing border cells when alternatives exist
            if r == 0 or r == self.N - 1:
                s -= w['edge_penalty']
            if c == 0 or c == self.N - 1:
                s -= w['edge_penalty']

            # Tie-break: prefer moves whose 4×4 windows have more stones (higher tactical density)
            s += max_window_stones * 0.001

            scores[move] = s

            # Undo
            board[move] = ' '

        return scores

    def _check_winner_at(self, board: list, move: int, turn: str) -> bool:
        """Return True if placing `turn` at `move` makes K-in-a-row (K=4)."""
        r, c = move // self.N, move % self.N
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in dirs:
            cnt = 1
            # forward
            for i in range(1, self.K):
                nr, nc = r + i*dr, c + i*dc
                if 0 <= nr < self.N and 0 <= nc < self.N and board[nr*self.N+nc] == turn:
                    cnt += 1
                else:
                    break
            # backward
            for i in range(1, self.K):
                nr, nc = r - i*dr, c - i*dc
                if 0 <= nr < self.N and 0 <= nc < self.N and board[nr*self.N+nc] == turn:
                    cnt += 1
                else:
                    break
            if cnt >= self.K:
                return True
        return False

    def _count_threats(self, board: list, turn: str, move: int) -> int:
        """Count how many immediate winning replies we have after `move`.
        Assumes `board[move]` already set to `turn`."""
        legal = [i for i, v in enumerate(board) if v == ' ']
        cnt = 0
        for nm in legal:
            board[nm] = turn
            if self._check_winner_at(board, nm, turn):
                cnt += 1
            board[nm] = ' '
        return cnt

    # Removed _has_open_4 and _has_open_3 in favor of unified check_immediate_threats


# Demo and testing
if __name__ == '__main__':
    print("=" * 60)
    print("AlphaZero Counter Agent - Enhanced for Connect-5")
    print("=" * 60)
    
    # Test with 15x15 board (standard Gomoku/Connect-5 size)
    agent = EnhancedAlphaZeroCounter(board_size=15)
    
    print(f"\nAgent initialized for {agent.N}x{agent.N} board")
    print(f"Win condition: {agent.K}-in-a-row")
    
    # Test first move
    board = [' '] * (15 * 15)
    move1 = agent.get_move(board, 'X', verbose=True)
    print(f"\nFirst move: {move1}")
    
    # Test second move
    board[move1] = 'X'
    move2 = agent.get_move(board, 'O', verbose=True)
    print(f"\nSecond move: {move2}")
    
    # Test tactical scenario
    print("\n" + "=" * 60)
    print("Testing tactical awareness...")
    print("=" * 60)
    
    board_test = [' '] * (15 * 15)
    # Create a scenario: X has 4 in a row, one more to win
    center = 7
    for i in range(4):
        board_test[center * 15 + (center + i)] = 'X'
    
    print("\nScenario: X has 4 in a row horizontally")
    move_win = agent.get_move(board_test, 'X', verbose=True)
    print(f"X should play: {move_win} (winning move)")
    
    # Test blocking
    board_test2 = [' '] * (15 * 15)
    for i in range(4):
        board_test2[center * 15 + (center + i)] = 'O'
    
    print("\nScenario: O has 4 in a row, X must block")
    move_block = agent.get_move(board_test2, 'X', verbose=True)
    print(f"X should block at: {move_block}")
