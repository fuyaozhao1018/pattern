import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

X, O, EMPTY = 1, -1, 0

class TicTacToe:
    """
    Generalized Tic-Tac-Toe board.
    Supports arbitrary board size N and win length K (<= N).
    Board cells:  1 (X), -1 (O), 0 (empty)
    """
    def _init_(self, size: int = 3, k: Optional[int] = None):
        if size < 3:
            raise ValueError("size must be >= 3")
        self.size = size
        self.k = k if k is not None else size  # default: N-in-a-row to win
        if not (1 < self.k <= self.size):
            raise ValueError("k must satisfy 2 <= k <= size")
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = X  # X starts

    def copy(self) -> "TicTacToe":
        new = TicTacToe(self.size, self.k)
        new.board = self.board.copy()
        new.current_player = self.current_player
        return new

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """All empty cells as (row, col)."""
        rows, cols = np.where(self.board == EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    def make_move(self, r: int, c: int) -> bool:
        if self.board[r, c] != EMPTY:
            return False
        self.board[r, c] = self.current_player
        self.current_player *= -1
        return True

    def undo_move(self, r: int, c: int) -> None:
        if self.board[r, c] == EMPTY:
            return
        self.board[r, c] = EMPTY
        self.current_player *= -1

    def get_board_hash(self) -> Tuple[int, ...]:
        return tuple(self.board.flatten().tolist())

    # ---- win detection ----
    def _has_k_from(self, r: int, c: int, dr: int, dc: int, player: int) -> bool:
        """Check exactly K in a row starting at (r, c) in direction (dr, dc)."""
        n = self.size
        k = self.k
        rr, cc = r, c
        for _ in range(k):
            if not (0 <= rr < n and 0 <= cc < n):
                return False
            if self.board[rr, cc] != player:
                return False
            rr += dr
            cc += dc
        return True

    def _any_k_in_line(self, player: int) -> bool:
        """Check any K-in-a-row (horizontal, vertical, two diagonals)."""
        n, k = self.size, self.k
        # Horizontal
        for r in range(n):
            for c in range(n - k + 1):
                if self._has_k_from(r, c, 0, 1, player):
                    return True
        # Vertical
        for c in range(n):
            for r in range(n - k + 1):
                if self._has_k_from(r, c, 1, 0, player):
                    return True
        # Diagonal down-right
        for r in range(n - k + 1):
            for c in range(n - k + 1):
                if self._has_k_from(r, c, 1, 1, player):
                    return True
        # Diagonal down-left
        for r in range(n - k + 1):
            for c in range(k - 1, n):
                if self._has_k_from(r, c, 1, -1, player):
                    return True
        return False

    def check_winner(self) -> Optional[int]:
        """Return 1 if X wins, -1 if O wins, 0 draw, None otherwise."""
        if self._any_k_in_line(X):
            return X
        if self._any_k_in_line(O):
            return O
        if not self.get_valid_moves():
            return 0
        return None

    def is_terminal(self) -> bool:
        return self.check_winner() is not None

    def display(self) -> None:
        symbols = {X: 'X', O: 'O', EMPTY: '_'}
        print("\n".join(" ".join(symbols[v] for v in row) for row in self.board))

class MinimaxSolver:
    """
    Minimax with optional alpha-beta pruning and depth limit (for larger N).
    Scores are from X's perspective:
      +infty good for X, -infty good for O.
    Terminal scores favor faster wins / slower losses.
    """
    def _init_(self, game: TicTacToe, depth_limit: Optional[int] = None, use_alpha_beta: bool = True):
        self.game = game
        self.memo = {}
        self.depth_limit = depth_limit
        self.use_alpha_beta = use_alpha_beta

    # Simple heuristic for non-terminal cutoffs: difference of open lines
    def _heuristic(self, g: TicTacToe) -> int:
        n, k = g.size, g.k
        b = g.board
        def count_open(player: int) -> int:
            total = 0
            # horizontal windows
            for r in range(n):
                for c in range(n-k+1):
                    window = b[r, c:c+k]
                    if not np.any(window == -player):
                        total += np.count_nonzero(window == player)
            # vertical
            for c in range(n):
                for r in range(n-k+1):
                    window = b[r:r+k, c]
                    if not np.any(window == -player):
                        total += np.count_nonzero(window == player)
            # diag down-right
            for r in range(n-k+1):
                for c in range(n-k+1):
                    window = np.array([b[r+i, c+i] for i in range(k)])
                    if not np.any(window == -player):
                        total += np.count_nonzero(window == player)
            # diag down-left
            for r in range(n-k+1):
                for c in range(k-1, n):
                    window = np.array([b[r+i, c-i] for i in range(k)])
                    if not np.any(window == -player):
                        total += np.count_nonzero(window == player)
            return total
        return count_open(X) - count_open(O)

    def minimax(self, g: TicTacToe, depth: int = 0, maximizing: bool = True,
                alpha: float = float("-inf"), beta: float = float("inf")) -> int:
        key = (g.get_board_hash(), maximizing)
        if key in self.memo:
            return self.memo[key]

        winner = g.check_winner()
        if winner is not None:
            if winner == X:
                score = 10_000 - depth   # prefer faster wins
            elif winner == O:
                score = depth - 10_000   # prefer slower losses
            else:
                score = 0
            self.memo[key] = score
            return score

        if self.depth_limit is not None and depth >= self.depth_limit:
            score = self._heuristic(g)
            self.memo[key] = score
            return score

        moves = g.get_valid_moves()
        if maximizing:
            best = float("-inf")
            for r, c in moves:
                g.make_move(r, c)
                val = self.minimax(g, depth + 1, False, alpha, beta)
                g.undo_move(r, c)
                if val > best:
                    best = val
                if self.use_alpha_beta:
                    if val > alpha: alpha = val
                    if beta <= alpha:
                        break
            self.memo[key] = int(best)
            return int(best)
        else:
            best = float("inf")
            for r, c in moves:
                g.make_move(r, c)
                val = self.minimax(g, depth + 1, True, alpha, beta)
                g.undo_move(r, c)
                if val < best:
                    best = val
                if self.use_alpha_beta:
                    if val < beta: beta = val
                    if beta <= alpha:
                        break
            self.memo[key] = int(best)
            return int(best)

    def best_move(self, g: TicTacToe) -> Optional[Tuple[int, int]]:
        moves = g.get_valid_moves()
        if not moves:
            return None
        maximizing = (g.current_player == X)
        best_val = float("-inf") if maximizing else float("inf")
        best_mv = None
        for r, c in moves:
            g.make_move(r, c)
            val = self.minimax(g, 1, not maximizing)
            g.undo_move(r, c)
            if maximizing and val > best_val:
                best_val, best_mv = val, (r, c)
            if (not maximizing) and val < best_val:
                best_val, best_mv = val, (r, c)
        return best_mv

def demo(n: int, k: Optional[int] = None, depth_limit: Optional[int] = None):
    print(f"N={n}, K={k if k else n}, depth_limit={depth_limit}")
    g = TicTacToe(n, k)
    solver = MinimaxSolver(g, depth_limit=depth_limit)
    mv = solver.best_move(g)
    print("Best opening move for X:", mv)

if _name_ == "_main_":
    # Examples:
    demo(3)              # classic 3x3, 3-in-a-row
    demo(4)              # 4x4, 4-in-a-row (exact solve may be large)
    demo(5, depth_limit=4)  # 5x5 with a cutoff + heuristic