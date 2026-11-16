"""Sudoku constraints; zero represents a blank."""
from collections import deque
from dataclasses import dataclass

DIGITS = set(range(1, 10))


def related(i, j):
    return (i // 9 == j // 9 or i % 9 == j % 9 or
            (i // 27, i % 9 // 3) == (j // 27, j % 9 // 3))


PEERS = [sorted(j for j in range(81) if i != j and related(i, j)) for i in range(81)]


@dataclass
class Counts:
    assignments: int = 0
    backtracks: int = 0
    pruned: int = 0


def solve(grid, fc=True, ac=True, limit=1, rng=None):
    if len(grid) != 81 or any(x not in DIGITS | {0} for x in grid):
        raise ValueError("Expected 81 digits in 0..9")
    board = list(grid)
    domains = [{x} if x else DIGITS.copy() for x in board]
    counts = Counts()
    solutions = []
    if ac:
        queue = deque((i, j) for i in range(81) for j in PEERS[i])
        while queue:
            i, j = queue.popleft()
            if len(domains[j]) == 1 and domains[i] & domains[j]:
                domains[i] -= domains[j]
                counts.pruned += 1
                if not domains[i]:
                    return solutions, counts
                queue.extend((k, i) for k in PEERS[i] if k != j)
    if any(board[i] and board[i] == board[j] for i in range(81) for j in PEERS[i]):
        return solutions, counts

    def visit(ds):
        choices = []
        for i, value in enumerate(board):
            if not value:
                legal = ds[i] - {board[j] for j in PEERS[i]}
                if not legal:
                    return
                choices.append((len(legal), i, sorted(legal)))
        if not choices:
            solutions.append(board.copy())
            return
        _, i, values = min(choices)
        if rng is not None:
            rng.shuffle(values)
        for v in values:
            counts.assignments += 1
            board[i] = v
            child = ds.copy()
            feasible = True
            if fc:
                for j in PEERS[i]:
                    if not board[j] and v in child[j]:
                        child[j] = child[j] - {v}
                        counts.pruned += 1
                        feasible &= bool(child[j])
            before = len(solutions)
            if feasible:
                visit(child)
            board[i] = 0
            if len(solutions) >= limit:
                return
            counts.backtracks += int(before == len(solutions))
    visit(domains)
    return solutions, counts
