"""Best-first search over OCR alternatives, with canonical edit sets."""
import heapq
import math
from itertools import count
from csp import PEERS, solve


def repair(grid, probs, method="ucs", budget=3):
    options = {}
    for i, digit in enumerate(grid):
        if digit:
            row = probs[i]
            alternatives = sorted(range(1, 10), key=lambda d: -row[d-1])[:3]
            options[i] = {d: max(0., math.log(max(row[digit-1], 1e-12) /
                                              max(row[d-1], 1e-12)))
                          for d in alternatives if d != digit}
    serial = count()
    frontier = [(0., next(serial), 0., ())]
    seen = {()}
    trace = []
    while frontier:
        _, _, cost, edits = heapq.heappop(frontier)
        board = list(grid)
        edited = dict(edits)
        for i, d in edits:
            board[i] = d
        conflict = next(((i, j) for i in options for j in PEERS[i]
                         if j > i and board[j] == board[i]), None)
        trace.append({"cost": cost, "edits": edits, "conflict": conflict})
        if conflict is None:
            solutions, _ = solve(board)
            if solutions:
                break
        if len(edits) >= budget:
            continue
        for i in conflict or options:
            if i in edited:
                continue
            for d, penalty in options[i].items():
                child = tuple(sorted((*edits, (i, d))))
                if child in seen:
                    continue
                seen.add(child)
                g = cost + penalty
                board[i] = d
                pair = next(((a, b) for a in options for b in PEERS[a]
                             if b > a and board[b] == board[a]), ())
                fixed = dict(child)
                bound = min((v for a in pair if a not in fixed
                             for v in options[a].values()), default=math.inf if pair else 0.)
                board[i] = grid[i]
                priority = {"ucs": g, "astar": g + bound, "greedy": bound}[method]
                heapq.heappush(frontier, (priority, next(serial), g, child))
    else:
        return None, [], trace, len(seen)
    log = [(i, grid[i], d, options[i][d]) for i, d in edits]
    return solutions[0], log, trace, len(seen)
