import unittest
from csp import solve
from search import repair

FULL = [int(c) for c in "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


class SolverTests(unittest.TestCase):
    def test_normal(self):
        for stride in (3,4,5,6,7):
            with self.subTest(stride=stride):
                grid = [0 if i%stride==0 else d for i,d in enumerate(FULL)]
                answer = solve(grid)[0][0]
                self.assertTrue(all(not clue or clue == answer[i] for i,clue in enumerate(grid)))
                self.assertTrue(all(set(answer[i:i+9]) == set(range(1,10)) for i in range(0,81,9)))
                self.assertTrue(all(len(set(answer[i::9])) == 9 for i in range(9)))
                for row in (0,3,6):
                    for col in (0,3,6):
                        self.assertEqual(len({answer[9*(row+r)+col+c] for r in range(3) for c in range(3)}),9)

    def test_full(self):
        self.assertEqual(solve(FULL)[0],[FULL])

    def test_blank(self):
        self.assertEqual(len(solve([0]*81,limit=2)[0]),2)

    def test_last_blank(self):
        self.assertEqual(solve(FULL[:-1]+[0])[0],[FULL])

    def test_impossible(self):
        for a,b in [(0,1),(0,9)]:
            grid = FULL.copy()
            grid[a] = grid[b]
            self.assertFalse(solve(grid)[0],"expected no solution")

    def test_known_correction(self):
        p = [[.001]*9 for _ in FULL]
        grid = FULL.copy()
        grid[0] = 3
        for i,d in enumerate(grid):
            p[i][d-1] = .99
        p[0][2],p[0][4] = .51,.48
        for method in ("ucs","astar","greedy"):
            ans,log,_,_ = repair(grid,p,method,budget=1)
            self.assertEqual(ans,FULL)
            self.assertEqual(log[0][:3],(0,3,5))
        self.assertIsNone(repair(grid,p,budget=0)[0])
