"""Paired experiments: 20 puzzles per seed x five seeds per error level."""
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
from integration import ReaderExperiment
from make_data import ROOT, puzzle
from search import repair
from csp import solve


def evaluate():
    reader = ReaderExperiment()
    rows = []
    for seed in range(5):
        for k in range(20):
            key = 374+seed*100+k
            for givens in (25,35,30):
                clues, truth = puzzle(key,givens)
                for fc,ac in [(False,False),(True,False),(True,True)]:
                    ans, counts = solve(clues,fc,ac)
                    rows.append(dict(exp="csp",method=f"BT FC={fc} AC={ac}",level=givens,
                                     seed=seed,puzzle=k,recovery=int(ans==[truth]),**vars(counts)))
            for level in range(5):
                grid,p,_ = reader.sample(clues,key,level)
                for method in ("ucs","greedy","astar","random","none"):
                    if method=="none":
                        ans = solve(grid)[0]
                        result = ans[0] if ans else None
                        trace, generated = [],0
                    else:
                        weights = p.copy()
                        if method=="random":
                            rng = random.Random(key)
                            for i,d in enumerate(grid):
                                if d:
                                    # Keep the OCR reading, randomize alternative ranking.
                                    scores = [rng.random() for _ in range(9)]
                                    scores[d-1] = 1.
                                    weights[i] = scores
                        result,_,trace,generated = repair(grid,weights,method if method!="random" else "ucs")
                    rows.append(dict(exp="ocr",method=method,level=level,seed=seed,puzzle=k,
                                     recovery=int(result==truth),expanded=len(trace),generated=generated))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/"trials.csv",index=False)
    means = df.drop(columns="puzzle").groupby(["exp","method","level","seed"]).mean()
    summary = means.groupby(level=[0,1,2]).agg(["mean","std"])
    summary.to_csv(ROOT/"summary.csv")
    for exp,metric in [("ocr","recovery"),("ocr","expanded"),("csp","backtracks")]:
        data = means.loc[exp,metric].groupby(level=[0,1]).mean().unstack(0)
        data.plot(marker="o",ylabel=metric)
        plt.tight_layout()
        plt.savefig(ROOT/f"{exp}-{metric}.png")
        plt.close()


if __name__=="__main__":
    evaluate()
