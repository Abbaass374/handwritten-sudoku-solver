import argparse
import matplotlib.pyplot as plt
import numpy as np
from integration import ReaderExperiment
from make_data import puzzle, ROOT
from csp import solve, solution_message
from search import repair


def template_read(mosaic):
    """Classify the same pixels with training-set digit means; no neural predictions."""
    with np.load(ROOT/"train.npz") as train:
        x, y = train["x"], train["y"]
    centers = np.stack([x[y == d].mean(axis=0) for d in range(1, 10)]) / 255
    cells = mosaic.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 784)
    distances = ((cells[:, None, :] / 255 - centers) ** 2).sum(axis=2)
    scores = np.exp(-(distances - distances.min(axis=1, keepdims=True)))
    weights = scores / scores.sum(axis=1, keepdims=True)
    return np.where(cells.max(axis=1) > 25, weights.argmax(axis=1) + 1, 0).tolist(), weights


def demo(seed, errors):
    clues, truth = puzzle(seed,30)
    grid, p, mosaic = ReaderExperiment().sample(clues,seed,errors)
    print("Raw OCR", np.reshape(grid,(9,9)))
    print("Without learned correction",solution_message(solve(grid)[0]))
    solution, edits, trace, generated = repair(grid,p)
    off_grid, off_weights = template_read(mosaic)
    off_solution, off_edits, off_trace, off_generated = repair(off_grid, off_weights)
    print("NN off (template OCR, same mosaic):", "solved" if off_solution else "no solution",
          "exact recovery", off_solution == truth, "expanded", len(off_trace),
          "generated", off_generated, "corrections", off_edits)
    print("NN on (MLP OCR): exact recovery", solution == truth)
    print("Search trace",trace[:30],"expanded",len(trace),"generated",generated)
    print("Corrections (zero-based cell, old, new, cost)",edits)
    if solution is None:
        print("no solution: unreadable. Least confident clues:",
              sorted((i for i,d in enumerate(grid) if d),key=lambda i:max(p[i]))[:5])
    else:
        corrected = grid.copy()
        for i,old,new,cost in edits:
            corrected[i] = new
        for fc,ac in [(False,False),(True,False),(True,True)]:
            print("FC",fc,"AC-3",ac,solve(corrected,fc,ac)[1])
        print("Solution",np.reshape(solution,(9,9)),"exact recovery",solution==truth)
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(mosaic,cmap="gray")
    axes[0].set_title("MNIST input")
    for ax, values, title in zip(axes[1:],[grid,solution],["OCR","CSP solution"]):
        ax.set(xlim=(0,9),ylim=(9,0),aspect="equal",title=title)
        for i,d in enumerate(values or []):
            if d:
                ax.text(i%9+.5,i//9+.5,str(d),ha="center",va="center",
                        color="red" if any(i==e[0] for e in edits) else "black")
    for ax in axes:
        ax.axis("off")
    fig.savefig(ROOT/"demo.png",bbox_inches="tight")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=374)
    parser.add_argument("--errors",type=int,choices=range(5),default=1)
    args = parser.parse_args()
    demo(args.seed,args.errors)
