"""Typeset the written project report from the saved experiment results."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class Page:
    def __init__(self, pdf, number, title):
        self.pdf = pdf
        self.fig = plt.figure(figsize=(8.27, 11.69))
        self.y = .925
        self.text(.075, .966, "INTELLIGENT SYSTEMS     /     PROJECT REPORT", 9, color="#176260")
        self.text(.075, .043, "Handwritten Sudoku reader and solver", 9)
        self.text(.90, .043, str(number), 9)
        self.text(.075, self.y, title, 21, weight="bold")
        self.y -= .062

    def text(self, x, y, value, size, **kwargs):
        self.fig.text(x, y, value, fontsize=size, va="top", **kwargs)

    def para(self, value):
        lines = textwrap.wrap(value, width=83)
        self.text(.075, self.y, "\n".join(lines), 10.7, linespacing=1.48)
        self.y -= len(lines) * .019 + .019

    def heading(self, title):
        self.text(.075, self.y, title, 13, weight="bold", color="#176260")
        self.y -= .032

    def table(self, headings, rows):
        height = .032 * (len(rows) + 1)
        ax = self.fig.add_axes([.075, self.y-height, .85, height])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headings, cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9.7)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#195659" if r == 0 else ("#e7f0ee" if r % 2 else "#f5f7f6"))
            if r == 0:
                cell.set_text_props(color="white", weight="bold")
        self.y -= height + .022

    def finish(self):
        if self.y < .05:
            raise ValueError(f"Page content exceeds margin: {self.y}")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def render(root=Path("outputs")):
    metrics = json.loads((root / "metrics.json").read_text())
    trials = pd.read_csv(root / "trials.csv")
    means = trials.groupby(["exp", "method", "level", "seed"]).mean(numeric_only=True)
    mlp, baseline = metrics["mlp"], metrics["logistic"]
    cm = np.array(mlp["confusion"])

    def measure(exp, method, level, field, scale=1, digits=1):
        values = means.loc[(exp, method, level), field] * scale
        return f"{values.mean():.{digits}f} ± {values.std():.{digits}f}"

    with PdfPages(root / "report.pdf", metadata={"Title": "Handwritten Sudoku reader and solver"}) as pdf:
        p = Page(pdf, 1, "Handwritten Sudoku reader and solver")
        p.heading("Abstract")
        p.para(f"A single misread clue can make a Sudoku impossible to solve or lead to a different valid solution. We combine a handwritten digit classifier with a search over plausible readings and a constraint solver. On {cm.sum():,} held-out MNIST digits, the neural classifier read {mlp['accuracy']:.2%} correctly. With two deliberately misread clues, A* recovered the original solution on {means.loc[('ocr','astar',2),'recovery'].mean():.0%} of boards. These outcomes measure different tasks: recognizing an isolated digit and reconstructing an entire board.")
        p.heading("Problem and success criterion")
        p.para("The input is an aligned nine-by-nine grid of handwritten clues and blank cells. The output is a completed board where every row, column and three-by-three box contains each digit exactly once. In the experiment, the generator retains the original solution. We count recovery only when all 81 returned digits match it.")
        p.heading("Why keep the alternative readings?")
        p.para("Consider a cell read as 3 that could also be a 5. If a second 3 appears in the same row, Sudoku rules establish a conflict but cannot identify which image was misread. The classifier supplies the relative evidence for each replacement. Search chooses a candidate interpretation, and the solver checks whether that interpretation can be completed.")
        p.heading("Scope")
        p.para("The experiment isolates recognition and reasoning. It uses real handwritten digit images arranged on generated boards. Cell boundaries and orientation are known. The evaluation separately measures digit recognition, recovery under controlled errors and the work performed by the constraint solver.")
        p.finish()

        p = Page(pdf, 2, "Data and digit recognition")
        p.heading("From MNIST files to Sudoku cells")
        p.para("The preparation script downloads the four canonical MNIST image and label archives, verifies their checksums and reads their IDX records. It splits the original training partition by digit label into 50,000 training and 10,000 validation images before removing zeros. The official 10,000-image test partition is kept separate and filtered in the same way. A blank Sudoku cell is represented independently by zero.")
        p.para("Each saved sample has a label y from 1 to 9 and a vector x of 784 grayscale pixels. The three NPZ files retain byte values. At training and inference, pixel values are divided by 255. For each puzzle, the generator removes clues from a solved board and keeps a removal only if the solution remains unique.")
        p.heading("Experiment 1   /   Held-out recognition")
        p.para("Logistic regression gives each digit a linear score. The MLP learns a nonlinear representation with 96 hidden units, early stopping and a maximum of 24 epochs. Both produce nine probabilities. For each classifier, a temperature fitted on validation data minimizes log loss. It rescales confidence while preserving the predicted class.")
        p.table(["Classifier", "Test accuracy", "Calibration error", "Temperature"],
                [[name, f"{s['accuracy']:.2%}", f"{s['ece']:.4f}", f"{s['temperature']:.2f}"] for name,s in [("Logistic regression",baseline),("MLP",mlp)]])
        p.para("Calibration error compares confidence with observed correctness in 15 equally populated groups. Calibration matters here because the repair cost depends on probability ratios. Early stopping helps control overfitting. The saved test metrics alone cannot establish the size of a training-to-test accuracy gap.")
        p.finish()

        p = Page(pdf, 3, "What the recognizer confuses")
        p.para("Rows give the handwritten label and columns give the predicted digit. Each entry counts held-out images. For example, the entry in row 5, column 3 counts images of a 5 read as a 3. Diagonal counts contribute to accuracy. A large off-diagonal count identifies a recurring error worth inspecting.")
        ax = p.fig.add_axes([.19,.36,.63,.43])
        ax.imshow(cm,cmap="Blues")
        ax.set(xticks=range(9),yticks=range(9),xticklabels=range(1,10),yticklabels=range(1,10),xlabel="Predicted digit",ylabel="True digit")
        for (i,j),v in np.ndenumerate(cm):
            ax.text(j,i,str(v),ha="center",va="center",fontsize=10,color="white" if v>cm.max()*.55 else "#142836")
        p.y=.30
        p.heading("Reading the result")
        errors=cm.copy()
        np.fill_diagonal(errors,0)
        i,j=np.unravel_index(errors.argmax(),errors.shape)
        p.para(f"The MLP correctly classified {cm.trace():,} of {cm.sum():,} digits. One of the most frequent errors was reading a {i+1} as a {j+1}, which occurred {cm[i,j]} times. The matrix records only the first choice. It cannot show whether the correct digit was the second choice. The repair experiment therefore retains the full probability vector for every occupied cell.")
        p.finish()

        p = Page(pdf, 4, "Repair search and Sudoku constraints")
        p.heading("Candidate readings and edit costs")
        p.para("Search starts from the highest-probability digit in every occupied cell. A move replaces one reading with another digit among its top three. Each cell can change once, with at most three changed clues. An edit costs log(p_original / p_alternative). Nearly tied readings are inexpensive to swap. Strong visual evidence makes a replacement more costly.")
        p.para("When a pair of clues violates a Sudoku rule, search branches on its two endpoints. A remaining duplicate requires at least one more edit. The cheapest eligible edit at either endpoint gives a lower bound on remaining cost. Uniform-cost search (UCS) ranks states by accumulated cost, greedy search by the bound alone, and A* by their sum. If no duplicate remains, the solver checks whether the puzzle has a completion.")
        p.heading("Constraint model and propagation")
        p.para("The 81 cells are variables with domains drawn from digits 1–9. Cells sharing a row, column or box must differ. Backtracking selects the unfilled cell with the fewest legal values (MRV). Forward checking removes the chosen digit from neighboring domains. Optional AC-3 runs before backtracking and propagates singleton exclusions until the queue is empty or a domain loses every value.")
        p.heading("An integrated correction")
        p.para("In the archived one-error trial for puzzle key 374, UCS changed cell 59 (row 7, column 6) from 3 to 5 at cost 0.0612. Solving the uncorrected clues directly failed. Greedy search found a feasible alternative by changing another clue. This illustrates why the evaluation compares the returned grid with the original solution even after a solver reports success.")
        p.finish()

        p = Page(pdf, 5, "Recovery under controlled OCR errors")
        p.heading("Experiment 2   /   Paired trials")
        p.para("For each of five puzzle seeds, the experiment builds 20 boards with 30 clues. At each error level, all methods receive the same boards and sampled images. Clean cells use correctly read test digits. Injected errors use genuinely misclassified test digits whose correct label remains among the top three. Each entry below is mean exact recovery (%) ± sample standard deviation across the five seed groups.")
        methods=[("UCS","ucs"),("A*","astar"),("Greedy","greedy"),("Random ranking","random"),("Direct CSP","none")]
        p.table(["Misreads"]+[name for name,_ in methods],[[str(n)]+[measure("ocr",m,n,"recovery",100,0) for _,m in methods] for n in range(5)])
        p.heading("How much search was needed?")
        p.para("Expanded states are removed from the priority queue for inspection. Generated states count distinct edit sets discovered, including the start state. At two errors per board, the methods incurred the following mean costs. Direct CSP skips the repair queue entirely.")
        p.table(["Method","Expanded states","Generated states"],[[name,measure("ocr",m,2,"expanded"),measure("ocr",m,2,"generated")] for name,m in methods[:-1]])
        p.para("UCS and A* obtained the same recovery rates at every tested level. A* expanded fewer states on average once errors were introduced. Randomizing the alternatives while retaining the original reading sharply reduced recovery. At four errors, at least one incorrect clue survives the three-edit budget, making exact recovery impossible.")
        p.finish()

        p = Page(pdf, 6, "Constraint study and conclusions")
        p.heading("Experiment 3   /   Clue density and pruning")
        p.para("We supplied correct clues to isolate the CSP from recognition errors. Across five seed groups of 20 boards, all three configurations recovered every board at each tested clue count. The table gives mean backtracks and mean domain values removed per puzzle, with sample standard deviations across seed groups.")
        configs=["BT FC=False AC=False","BT FC=True AC=False","BT FC=True AC=True"]
        p.table(["Clues","Backtracks¹","Pruned (BT)","Pruned (+FC)","Pruned (+FC, AC-3)"],[[str(n),measure("csp",configs[0],n,"backtracks")]+[measure("csp",m,n,"pruned") for m in configs] for n in (25,30,35)])
        p.para("¹ Backtrack counts matched across configurations. Even plain backtracking recomputes legal values against assigned neighbors when applying MRV. Explicit domain removal changed the pruning counter without reducing the measured backtracks on these boards. The higher backtrack count at 25 clues shows how much more guessing those boards required.")
        p.heading("Limits and conclusion")
        p.para("Learned probabilities help search choose which clues to revisit. Sudoku feasibility supplies a check that image recognition alone cannot provide. The results apply to aligned MNIST mosaics and deliberately selected mistakes. Camera perspective, cell segmentation, and errors where the correct digit ranks below third remain outside this evaluation. A field study on photographed boards would test all three limitations.")
        p.heading("Reproducing and auditing the results")
        p.para("Run make_data.py, nn.py and eval.py in that order using the repository environment. The default data and training seed is 374. metrics.json stores test metrics and confusion counts. trials.csv records one method and condition per puzzle, summary.csv aggregates by seed, and corrections.jsonl records proposed edits. run_demo.py --seed 374 --errors 1 displays a worked example. Changing a demo seed selects another puzzle; retraining requires running nn.py.")
        p.finish()


if __name__ == "__main__":
    render()
