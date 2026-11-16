"""Render the report and presentation directly from measured outputs."""
import json
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from make_data import ROOT


def render():
    metrics = json.loads((ROOT/"metrics.json").read_text())
    pages = [
        ("Handwritten Sudoku reader and solver",
         "Goal: recover the original Sudoku solution despite mistaken digit readings.\n"
         "Each pipeline stage exposes its decision: probability, correction cost, feasibility.", "demo.png"),
        ("MNIST data and experimental design",
         "Split the original training set into 50,000 training and 10,000 validation images before filtering zeros.\n"
         "Reserve the official test partition for evaluation. Generate unique puzzles by clue removal.\n"
         "Compose centered 28×28 test digits into a 9×9 mosaic. Detect blanks by peak brightness.\n"
         "Choose actual misclassified test images to inject controlled errors.\n"
         "Use 20 puzzles per seed and five seeds, pairing methods on the same inputs.", None),
        ("Digit classification and calibration",
         "MLP: 96 hidden units, early stopping. Baseline: multinomial logistic regression.\n"
         "Fit temperature on validation data by minimizing negative log likelihood.\n"+
         "\n".join(f"{k}: accuracy {v['accuracy']:.3f}, calibration error {v['ece']:.3f}, T {v['temperature']:.2f}"
                   for k,v in metrics.items()), "confusion"),
        ("Searching for corrections",
         "One edit changes one OCR reading to a top-three alternative.\n"
         "Edge cost = log(original probability / alternative probability). Maximum three edits.\n"
         "UCS sorts by total cost. Greedy sorts by the lower bound alone. A* adds them.\n"
         "A duplicate requires changing an endpoint; its cheapest eligible correction bounds the cost.", "ocr-expanded.png"),
        ("Constraint propagation",
         "Variables: 81 cells. Domains: digits 1–9. Neighbors share a row, column or box.\n"
         "Backtracking uses minimum remaining values. Forward checking prunes assigned values.\n"
         "Initial AC-3 removes unsupported values and revisits affected arcs.", "csp-backtracks.png"),
        ("Exact solution recovery",
         "Compare the returned 81 digits against the generator's full solution.\n"
         "Report seed means and sample standard deviations in summary.csv.", "ocr-recovery.png"),
        ("Conclusions and limits",
         "A consistent Sudoku is accepted by search; exact recovery separately measures correctness.\n"
         "Three edits cannot restore four wrong givens. The true alternative can fall outside the search domain.\n"
         "Synthetic aligned mosaics omit lighting, perspective and grid detection errors.\n"
         "Controlled error levels describe solver robustness. Natural OCR accuracy is measured separately.", None)]
    results = pd.read_csv(ROOT/"trials.csv")
    comparison = results.query("exp == 'ocr' and level == 2").groupby("method").recovery.mean()
    pages[-1] = (pages[-1][0],pages[-1][1]+"\n\nTwo-error recovery: "+
                 ", ".join(f"{k} {v:.1%}" for k,v in comparison.items()),None)
    for filename, size in [("report.pdf",(11.7,8.3)),("presentation.pdf",(13.3,7.5))]:
        with PdfPages(ROOT/filename) as pdf:
            for title, body, visual in pages:
                fig = plt.figure(figsize=size)
                fig.text(.06,.90,title,fontsize=22,weight="bold")
                fig.text(.06,.82,body,fontsize=11,va="top",linespacing=1.5)
                if visual:
                    ax = fig.add_axes([.08,.05,.84,.55])
                    if visual=="confusion":
                        ax.imshow(metrics["mlp"]["confusion"],cmap="Blues")
                        ax.set(xticks=range(9),xticklabels=range(1,10),yticks=range(9),
                               yticklabels=range(1,10),xlabel="Predicted digit",ylabel="True digit")
                    else:
                        ax.imshow(plt.imread(ROOT/visual))
                        ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
