import numpy as np
from make_data import ROOT, puzzle


class ReaderExperiment:
    def __init__(self):
        test = np.load(ROOT/"test.npz")
        self.x, y = test["x"], test["y"]
        self.p = np.load(ROOT/"probabilities.npy")
        top = self.p.argmax(1)+1
        ranking = self.p.argsort(1)[:,-3:]+1
        self.clean = {}
        self.hard = {}
        for d in range(1,10):
            self.clean[d] = np.flatnonzero((y==d) & (top==d))
            self.hard[d] = np.flatnonzero((y==d) & (top!=d) & (ranking==d).any(1))

    def sample(self, clues, seed, errors):
        rng = np.random.default_rng(seed)
        eligible = [i for i,d in enumerate(clues) if d and len(self.hard[d])]
        hard = set(rng.choice(eligible,errors,replace=False))
        images = np.zeros((81,784),dtype=np.uint8)
        probs = np.full((81,9),1/9)
        for i,d in enumerate(clues):
            if d:
                idx = rng.choice((self.hard if i in hard else self.clean)[d])
                images[i], probs[i] = self.x[idx], self.p[idx]
        grid = np.where(images.max(1)>25,probs.argmax(1)+1,0).tolist()
        mosaic = images.reshape(9,9,28,28).transpose(0,2,1,3).reshape(252,252)
        return grid, probs, mosaic
