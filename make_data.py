"""Download canonical MNIST IDX shards and generate unique Sudoku puzzles."""
import gzip
import hashlib
import random
import struct
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from sklearn.model_selection import train_test_split
from csp import solve

ROOT = Path("outputs")
ROOT.mkdir(exist_ok=True)
HASHES = ["f68b3c2dcbeaaa9fbdd348bbdeb94873","d53e105ee54ea40749a09fcbcd1e9432",
          "9fb629c4189551a2d022fa330f9573f3","ec29112dd5afa0611ce80d1b7f02629c"]


def prepare(seed=374):
    arrays = []
    for name, digest in zip(["train-images-idx3-ubyte","train-labels-idx1-ubyte",
                             "t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte"], HASHES):
        with urlopen("https://storage.googleapis.com/cvdf-datasets/mnist/"+name+".gz", timeout=120) as f:
            raw = f.read()
        if hashlib.md5(raw).hexdigest() != digest:
            raise ValueError("Invalid MNIST checksum")
        data = gzip.decompress(raw)
        magic, n = struct.unpack(">II", data[:8])
        offset = 16 if magic == 2051 else 8
        arr = np.frombuffer(data[offset:], dtype=np.uint8)
        arrays.append(arr.reshape(n,784) if offset == 16 else arr)
    x,y,xt,yt = arrays
    tr,va = train_test_split(np.arange(len(y)),test_size=10000,stratify=y,random_state=seed)
    for tag, images, labels in [("train",x[tr],y[tr]),("val",x[va],y[va]),("test",xt,yt)]:
        np.savez_compressed(ROOT/(tag+".npz"), x=images[labels>0], y=labels[labels>0])


def puzzle(seed, givens):
    rng = random.Random(seed)
    while True:
        full = solve([0]*81, rng=rng)[0][0]
        clues = full.copy()
        for i in rng.sample(range(81),81):
            clues[i] = 0
            if len(solve(clues, limit=2)[0]) != 1:
                clues[i] = full[i]
            if sum(bool(v) for v in clues) == givens:
                return clues, full


if __name__ == "__main__":
    prepare()
