import json
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from make_data import ROOT


def calibrated(p, temperature):
    return softmax(np.log(np.clip(p,1e-12,1))/temperature, axis=1)


def train():
    train, val, test = [np.load(ROOT/(key+".npz")) for key in ("train","val","test")]
    results = {}
    for name, model in [("logistic",LogisticRegression(max_iter=250)),
                        ("mlp",MLPClassifier((96,),max_iter=24,random_state=374,early_stopping=True))]:
        model.fit(train["x"].astype("float32")/255,train["y"])
        pv = model.predict_proba(val["x"].astype("float32")/255)
        temp = minimize_scalar(lambda t: log_loss(val["y"],calibrated(pv,t)),
                               bounds=(.25,4),method="bounded").x
        p = calibrated(model.predict_proba(test["x"].astype("float32")/255),temp)
        pred = p.argmax(1)+1
        confidence = p.max(1)
        correct = pred == test["y"]
        ece = 0.
        for group in np.array_split(np.argsort(confidence),15):
            ece += len(group)/len(pred)*abs(confidence[group].mean()-correct[group].mean())
        results[name] = dict(accuracy=accuracy_score(test["y"],pred),ece=ece,temperature=temp,
                             confusion=confusion_matrix(test["y"],pred).tolist())
        if name == "mlp":
            joblib.dump((model,temp),ROOT/"model.joblib")
            np.save(ROOT/"probabilities.npy",p)
    (ROOT/"metrics.json").write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    train()
