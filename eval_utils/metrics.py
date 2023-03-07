import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def auc_pytorch(y, out):
    y = y.numpy()
    out = out.numpy()
    if out.shape[1] == 2:
        out = out[:, 1]
    return roc_auc_score(y, out)

def round_acc(y, y_pred):
    labels_pred = np.where(y_pred > 0.5, 1, 0)
    return(y == labels_pred).mean()

def acc(y, y_pred):
    return (y == y_pred).mean()


def mse(y, y_pred):
    return ((y - y_pred)**2).mean()


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, end_epoch_reset=True):
        self.end_epoch_reset = end_epoch_reset
        self.steps = 0
        self.total = 0

    def func(self, *values):
        raise NotImplemented

    def update(self, *values):
        val = self.func(*values)
        self.total += val
        self.steps += 1

    def reset(self):
        self.steps = 0
        self.total = 0

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter(RunningAverage):
    def __init__(self):
        super().__init__()

    def func(self, loss_value):
        return loss_value


class AverageAcc(RunningAverage):
    def __init__(self):
        super().__init__()

    def func(self, preds, y):
        return ((torch.argmax(preds, 1) == y).sum().float() / len(y)).item()


class AverageReg():
    def __init__(self):
        self.total = []

    def func(self, preds, y):
        return ((preds - y)**2).tolist()

    def update(self, *values):
        val = self.func(*values)
        self.total += val

    def reset(self):
        self.total = []

    def __call__(self):
        return np.array(self.total).mean()


class MaskedAverageReg():
    def __init__(self, ignore=-1):
        self.total = []
        self.ignore = ignore

    def func(self, preds, y):
        mask = y != self.ignore
        y = y[mask]
        preds = preds[mask]
        return ((preds - y)**2).tolist()

    def update(self, *values):
        val = self.func(*values)
        self.total += val

    def reset(self):
        self.total = []

    def __call__(self):
        return np.array(self.total).mean()


class DummyMetric(RunningAverage):
    def __init__(self):
        super().__init__()

    def func(self, preds, y):
        return 0


