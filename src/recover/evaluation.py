from typing import TYPE_CHECKING

from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error
from scipy.special import rel_entr
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from recover.dataset import RasterDataset


def kl_divergence(y_true, y_pred, eps=1e-12):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.copy().to_numpy()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.copy().to_numpy()

    y_pred = np.clip(y_pred, eps, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return np.mean(np.sum(rel_entr(y_true, y_pred), axis=1))


def js_divergence(y_true, y_pred, eps=1e-12):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.copy().to_numpy()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.copy().to_numpy()

    y_true = np.clip(y_true, eps, 1.0)
    y_pred = np.clip(y_pred, eps, 1.0)

    y_true /= y_true.sum(axis=1, keepdims=True)
    y_pred /= y_pred.sum(axis=1, keepdims=True)

    m = 0.5 * (y_true + y_pred)

    js = 0.5 * (
        np.sum(rel_entr(y_true, m), axis=1)
        + np.sum(rel_entr(y_pred, m), axis=1)
    )

    return np.mean(js)


def evaluate(y_true, y_pred, metrics=[kl_divergence, js_divergence, mean_absolute_error], name='result'):
    return pd.Series({m.__name__: m(y_true, y_pred) for m in metrics}, name=name)


class ImagePatchCV(BaseCrossValidator):
    def __init__(self, test_size: float = 0.2, k=5, min_size=5, max_size=30):
        self.test_size = test_size
        self.k = k
        self.min_size = min_size
        self.max_size = max_size

    def split(self, X, y = None, groups = None):
        # Reduce from (T, B, H, W) to (H, W)
        valid_mask = ~np.isnan(X.mean(axis=0).mean(axis=0))
        h, w = valid_mask.shape
        
        train_mask = valid_mask.copy()
        test_mask = np.zeros_like(valid_mask, dtype=bool)

        # Generate 200 random boxes
        box_positions = np.random.randint(0, w - self.min_size, (200, 2))
        box_shapes = np.random.randint(self.min_size, self.max_size, (200, 2))

        for pos, shape in zip(box_positions, box_shapes):
            # Parse a random box
            bx, by = pos
            bw, bh = shape

            x0, x1 = bx, min(bx + bw, w - 1)
            y0, y1 = by, min(by + bh, h - 1)
            
            train_mask[y0:y1, x0:x1] = False
            test_mask[y0:y1, x0:x1] = True
            test_mask *= valid_mask

            if (test_mask.sum() / valid_mask.sum()) >= self.test_size:
                break

        X_train, X_test = np.where(train_mask, X, np.nan), np.where(test_mask, X, np.nan)
        if y is not None:
            y_train, y_test = np.where(train_mask, y, np.nan), np.where(test_mask, y, np.nan)
            return X_train, X_test, y_train, y_test

        return X_train, X_test, None, None


class TimesplitCV(BaseCrossValidator):
    def __init__(self, train_size: int | None, test_size: int, forward: bool = True, k: int | None = None):
        self.forward = forward
        self.train_size = train_size
        self.test_size = test_size
        self.k = k

    def split(self, X, y = None, groups = None):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'X and y must match in dimension 0 (time). X shape: {X.shape} y shape: {y.shape}')
        
        # TODO Finish implementation

        return super().split(X, y, groups)