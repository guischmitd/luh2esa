import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from sklearn.linear_model import LinearRegression


class SimplexRegression:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        n, p = X.shape  # input dimensions
        d = y.shape[1]  # output dimensions

        def objective(b):
            B = b.reshape(d, p)
            y_pred = X @ B.T
            return np.sqrt(np.mean((y - y_pred) ** 2))  # minimise the RMSE

        # column sums = 1
        constraints = [
            {
                "type": "eq",
                "fun": lambda b, j=j: b.reshape(d, p)[:, j].sum() - 1.0
            }
            for j in range(p)
        ]

        # B >= 0
        bounds = [(0, None)] * (d * p)

        B0 = np.full((d, p), 1 / d)

        self._result = minimize(
            objective,
            B0.ravel(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-4,
            options={"maxiter": 100},
            callback=lambda intermediate_result: print(intermediate_result) if self.verbose else None
        )

        self.B = self._result.x.reshape(d, p)
        return self

    def predict(self, X):
        return X @ self.B.T
    

class MultiOutputOLS():
    def __init__(self):
        self.models = []
    
    def fit(self, X, y):
        assert y.ndim == 2, f'y must be a 2d matrix or dataframe, received {type(y)} with shape {y.shape}'
        if isinstance(y, pd.DataFrame):
            y = y.copy().to_numpy()

        for col_idx in range(y.shape[1]):
            model = LinearRegression(positive=True)
            model.fit(X, y[:, col_idx])
            self.models.append(model)

        return self

    def _process_preds(self, preds: np.ndarray, eps=1e-12):
        preds = preds.clip(eps, 1.0)
        preds = preds / preds.sum(axis=1).reshape(-1, 1)
        return preds

    def predict(self, X):
        raw_preds = []
        for m in self.models:
            raw_preds.append(m.predict(X).reshape(-1, 1))
        
        raw_preds = np.hstack([
            np.array(preds) for preds in raw_preds
        ])

        return self._process_preds(raw_preds)