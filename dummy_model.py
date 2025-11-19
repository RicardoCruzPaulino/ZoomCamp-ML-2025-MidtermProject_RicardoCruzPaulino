import numpy as np

class DummyModel:
    """A simple dummy model with a predict_proba method compatible with the app."""
    def predict_proba(self, X):
        # return a fixed probability for the positive class for every input row
        n = len(X)
        return np.array([[0.25, 0.75] for _ in range(n)])
