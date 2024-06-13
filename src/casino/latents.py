import logging

try:
    import numpy as np
except:
    logging.debug("numpy not availble. Most functionality in latents.py will break")


class SVDProjector:
    def __init__(self, X: "np.ndarray", info: bool = True):
        self.X_mean = X.mean(axis=0)
        self.U, self.S, self.Vh = np.linalg.svd(X - self.X_mean)
        relative_S = self.S / np.sum(self.S)
        if info:
            print(f"S (normalized): {relative_S}")

    def project_to_lower(self, X: "np.ndarray", dim_: int = 2):
        return (X - self.X_mean) @ self.Vh.T[:, :dim_]
        # return X @ Vh[:dim, :].T

    def project_to_higher(self, X: "np.ndarray", dim_: int = 2):
        return (X @ self.Vh.T[:, :dim_].T) + self.X_mean
        # return X @ Vh[:dim, :]
