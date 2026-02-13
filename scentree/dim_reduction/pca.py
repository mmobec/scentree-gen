import numpy as np
from pydantic import BaseModel, PrivateAttr
from sklearn.decomposition import PCA
from typing import Optional, TypeVar

R = TypeVar("R", bound="BasePCA")


class BasePCA(BaseModel):
    """
    PCA wrapper around scikit-learn's PCA implementation.

    This class encapsulates a scikit-learn PCA reducer and manages
    it as an internal (non-validated, non-serialized) attribute.
    """

    _reducer: Optional[PCA] = PrivateAttr(default=None)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to a given matrix.

        Args:
            X (np.ndarray): Data to be reduced.

        Raises:
            RuntimeError: If the method has not been fitted previously
                (i.e., `self._reducer` is None).

        Returns:
            np.ndarray: Data in the low-dimensional space.
        """
        if self._reducer is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._reducer.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to obtain data in the high-dimensional space.

        Args:
            X (np.ndarray): Data in the low-dimensional space.

        Raises:
            RuntimeError: If the method has not been fitted previously
                (i.e., `self._reducer` is None).

        Returns:
            np.ndarray: Data in the low-dimensional space.
        """
        if self._reducer is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_rec = self._reducer.inverse_transform(X)
        return X_rec

    def fit_auto_components(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Fit a PCA reducer automatically selecting the number of components
        based on the explained variance threshold and transform the input data.

        This method first fits a full PCA to compute the cumulative explained
        variance. It then selects the smallest number of components such that
        the cumulative variance exceeds the specified threshold. Finally, it
        fits a PCA reducer with this number of components and transforms the data.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            threshold (float): Desired cumulative explained variance ratio (between 0 and 1)
                used to select the number of principal components.

        Returns:
            np.ndarray: Transformed input matrix with reduced dimensions based on
                the automatically selected number of components.
        """
        pca_full = PCA()
        pca_full.fit(X)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_th = np.argmax(cumulative_variance >= threshold) + 1
        self._reducer = PCA(n_components=n_components_th)
        X_reduced = self._reducer.fit_transform(X)
        return X_reduced
