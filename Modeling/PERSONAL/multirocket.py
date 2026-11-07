"""MultiRocketHydra regressor.

Pipeline regressor concatenating the MultiRocket and Hydra transformers with an XGBoost estimator.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["MultiRocketHydraRegressor"]

import gc
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from aeon.regression import BaseRegressor
from aeon.regression.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.utils.validation import check_n_jobs
import torch
import xgboost as xgb
import optuna


class MultiRocketRegressor(BaseRegressor):
    """MultiRocket-Hydra Regressor.

    Combines Hydra and MultiRocket transformations and trains an XGBoost regressor
    on the concatenated features.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group for the Hydra transform.
    n_groups : int, default=64
        Number of groups per dilation for the Hydra transform.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int or None, default=None
        Random seed.
    xgb_params : dict, default=...
        Parameters for XGBRegressor.
    callback_early_stopping : dict, default=...
        Parameters for early stopping callback.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        n_kernels: int = 10000,
        rocket_transform: str = "rocket",
        max_dilations_per_kernel: int = 32,
        n_features_per_kernel: int = 4,
        estimator=None,
        random_state: int = 77,
        n_jobs: int = 1,
        xgb_params: dict = None,
        callback_early_stopping: dict = None,
        use_optuna: Optional[bool] = False,
        optuna_callback: Optional[optuna.integration.XGBoostPruningCallback] = None,
        eval_set: list = [],
    ):
        """
        Initialize the time series classification model with ROCKET-based feature extraction.

        Parameters
        ----------
        n_kernels : int, optional
            Number of randomly generated ROCKET kernels. Default is 10000.

        rocket_transform : str, optional
            Type of transformation to apply. Common options include "rocket" or "minirocket". Default is "rocket".

        max_dilations_per_kernel : int, optional
            Maximum number of dilations applied per kernel. Default is 32.

        n_features_per_kernel : int, optional
            Number of features extracted per kernel. Default is 4.

        estimator : object, optional
            Machine learning model to be trained on the extracted features (e.g., XGBoost, RandomForest). Default is None.

        random_state : int, optional
            Seed for random number generation to ensure reproducibility. Default is 77.

        n_jobs : int, optional
            Number of parallel jobs to use for computation. Default is 1.

        xgb_params : dict, optional
            Dictionary of hyperparameters for the XGBoost estimator, if used. Default is None.

        callback_early_stopping : dict, optional
            Configuration dictionary for early stopping during model training. Default is None.

        use_opunta : bool, optional
            Whether to use Optuna for hyperparameter optimization. Default is False.

        optuna_callback : optuna.integration.XGBoostPruningCallback, optional
            Callback for Optuna to enable pruning of unpromising trials during optimization. Default is None.
        """

        self.n_kernels = n_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.xgb_params = xgb_params or {
            "objective": "reg:squarederror",
            "eval_metric": ["mape", "rmse", "mae"],
            "device": "cuda",
        }
        self.callback_early_stopping = callback_early_stopping or {
            "rounds": 30,
            "min_delta": 7e-4,
            "save_best": False,
            "maximize": False,
            "data_name": "validation_1",
            "metric_name": "mae",
        }
        self.eval_set = eval_set
        self.use_optuna = use_optuna
        self.optuna_callback = optuna_callback

        super().__init__()

    def set_eval_set(self, eval_set: list = []):
        self.eval_set = eval_set

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        torch.cuda.empty_cache()
        gc.collect()
        self._transformer = MultiRocket(
            n_kernels=self.n_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            n_features_per_kernel=self.n_features_per_kernel,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._scaler = StandardScaler(with_mean=False)

        Xt = self._transformer.fit_transform(X)
        Xt = self._scaler.fit_transform(Xt)

        # Early stopping callback
        if self.callback_early_stopping and self.use_optuna:
            early_stopping = xgb.callback.EarlyStopping(**self.callback_early_stopping)
            try:
                self.regressor = xgb.XGBRegressor(
                    **self.xgb_params, callbacks=[early_stopping, self.optuna_callback]
                )
            except (AttributeError, TypeError) as e:
                raise AttributeError(
                    "Incorrect specification of the XGBoost integrated Pruning callback for optuna!"
                )

        elif self.callback_early_stopping and (not self.use_optuna):
            early_stopping = xgb.callback.EarlyStopping(**self.callback_early_stopping)
            self.regressor = xgb.XGBRegressor(
                **self.xgb_params, callbacks=[early_stopping]
            )
        else:
            self.regressor = xgb.XGBRegressor(**self.xgb_params)
        # Transform eval_set if provided
        if len(self.eval_set) > 0:
            eval_set_transformed = [
                (self._scaler.transform(self._transformer.transform(X_val)), y_val)
                for X_val, y_val in self.eval_set
            ]
            self.regressor.fit(Xt, y, eval_set=eval_set_transformed)
        else:
            import warnings

            warnings.WarningMessage(
                "No eval set speficied, please call estimator.set_eval_set() before fit()"
            )
            self.regressor.fit(Xt, y)
        del eval_set_transformed
        del self.eval_set
        torch.cuda.empty_cache()
        gc.collect()
        return self

    def _predict(self, X) -> np.ndarray:
        torch.cuda.empty_cache()
        gc.collect()
        Xt_mr = self._transformer.transform(X)
        del X
        torch.cuda.empty_cache()
        gc.collect()

        Xt = self._scaler.transform(Xt_mr)
        del Xt_mr
        torch.cuda.empty_cache()
        gc.collect()

        return self.regressor.predict(Xt)
