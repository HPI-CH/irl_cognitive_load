# Imports 
import copy
import gc
import platform
import sklearn.metrics
import xgboost
import optuna
import numpy as np
import utils
import sklearn
import pandas as pd
from typing import Optional, Any
import os
import pandas as pd 
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.feature_selection import RFE, RFECV
import copy
import gc
import copy
import gc

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC, NuSVC, LinearSVC
from typing import Optional
import statistics
from statistics import mean
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, r2_score
import math
import inspect
import math
import inspect
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from optuna.samplers import TPESampler
import utils


class xgboost:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), trial_name: str='regression_cognitve_load', optuna:bool=True, trials:int=1000, decode_dict=None, init=False) -> None:
        """Initialization of the xgboost class. This class contains the finetuning, training, and testing of/with XGBoost. The model will be validated with the LOOCV (currently a 4-fold CV). \
            The class also contains multiple plot functions to visualize the predictions and the importance of features with the built feature importance module.

        Parameters
        ----------
        data : dict
            contains the cleaned sensor data with the features incl. label
        path : str, optional
            path-part where the model's folder, which will contain the plots, will be generated, by default os.path.abspath(os.getcwd())
        name : str, optional
            name of the use case will be specified in the run_xgb file, e.g., before_finetuning or after_finetuning , by default ''
        name_label : str, optional
            name of the column wihch contains the label in the dataframe, by default 'label'
        num_folds : int, optional
            number of folds in kfold, should be the number of the participants in the dataset, by default 4
        trial_name : str, optional
            name of the use case will be specified in the run_xgb file, e.g., before_finetuning or after_finetuning, by default 'regression_cognitve_load'
        optuna : bool, optional
            wether to finetune the model and then train it, by default True
        """
        
        self.parameters={
                  "objective": 'reg:squarederror',
                  'device':'cuda', 
                  "eval_metric":['mape','rmse', 'mae']
             }
        if 'percent' in trial_name:
            self.parameters={
                  "objective": 'reg:squarederror',
                  'device':'cuda', 
                  "eval_metric":['mape','rmse', 'mae']
             }
        self.early_stopping = {
            'rounds': 75,
            'min_delta':9.917326508203772e-05,
            'save_best':False,
            'maximize':False,
            'data_name':'validation_1',
            'metric_name':'mae'
        }

        if init:
             self.parameters={
                  "objective": 'reg:squarederror',
                  'device':'cuda', 
                  "eval_metric":['mape','rmse', 'mae']
             } 
        self.optuna =optuna
        self.data = data
        self.trails = trials
        self.decode_dict = decode_dict
        trial_name = 'xg_boost_'+trial_name+'_LOOCV'
        path = path +'/'+trial_name
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
     

      
        self.path_save_preds =r''



    def run_anova_feature_selection(
        self,
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Determine the optimal feature subset for XGBoost regression.

        The method performs K-fold cross-validation on `self.data`. Within
        each fold, univariate regression F-tests are calculated using only
        the training data. Features are ranked from the highest to the lowest
        F-score.

        An XGBoost regressor is subsequently trained and evaluated using the
        first 1, 2, 3, up to all ranked features. The feature subset with the
        lowest mean absolute error is selected as the best subset for the
        respective fold.

        If multiple subsets produce the same mean absolute error, the subset
        containing fewer features is preferred.

        Results are stored in the following instance attributes:

        `self.anova_fold_results`
            DataFrame containing the best result for every fold.

        `self.anova_best_features`
            Dictionary mapping each fold to its best feature subset.

        `self.anova_feature_rankings`
            Dictionary mapping each fold to its complete feature ranking.

        `self.anova_score_tables`
            Dictionary containing the F-score and p-value of every feature
            in each fold.

        `self.anova_all_results`
            Dictionary containing the results for every tested feature count
            in each fold.

        `self.anova_best_models`
            Dictionary mapping each fold to the final fitted XGBoost model.

        `self.anova_best_predictions`
            Dictionary containing the true and predicted values of the best
            model in every fold.

        Parameters
        ----------
        n_splits : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        pd.DataFrame
            Fold-level results containing the best number of features,
            selected feature names, and corresponding mean absolute error.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If the cross-validation configuration, features, or targets are
            invalid.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        The regression F-test is a univariate feature-selection method.
        Features are evaluated independently according to their linear
        association with the continuous target. Feature interactions and
        nonlinear associations are not considered during feature ranking.

        The optimal feature count is selected using the test data of the
        respective fold. Therefore, the minimum test MAE is not an independent
        estimate of generalization performance.
        """
        required_attributes = [
            "data",
            "parameters",
            "decode_dict",
            "path"
        ]

        for attribute_name in required_attributes:
            if not hasattr(self, attribute_name):
                raise AttributeError(
                    f"Required instance attribute "
                    f"'self.{attribute_name}' is missing."
                )

        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        if n_splits > len(self.data):
            raise ValueError(
                "n_splits cannot exceed the number of elements "
                "in self.data."
            )


        cross_validator = KFold(
            n_splits=n_splits,
            shuffle=False
        )

        columns_to_drop = [
            "id",
            "task",
            "cat_2",
            "participant",
            "PPG_fixed_peaks",
            "PPG_Hr",
            "EDA_SCR_Peaks",
            "EDA_SCR_Height",
            "EDA_SCR_Recovery",
            "EDA_SCR_Onsets",
            "EDA_SCR_Amplitude",
            "EDA_SCR_RiseTime",
            "HRV_TINN",
            "HRV_SDRMSSD"
        ]

   
        model_parameters = copy.deepcopy(
            self.parameters
        )

       
        model_parameters.pop("svm", None)
        model_parameters.pop("model", None)
        model_parameters.pop("model_type", None)

    
        model_parameters.pop(
            "early_stopping_rounds",
            None
        )

        model_type = "XGBRegressor"

  
        self.anova_fold_results = None
        self.anova_best_features = {}
        self.anova_feature_rankings = {}
        self.anova_score_tables = {}
        self.anova_all_results = {}
        self.anova_best_models = {}
        self.anova_best_predictions = {}

        all_fold_results = []

        print("=" * 70)
        print("ANOVA feature selection for XGBoost regression")
        print(f"Number of folds: {n_splits}")
        print("=" * 70)

  
        for fold, (train_indices, test_indices) in enumerate(
            cross_validator.split(self.data)
        ):
            gc.collect()
            data_copy = copy.deepcopy(self.data)
            train_data = [
                data_copy[index]
                for index in train_indices
            ]

            test_data = [
                data_copy[index]
                for index in test_indices
            ]

            (
                train_set,
                test_set,
                label_train,
                label_test
            ) = utils.loso_datasets_ml(
                train_data=train_data,
                test_data=test_data
            )

       
            test_ids = pd.DataFrame({
                "id": np.asarray(
                    test_set["id"]
                ).ravel()
            })

            decoded_ids = utils.decode_string_columns(
                test_ids,
                decode_dict=self.decode_dict
            )

            participants = (
                decoded_ids["id"]
                .drop_duplicates()
                .tolist()
            )

            print("\n" + "=" * 70)
            print(f"Fold {fold}")
            print(f"Test participants: {participants}")
            print("=" * 70)

            X_train = utils.clean_up_features(
                train_set,
                extra_columns_to_drop=columns_to_drop
            )

            X_test = utils.clean_up_features(
                test_set,
                extra_columns_to_drop=columns_to_drop
            )

            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)

            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)

            if X_train.empty:
                raise ValueError(
                    f"Fold {fold}: X_train contains no features."
                )

          
            missing_test_features = [
                feature
                for feature in X_train.columns
                if feature not in X_test.columns
            ]

            if missing_test_features:
                raise ValueError(
                    f"Fold {fold}: The following training features are "
                    f"missing in X_test: {missing_test_features}."
                )

         
            X_test = X_test.loc[
                :,
                X_train.columns
            ].copy()

            X_train = X_train.copy()

            try:
                X_train = X_train.astype(float)
                X_test = X_test.astype(float)

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Fold {fold}: The feature matrices contain "
                    "non-numeric values."
                ) from error


            try:
                y_train = np.asarray(
                    label_train,
                    dtype=float
                ).ravel()

                y_test = np.asarray(
                    label_test,
                    dtype=float
                ).ravel()

            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Fold {fold}: The target values could not be "
                    "converted to floating-point values."
                ) from error


            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"Fold {fold}: X_train contains "
                    f"{X_train.shape[0]} samples, but y_train contains "
                    f"{y_train.shape[0]} target values."
                )

            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(
                    f"Fold {fold}: X_test contains "
                    f"{X_test.shape[0]} samples, but y_test contains "
                    f"{y_test.shape[0]} target values."
                )

            if X_train.isna().any().any():
                columns_with_nan = X_train.columns[
                    X_train.isna().any()
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: X_train contains missing values in "
                    f"the following columns: {columns_with_nan}."
                )

            if X_test.isna().any().any():
                columns_with_nan = X_test.columns[
                    X_test.isna().any()
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: X_test contains missing values in "
                    f"the following columns: {columns_with_nan}."
                )

            train_values = X_train.to_numpy(
                dtype=float
            )

            test_values = X_test.to_numpy(
                dtype=float
            )

            if not np.isfinite(train_values).all():
                invalid_columns = X_train.columns[
                    ~np.isfinite(train_values).all(axis=0)
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: X_train contains infinite values in "
                    f"the following columns: {invalid_columns}."
                )

            if not np.isfinite(test_values).all():
                invalid_columns = X_test.columns[
                    ~np.isfinite(test_values).all(axis=0)
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: X_test contains infinite values in "
                    f"the following columns: {invalid_columns}."
                )

            if not np.isfinite(y_train).all():
                raise ValueError(
                    f"Fold {fold}: y_train contains missing or "
                    "infinite values."
                )

            if not np.isfinite(y_test).all():
                raise ValueError(
                    f"Fold {fold}: y_test contains missing or "
                    "infinite values."
                )

            if np.unique(y_train).size < 2:
                raise ValueError(
                    f"Fold {fold}: y_train is constant. Regression "
                    "feature selection cannot be performed."
                )

            total_number_of_features = X_train.shape[1]

            print(
                f"Training samples: {X_train.shape[0]}"
            )
            print(
                f"Test samples: {X_test.shape[0]}"
            )
            print(
                f"Available features: {total_number_of_features}"
            )


            print("\nCalculating regression F-scores...")

            with np.errstate(
                divide="ignore",
                invalid="ignore"
            ):
                f_scores, p_values = f_regression(
                    X_train,
                    y_train,
                    center=True,
                    force_finite=True
                )

            anova_score_table = pd.DataFrame({
                "feature": X_train.columns.tolist(),
                "f_score": np.asarray(f_scores, dtype=float),
                "p_value": np.asarray(p_values, dtype=float),
                "original_position": np.arange(
                    total_number_of_features
                )
            })

          
            anova_score_table["f_score"] = (
                anova_score_table["f_score"]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
            )

            anova_score_table["p_value"] = (
                anova_score_table["p_value"]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(1.0)
            )

            anova_score_table = (
                anova_score_table
                .sort_values(
                    by=[
                        "f_score",
                        "original_position"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            anova_score_table.insert(
                0,
                "rank",
                np.arange(
                    1,
                    total_number_of_features + 1
                )
            )

            feature_ranking = (
                anova_score_table["feature"]
                .tolist()
            )

            self.anova_feature_rankings[fold] = (
                feature_ranking.copy()
            )

            self.anova_score_tables[fold] = (
                anova_score_table.copy()
            )

            print("\nFeature ranking:")

            for row in anova_score_table.itertuples(
                index=False
            ):
                print(
                    f"{row.rank:3d}. "
                    f"{row.feature} | "
                    f"F-score: {row.f_score:.4f} | "
                    f"p-value: {row.p_value:.6f}"
                )


            feature_counts = []
            mae_values = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for number_of_features in range(
                1,
                total_number_of_features + 1
            ):
                selected_features = feature_ranking[
                    :number_of_features
                ]

                regressor = xgb.XGBRegressor(
                    **model_parameters
                )

                try:
                    regressor.fit(
                        X_train.loc[:, selected_features],
                        y_train
                     
                )

                    prediction= regressor.predict(X_test.loc[:, selected_features]
                    )

                    current_mae = mean_absolute_error(
                        y_test,
                        prediction
                    )

                except Exception as error:
                    print(
                        f"Evaluation failed for "
                        f"{number_of_features} features."
                    )
                    print(f"Error: {error}")
                    continue

                feature_counts.append(
                    number_of_features
                )

                mae_values.append(
                    float(current_mae)
                )

                selected_feature_sets.append(
                    selected_features.copy()
                )

                print(
                    f"Number of features: "
                    f"{number_of_features:3d} | "
                    f"MAE: {current_mae:.4f}"
                )

            if len(mae_values) == 0:
                raise RuntimeError(
                    f"Fold {fold}: No feature subset could be "
                    "evaluated successfully."
                )

            ordered_result_indices = sorted(
                range(len(mae_values)),
                key=lambda index: (
                    mae_values[index],
                    feature_counts[index]
                )
            )

            best_result_index = ordered_result_indices[0]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_mae = mae_values[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ].copy()

            self.anova_best_features[fold] = (
                best_features.copy()
            )


            best_model = xgb.XGBRegressor(
                **model_parameters
            )

            best_model.fit(
                X_train.loc[
                    :,
                    best_features
                ],
                y_train
            )

            best_predictions = best_model.predict(
                X_test.loc[
                    :,
                    best_features
                ]
            )

            final_mae = mean_absolute_error(
                y_test,
                best_predictions
            )

      
            if not np.isclose(
                best_mae,
                final_mae,
                rtol=1e-6,
                atol=1e-8
            ):
                print(
                    "Warning: The refitted model produced a different "
                    "MAE. This may occur when the estimator uses "
                    "non-deterministic settings."
                )

                best_mae = float(final_mae)

            self.anova_best_models[fold] = (
                best_model
            )

            self.anova_best_predictions[fold] = {
                "participants": participants,
                "true_values": y_test.copy(),
                "predictions": np.asarray(
                    best_predictions
                ).copy()
            }


            fold_evaluation_results = pd.DataFrame({
                "number_of_features": feature_counts,
                "mean_absolute_error": mae_values,
                "selected_features": selected_feature_sets
            })

            fold_evaluation_results = (
                fold_evaluation_results
                .sort_values(
                    by=[
                        "mean_absolute_error",
                        "number_of_features"
                    ],
                    ascending=[
                        True,
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            self.anova_all_results[fold] = (
                fold_evaluation_results.copy()
            )

            fold_result = {
                "fold": fold,
                "participants": participants,
                "model_type": model_type,
                "selection_method": "f_regression",
                "optimization_metric": "mean_absolute_error",
                "number_of_training_samples": X_train.shape[0],
                "number_of_test_samples": X_test.shape[0],
                "total_number_of_features": total_number_of_features,
                "best_number_of_features": best_number_of_features,
                "best_features": best_features,
                "best_mae": float(best_mae),
                "feature_ranking": feature_ranking
            }

            all_fold_results.append(
                fold_result
            )


            print("\n" + "-" * 70)
            print(f"Best result for fold {fold}")
            print("-" * 70)

            print(
                f"Best number of features: "
                f"{best_number_of_features}"
            )

            print(
                f"Best MAE: {best_mae:.4f}"
            )

            print("Best features:")

            for feature_name in best_features:
                print(f"  - {feature_name}")




        self.anova_fold_results = pd.DataFrame(
            all_fold_results
        )

 

        print("\n" + "=" * 70)
        print("Best feature subset per fold")
        print("=" * 70)

        summary_columns = [
            "fold",
            "participants",
            "total_number_of_features",
            "best_number_of_features",
            "best_mae"
        ]

        print(
            self.anova_fold_results[
                summary_columns
            ].to_string(index=False)
        )

        print("\nOverall summary:")

        print(
            f"Mean best MAE: "
            f"{self.anova_fold_results['best_mae'].mean():.4f}"
        )

        if len(self.anova_fold_results) > 1:
            print(
                f"Standard deviation of best MAE: "
                f"{self.anova_fold_results['best_mae'].std():.4f}"
            )

        print(
            f"Mean number of selected features: "
            f"{self.anova_fold_results['best_number_of_features'].mean():.2f}"
        )

        return self.anova_fold_results


    def visualize_results(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int, participant: str) -> None:
            """
            Visualizing the results. This will plot a point plot of the predictions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.

            Parameters
            ----------
            ground_truth : np.array
                True values of the cognitive load.
            pred : np.array
                Predicted values of the cognitive load.
            nombre : str
                Name of the plot, e.g., name of the model.
            compare : pd.DataFrame
                DataFrame of the error metrics, only one row.
            fold : int
                The current fold number.
            participant : str
                Name of the participant.
            """
            
            fig12 = plt.figure(figsize=(40,7))
            colors = ['#E6BFCC', '#E98DAD', '#C74777', '#7A042A', '#4F021B', '#663845', '#8B4E5E']
            compare.plot(kind='bar', color=colors, rot=30)
            plt.xlabel('Error Metrics')
            plt.ylabel('Error')
            plt.title('Error for Fold ' + str(fold) + ' of Participant ' + str(participant), wrap=True)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            name = 'Error_metrics_fold_' + nombre + str(fold) + '_participant_' + str(participant) + '.png'
            plt.savefig(self.path + '/' + name, bbox_inches='tight')
            plt.close(fig12)

            temp = pd.DataFrame({'True label': ground_truth, 'Predicted label': pred})
            fig22 = plt.figure(figsize=(40,7))
            import seaborn as sns
            sns.regplot(data=temp, x='True label', y='Predicted label', marker='o', scatter_kws={'color': '#E98DAD'}, line_kws={'color': '#c74777'})
            plt.xlabel('True Cognitive Load')
            plt.ylabel('Predicted Cognitive Load')
            plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
            name_scatter = 'cl_true_pred_participant_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
            plt.savefig(self.path + '/' + name_scatter)
            plt.close(fig22)

            fig323c = plt.figure(figsize=(40,7))
            sns.scatterplot(data=temp, x=temp.index, y='True label', color='#e84a7f', label='True Cognitive Load', marker='*')
            sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#E98DAD', label='Predicted Cognitive Load', marker='o')
            plt.xlabel('Time windows')
            plt.ylabel('Cognitive Load')
            plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
            name_scatter = 'cl_true_pred_participant_line_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
            plt.savefig(self.path + '/' + name_scatter)
            plt.close(fig323c)
            fig323cc = plt.figure(figsize=(40,7))
        
            sns.lineplot(data=temp['True label'], color='#E84a7f', label='True True Cognitive Load',marker='o' , linestyle = 'dotted')
            sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Cognitive load',marker='o',  linestyle = '')
            
            plt.xlabel('Time windows')
            plt.ylabel('True Cognitive Load')
            plt.title('Cognitve load estimated from psycho-physiological data vs. subjective perception of participant '+str(participant)+ ' from fold '+str(fold), wrap=True)
        
            name_scatter = 'cl_true_pred_participant_with_line_'+str(participant)+'_Fold_'+str(fold)+nombre+'.pdf'
            plt.savefig(self.path+'/'+name_scatter)
            plt.close(fig323cc)



    def visualize_results_frederic(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int, participant: str) -> None:
        """
        Visualizing the results. This will plot a point plot of the predictions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.
        The colors used in the plots are suitable for individuals with red-green color blindness.

        Parameters
        ----------
        ground_truth : np.array
            True values of the cognitive load.
        pred : np.array
            Predicted values of the cognitive load.
        nombre : str
            Name of the plot, e.g., name of the model.
        compare : pd.DataFrame
            DataFrame of the error metrics, only one row.
        fold : int
            The current fold number.
        participant : str
            Name of the participant.
        """
        
        fig12 = plt.figure(figsize=(40,7))
        colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        compare.plot(kind='bar', color=colors, rot=30)
        plt.xlabel('Error Metrics')
        plt.ylabel('Error')
        plt.title('Error for Fold ' + str(fold) + ' of Participant ' + str(participant), wrap=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        name = 'Error_metrics_foldf_' + nombre + str(fold) + '_participant_' + str(participant) + '.pdf'
        plt.savefig(self.path + '/' + name, bbox_inches='tight')
        plt.close(fig12)

        temp = pd.DataFrame({'True label': ground_truth, 'Predicted label': pred})
        fig22 = plt.figure(figsize=(40,7))
        import seaborn as sns
        sns.regplot(data=temp, x='True label', y='Predicted label', marker='o', scatter_kws={'color': '#377eb8'}, line_kws={'color': '#4daf4a'})
        plt.xlabel('True Cognitive Load')
        plt.ylabel('Predicted Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participantf_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#377eb8', label='True Cognitive Load', marker='o')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#4daf4a', label='Predicted Cognitive Load', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)

        fig323cc = plt.figure(figsize=(10,7))
        sns.lineplot(data=temp['True label'], color='#377eb8', label='True Cognitive Load', marker='o',linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Cognitive load',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_with_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323cc)




    def dump_files_for_shap(self, files_list: list, path: str, suffix: str = '') -> None:
            """
            This function will take the variable name and use it as a file name to dump it in pickle files.

            :param files_list: list of variable names which should be dumped, just copy and paste the names
            :param path: path which you want to dump the files
            :param suffix: optional string to be added to the file name
            :return: None
            """
            for file in files_list:
                callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                name = [var_name for var_name, var_val in callers_local_vars if var_val is file]
                if name[0].startswith('_') and name[0].endswith('_') and len(name) > 1:
                    name[0] = name[1]
                with open(f"{path}/{name[0]}{suffix}.pickle", 'wb') as handle:
                    pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)




    # Save all for later
    def dump_files(self,files_list, path):
        """
            This function will take the variable name and use it as a file name to dump it in pickle files.

            :param files_list: list of variable names which should be dumped, just copy and paste the names
            :param path: path which you want to dump the files
            :return: None
        """
        for file in files_list:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            name=[var_name for var_name, var_val in callers_local_vars if var_val is file]
            with open(path+'/' + name[0]+'.pickle', 'wb') as handle: 
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)    

            
    

   
    def final_plot(self,score_pd, path, top_k :Optional[int]=None):
        """Feature Importance. Plots the results of the feature importance tool of the best XGBoost.

        Parameters
        ----------
        score_pd : pd.DataFrame
            Contains the achieved MAE    
        path : Optional[str], optional
            where to save the plot, by default ''
        
        top_k : Optional[int], optional
            number of important features, be careful when changing it, if you have less features it will throw an error, by default 30

        
        """
        average_feature = pd.DataFrame(columns =score_pd.columns)
        name = 'Best_Mean_Top_40_Features'
        score_pd.fillna(0, inplace=True)
        if not top_k: top_k = 30;
        
        for i in score_pd.columns:
            average_feature[i] = [score_pd[i].mean()]   
        average_feature = average_feature.transpose()
        average_feature.columns = ['score']
        average_feature.nlargest(n = top_k, columns='score').plot(title='Feature Importance',kind='barh', figsize = (20,10), ylabel='Features', xlabel='Features Importance', color = '#fe83cc') ## plot top 40 features
        plt.savefig(path+'/'+ name+'.png',bbox_inches = 'tight')


    def objective_old_original(self,trial):
        """Objective function of optuna will carry the finetuning trial out and contains the search space

        Parameters
        ----------
        trial : Optuna.Trial.trial
            trial

        Returns
        -------
        int
            error metric on which Optuna determines how successfull the trail was.
        """
        raise DeprecationWarning
        param = {
            "objective": 'reg:absoluteerror',
            "eval_metric": 'mae',
            "seed": 77,
            "verbosity": 0,
        "device": 'cuda'
        ,
            "n_estimators" : trial.suggest_int('n_estimators', 20, 1000),
            'min_child_weight':trial.suggest_float('min_child_weight', 0.01, 1.0, log=True),
           "booster": trial.suggest_categorical("booster", ["gbtree", "dart", 'gblinear']),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
           'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10,100)
        }
        
        if param["booster"] == "gbtree":
            param["eta"] = trial.suggest_float("eta", 1e-8, 0.01, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["max_depth"] = trial.suggest_int("max_depth", 1, 300)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            param['sampling_method'] = trial.suggest_categorical('sampling_method', ['uniform'])
            
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
            param["eta"] = trial.suggest_float("eta", 1e-8, 0.01, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["max_depth"] = trial.suggest_int("max_depth", 1, 300)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        
        if param["booster"] =="gblinear":
            param['updater'] = trial.suggest_categorical('updater', ['coord_descent' ])  
           
            param['feature_selector'] = trial.suggest_categorical('feature_selector',['shuffle', 'cyclic'])
           
            param['top_k'] = trial.suggest_int('top_k',0 ,100)
            param["eta"] = trial.suggest_float("eta", 1e-8, 0.01, log=True)
          
        early_stopping = {
            'rounds': trial.suggest_int('rounds', 10,100),
            'min_delta':trial.suggest_float('min_delta', 1e-5,2, log=True ),
            'save_best':False,
            'maximize':False,
            'data_name':'validation_0',
            'metric_name':'mae'
        }
        early_stopping = xgb.callback.EarlyStopping(**early_stopping)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-mae")
        model = xgb.XGBRegressor(**param,  callbacks = [pruning_callback, early_stopping])
        model.fit(self.train_set,self.label_train,
                eval_set=[(self.test_set, self.label_test)],
                    verbose=0)
        pred = model.predict(self.test_set)
        mae = mean_absolute_error(self.label_test, pred)
        r2 = r2_score(self.label_test, pred)
        mse =mean_squared_error(self.label_test, pred)
        rmse = math.sqrt(mse)
        mape =mean_absolute_percentage_error(self.label_test, pred)
        return mae


    def objective(self, trial) -> float:
        """Objective function of optuna will carry the finetuning trial out and contains the search space

        Parameters
        ----------
        trial : Optuna.Trial.trial
            trial

        Returns
        -------
        int
            error metric on which Optuna determines how successfull the trail was.
        """
        param = {
            "objective": "reg:squarederror",  # für MSE/RMSE
            "eval_metric": "rmse",
            "seed": 77,
            "verbosity": 0,
            "device": "cuda",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart", "gblinear"]),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        }

        booster = param["booster"]

        if booster in ["gbtree", "dart"]:
            param.update({
                "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                "gamma": trial.suggest_float("gamma", 0, 10.0),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            })

        if booster == "gbtree":
            param["sampling_method"] = trial.suggest_categorical("sampling_method", ["uniform", "gradient_based"])

        if booster == "dart":
            param.update({
                "sample_type": trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
                "normalize_type": trial.suggest_categorical("normalize_type", ["tree", "forest"]),
                "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
                "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
            })

        if booster == "gblinear":
            param.update({
                "updater": trial.suggest_categorical("updater", ["coord_descent"]),
                "feature_selector": trial.suggest_categorical("feature_selector", ["shuffle", "cyclic"]),
                "top_k": trial.suggest_int("top_k", 0, 100),
                "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
            })
        param['device'] = 'cuda'
        early_stopping = xgb.callback.EarlyStopping(
            rounds=trial.suggest_int("early_stopping_rounds", 10, 50),
            min_delta=trial.suggest_float("min_delta", 1e-5, 1e-1, log=True),
            save_best=False,
            maximize=False,
            data_name="validation_0",
            metric_name="rmse"
        )

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")

        model = xgb.XGBRegressor(**param, callbacks=[early_stopping, pruning_callback])

        model.fit(
            self.train_set,
            self.label_train,
            eval_set=[(self.test_set, self.label_test)],
            verbose=0
        )

        pred = model.predict(self.test_set)
        mse = mean_squared_error(self.label_test, pred)
        rmse = math.sqrt(mse)

        return rmse


    def run_rfe(self):
            """
            Runs an iterative version of sklearn's RFE. Here it iterates through all features and selects for each of the number the best features
            

            """
            X_train, X_test,label_train, label_test = utils.loso_finetuning_ml(self.data)
            X_train = utils.clean_up_features(X_train, extra_columns_to_drop=['id', 'task', 'cat_1', 'participant', 'PPG_fixed_peaks','PPG_Hr'])
            X_test = utils.clean_up_features(X_test, extra_columns_to_drop=['id', 'task', 'cat_1', 'participant', 'PPG_fixed_peaks','PPG_Hr'])
            params = copy.deepcopy(self.parameters)
            try: 
                # remove the early stopping parameter
                del params['early_stopping_rounds']
            except KeyError: pass      
            xg = xgb.XGBRegressor(**params)
            model = RFE(estimator=xg)
            model.set_output(transform="pandas")
            model.fit(X_train, label_train)
            X_train2 = model.transform(X_train)
            model.predict(X_test)
            selected_features = pd.DataFrame(columns=['Name_Features', 'Feature_in_Use', 'Rank'])
            for i in range(X_train.shape[1]):
                 print('Column: %s, Selected %s, Rank: %.3f' % (X_train.columns[i], model.support_[i], model.ranking_[i]))
                 temp = pd.DataFrame({'Name_features':X_train.columns[i], 'Feature_in_Use':model.support_[i], 'Rank':model.ranking_[i]}, index=[i])
                 selected_features = pd.concat([selected_features, temp], axis='index')
                 
                 
            selected_features.sort_values(by='Rank', inplace=True)
            self.dump_files([selected_features], path=path)
            columns = X_train.columns.tolist()
            columns_bool = model.support_
            from itertools import compress
            col_list = list(compress(columns, columns_bool))
            print(col_list)
            ranking_list = list(compress(columns, model.ranking_))
            print(ranking_list)
            return col_list, ranking_list


    def run_rfe_number(self):
            """
            Runs an iterative version of sklearn's RFE. Here it iterates through all features and selects for each of the number the best features

            """
            X_train, X_test,label_train, label_test = utils.loso_finetuning_ml(self.data)
            X_train = utils.clean_up_features(X_train, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant', 'PPG_fixed_peaks','PPG_Hr'])
            X_test = utils.clean_up_features(X_test, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant', 'PPG_fixed_peaks','PPG_Hr'])
            params = copy.deepcopy(self.parameters)
            try: 
               
                del params['early_stopping_rounds']
            except KeyError: pass      
            
            results, names, selected_features_names = [], [], []

            for i in range(1,len(X_train.columns)+1):
                 xg = xgb.XGBRegressor(**params)
                 model = RFE(estimator=xg, n_features_to_select=i)
                 model.set_output(transform="pandas")
                 model.fit(X_train,label_train)
                 selected_features_train = model.transform(X_train)
                 selected_features_test = model.transform(X_test)
                 xg.fit(selected_features_train,label_train)
                 preds = xg.predict(selected_features_test)
                 mae = mean_absolute_error(label_test, preds)
                 selection = np.array(X_train.columns)[model.support_]
                 results.append(mae)
                 selected_features_names.append(selection)
                 names.append(str(i))



            self.dump_files([results, selected_features_names, names], path=path)

            ranked_features = sorted(zip(names, results, selected_features_names), key=lambda x: x[1])
            print("Ranked features based on MAE:")
            for rank in ranked_features:print(f'Number of features: {rank[0]}, MAE: {rank[1]:.3f}, Selected Features: {rank[2]}')




    def run_rfe_number_kfold(self):
        """
        Bestimmt für jeden äußeren Fold:

        1. die optimale Anzahl an Features mittels RFECV
        2. die ausgewählten Feature-Namen
        3. den MAE auf dem äußeren Testdatensatz

        Die Auswahl der Feature-Anzahl erfolgt ausschließlich innerhalb
        der Trainingsdaten des jeweiligen äußeren Folds.

        Ergebnisse werden gespeichert in:

        self.rfe_fold_results
            DataFrame mit einer Zeile pro Fold.

        self.rfe_selected_features
            Dictionary mit den ausgewählten Features pro Fold.

        self.rfe_cv_results
            Dictionary mit den vollständigen inneren RFECV-Ergebnissen.

        self.rfe_models
            Dictionary mit dem finalen Modell pro Fold.

        Returns
        -------
        fold_results : pd.DataFrame
            Zusammenfassung aller äußeren Folds.

        selected_features_per_fold : dict
            Ausgewählte Feature-Namen pro Fold.

        rfecv_results_per_fold : dict
            Ergebnisse der inneren RFECV pro Fold.
        """



        if not hasattr(self, "data"):
            raise AttributeError(
                "self.data ist nicht vorhanden. "
                "Die Daten müssen vor dem Methodenaufruf gesetzt werden."
            )

        if not hasattr(self, "parameters"):
            raise AttributeError(
                "self.parameters ist nicht vorhanden."
            )

        if len(self.data) < 2:
            raise ValueError(
                "Für die äußere Cross-Validation werden mindestens "
                "zwei Datensätze beziehungsweise Teilnehmer benötigt."
            )

    
        outer_cv = KFold(
            n_splits=5,
            shuffle=False
        )

        columns_to_drop = [
            "id",
            "task",
            "cat_2",
            "participant",
            "PPG_fixed_peaks",
            "PPG_Hr"
        ]

    
        model_params = copy.deepcopy(self.parameters)
        model_params.pop("early_stopping_rounds", None)

      
        self.rfe_fold_results = None
        self.rfe_selected_features = {}
        self.rfe_cv_results = {}
        self.rfe_models = {}
        self.rfe_selectors = {}

        fold_results = []

  

        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(self.data)
        ):
            gc.collect()
            data_copy = copy.deepcopy(self.data)

           
            train_data = [
                data_copy[index]
                for index in train_idx
            ]

            test_data = [
                data_copy[index]
                for index in test_idx
            ]

            (
                train_set,
                test_set,
                label_train,
                label_test
            ) = utils.loso_datasets_ml(
                train_data=train_data,
                test_data=test_data
            )



            decoded_ids = utils.decode_string_columns(
                pd.DataFrame(
                    test_set["id"],
                    columns=["id"]
                ),
                decode_dict=self.decode_dict
            )

            participant = decoded_ids.iloc[0, 0]

            print("\n" + "=" * 70)
            print(
                f"Fold {fold} | Testset: Participant {participant}"
            )
            print("=" * 70)


            X_train = utils.clean_up_features(
                train_set,
                extra_columns_to_drop=columns_to_drop
            )

            X_test = utils.clean_up_features(
                test_set,
                extra_columns_to_drop=columns_to_drop
            )

            y_train = np.asarray(label_train).ravel()
            y_test = np.asarray(label_test).ravel()

            if X_train.empty:
                raise ValueError(
                    f"Fold {fold}: X_train enthält keine Features."
                )

          
            X_test = X_test.reindex(columns=X_train.columns)

           
            missing_test_columns = X_test.columns[
                X_test.isna().all()
            ].tolist()

            if missing_test_columns:
                raise ValueError(
                    f"Fold {fold}: Folgende Features fehlen vollständig "
                    f"im Testdatensatz: {missing_test_columns}"
                )

   
            n_inner_splits = min(5, len(X_train))

            if n_inner_splits < 2:
                raise ValueError(
                    f"Fold {fold}: Zu wenige Trainingsbeispiele für "
                    "die innere Cross-Validation."
                )

            inner_cv = KFold(
                n_splits=n_inner_splits,
                shuffle=True,
                random_state=42
            )


            base_estimator = xgb.XGBRegressor(
                **model_params
            )

            rfecv = RFECV(
                estimator=base_estimator,

                step=1,

                min_features_to_select=1,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            rfecv.fit(
                X_train,
                y_train
            )

            selected_features = X_train.columns[
                rfecv.support_
            ].tolist()

            optimal_n_features = int(
                rfecv.n_features_
            )


            X_train_selected = X_train.loc[
                :,
                selected_features
            ]

            X_test_selected = X_test.loc[
                :,
                selected_features
            ]

            final_model = xgb.XGBRegressor(
                **model_params
            )

            final_model.fit(
                X_train_selected,
                y_train
            )

            predictions = final_model.predict(
                X_test_selected
            )

            test_mae = mean_absolute_error(
                y_test,
                predictions
            )


            inner_cv_results = pd.DataFrame(
                rfecv.cv_results_
            )

            if "mean_test_score" in inner_cv_results.columns:
                inner_cv_results["mean_validation_mae"] = (
                    -inner_cv_results["mean_test_score"]
                )

            if "std_test_score" in inner_cv_results.columns:
                inner_cv_results["std_validation_mae"] = (
                    inner_cv_results["std_test_score"]
                )


            if "n_features" in inner_cv_results.columns:
                inner_cv_results = inner_cv_results.sort_values(
                    by="n_features"
                ).reset_index(drop=True)

 

            self.rfe_selected_features[fold] = selected_features
            self.rfe_cv_results[fold] = inner_cv_results
            self.rfe_models[fold] = final_model
            self.rfe_selectors[fold] = rfecv

            fold_results.append({
                "fold": fold,
                "participant": participant,
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "total_n_features": X_train.shape[1],
                "optimal_n_features": optimal_n_features,
                "selected_features": selected_features,
                "outer_test_mae": float(test_mae)
            })


            fold_output = {
                "fold": fold,
                "participant": participant,
                "optimal_n_features": optimal_n_features,
                "selected_features": selected_features,
                "outer_test_mae": float(test_mae),
                "predictions": predictions,
                "true_values": y_test
            }

           

            print(
                f"Optimale Feature-Anzahl: {optimal_n_features}"
            )
            print(
                f"Ausgewählte Features: {selected_features}"
            )
            print(
                f"MAE auf dem äußeren Test-Fold: {test_mae:.4f}"
            )

       

        self.rfe_fold_results = pd.DataFrame(
            fold_results
        )

        

        print("\n" + "=" * 70)
        print("Zusammenfassung aller Folds")
        print("=" * 70)

        print(
            self.rfe_fold_results[
                [
                    "fold",
                    "participant",
                    "total_n_features",
                    "optimal_n_features",
                    "outer_test_mae"
                ]
            ].to_string(index=False)
        )

        print(
            "\nDurchschnittlicher äußerer Test-MAE: "
            f"{self.rfe_fold_results['outer_test_mae'].mean():.4f}"
        )

        return (
            self.rfe_fold_results,
            self.rfe_selected_features,
            self.rfe_cv_results
        )


    def make_train_test_sets(self, data:dict, test_size:float=0.3, normalizing:bool=True, label:Optional[pd.DataFrame]=pd.DataFrame())-> tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]:
        """Make the train and test sets for XGB and Optuna.

        Parameters
        ----------
        data : dict
            Training data, 
        test_size : float, optional
            size of the test data, by default 0.3
        normalizing : bool, optional
            wether the data should be normalized or not, by default True
        label : Optional[pd.DataFrame], optional
            passng the label corresponding to the data, by default pd.DataFrame()

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]
            train data, train  label, test data, test label
        """
       
      
        if normalizing: 
            train_samples = []
            test_samples = []
            train2 = np.array([*data.values()]).tolist()
            for participant_data in data.values():
                for task_data in participant_data.values():
                    task_data = task_data.sample(frac=1, random_state=42)  # Shuffle the data but keep the index
                    split_index = int(len(task_data) * (1 - test_size))
                    train_samples.append(task_data.iloc[:split_index])
                    test_samples.append(task_data.iloc[split_index:])

            train_data = pd.concat(train_samples)
            test_data = pd.concat(test_samples)

            train_list = utils.generate_train_test_dict(train2=train2, x_split=train_data)
            test_list = utils.generate_train_test_dict(train2=train2, x_split=test_data)
            test_dicts_scaled, test_scaled, train_scaled, train_dicts_scaled = utils.normalize(train_list, test_list)

            

            test_set = pd.concat([pd.concat([*x.values()], axis='index') for x in test_dicts_scaled], axis='index').sample(frac=1, random_state=42)
            train_set = pd.concat([pd.concat([*x.values()], axis='index') for x in train_dicts_scaled], axis='index').sample(frac=1, random_state=42)  # added shuffling# added shuffling

            train_set.dropna(how='any', axis='index', inplace=True)
            test_set.dropna(how='any', axis='index', inplace=True)
            label_test = test_set['label']
            test_set = test_set.drop(['label'], axis='columns')
            label_train = train_set['label']
            train_set = train_set.drop(['label'], axis='columns')

            
        else: 
            all_data = pd.concat([x for xs in [[*x.values()] for x in list(data.values())] for x in xs])
            train_data, test_data = train_test_split(all_data, test_size=test_size)
            train_set.dropna(how='any', axis='index', inplace=True)
            test_set.dropna(how='any', axis='index', inplace=True)
            label_test = test_set['label']
            test_set = test_set.drop(['label'], axis='columns')
            label_train = train_set['label']
            train_set = train_set.drop(['label'], axis='columns')
            
        return train_set, label_train, test_set, label_test      


    def normalize(self,train_data:pd.DataFrame, test_data:pd.DataFrame, scale_method:Optional[str]='standard')-> tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]:
        """Calls scikit normalizer to scale the features

        Parameters
        ----------
        train_data : pd.DataFrame
            contains the training data
        test_data : pd.DataFrame
            contains the testing data
        scale_method : Optional[str], optional
            which Scikti scaler should be used: standard, minmax, by default 'standard'

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]
            scaled train data, train  label, scaled test data, test label
        """
        if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
        elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        
        label_train = train_data['label'].astype(int)
        label_test = test_data['label'].astype(int)
        train_data.drop('label', inplace=True, axis='columns')    
        test_data.drop('label', inplace=True, axis='columns')
        columns = train_data.columns    
        X_train = pd.DataFrame(scaler.fit_transform(train_data), columns=columns)
        X_test = pd.DataFrame(scaler.transform(test_data), columns=columns)
        X_train.interpolate('cubicspline', order=5, limit_direction='both', inplace = True)
        X_test.interpolate('cubicspline', order=5, limit_direction='both', inplace = True)
        return X_train, label_train, X_test, label_test







    def optimizing_with_optuna(self,n_iterations:int , sampler, 
                           warmup_steps:int, study_name:str='regression',
                           direction:str ='minimize', test_size:float=0.2, val_size: Optional[float]=None)-> dict: 
        """Optimizing with the optuuna framewoek. This function will call the objective function and give her the data.

        Parameters
        ----------
        n_iterations : int
            number of trials
        sampler : optuna.samplers
            sampling algorthim, whch samples the data from the search space
        warmup_steps : int
            Number of warmup steps only needed with specific Pruners
        study_name : str, optional
            name of the optuna's study, by default 'regression'
        direction : str, optional
            direction in which the error metric should go, by default 'minimize'
        test_size : float, optional
            size of the test data, by default 0.2
        val_size : Optional[float], optional
            size of validation data, by default None

        Returns
        -------
        dict
            found parameters and together with the best trial
        """
      
        X_train, X_test,label_train, label_test = utils.loso_finetuning_ml(self.data)
        X_train = utils.clean_up_features(X_train, extra_columns_to_drop=['id', 'task', 'cat_1', 'participant', 'PPG_fixed_peaks','PPG_Hr','EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Recovery', 'EDA_SCR_Onsets', 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'HRV_TINN', 'HRV_SDRMSSD'])
        X_test = utils.clean_up_features(X_test, extra_columns_to_drop=['id', 'task', 'cat_1', 'participant', 'PPG_fixed_peaks','PPG_Hr','EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Recovery', 'EDA_SCR_Onsets', 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'HRV_TINN', 'HRV_SDRMSSD'])
        
        self.train_set = X_train
        self.label_train = label_train
        self.test_set = X_test
        self.label_test = label_test
        results_tpe={}
        
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(
            min_resource=30, reduction_factor=6
            ),
            sampler=sampler,
            direction='minimize', study_name=study_name)
        study.optimize(self.objective, n_trials=n_iterations, catch=(NameError,AttributeError, ModuleNotFoundError, TypeError, ValueError), gc_after_trial=True)


        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))



        print('Best parameters', study.best_params)
        print('Best value', study.best_value)
        print('Best trial', study.best_trial)

        results_tpe['best_params'] = study.best_params
        results_tpe['best_value'] = study.best_value
        results_tpe['best_trial'] = study.best_trial
        results_tpe['study'] = study
        return results_tpe    
        
   
    def calculate_mean_sum_scores(self, scores : pd.DataFrame):
        """Calculates the mean scores for the error MAE

        Parameters
        ----------
        scores : pd.DataFrame
            Results of the error

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            average scores, summed up scores
        """
        average_score=pd.DataFrame(scores.mean(axis='rows'),  columns=['Mean_Scores']).sort_values(by=['Mean_Scores'], ascending=False)
        sum_score = pd.DataFrame(scores.sum( axis='rows' ), columns=['Scores_Sum']).sort_values(by=['Scores_Sum'], ascending=False)
        print('Average MAE across all folds {} '.format(average_score.loc['MAE']))
        return average_score, sum_score    


    def xgboost_trial(self):
        """Will execute the class and train, test and if specified fine-tune the XGBoost

        Returns
        -------
        tuple[dict, dict,pd.DataFrame]
           Errormetrics, predicitions, errors for the best model
        """
        
        sampler = TPESampler()
        warmup_steps = 7
        N_ITER = self.trails or 10000 
        test_size = 0.2
        name = 'regression'
        direction = 'maximize'
        device = 'cuda'

        if self.optuna: 
            optuna_results = self.optimizing_with_optuna(N_ITER,sampler, warmup_steps=warmup_steps,study_name=name, direction=direction)
            self.dump_files([optuna_results],self.path)
            optuna_results['best_params']["eval_metric"] = ['mape','rmse', 'mae']

            param = copy.deepcopy(optuna_results['best_params'])
            try:
                md = copy.deepcopy(param['min_delta'])
                r = copy.deepcopy(param['rounds'])
                del param['min_delta'], param['rounds']
                self.early_stopping = {
                            'rounds': r,
                            'min_delta':md,
                            'save_best':False,
                            'maximize':False,
                            'data_name':'validation_1',
                            'metric_name':'mae'
                        }
            except KeyError: pass 
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data, self.parameters, self.path, top_k=30)
            
        
            mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)
        else: 
           
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data,self.parameters, self.path, top_k=30)
            
        
            mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)    
        return scores_table,preds,score_pd
    



import argparse
import datetime
import torch
parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')   
parser.add_argument('--final_selected_features', default=False, type=bool, help='Run the final models')   
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--annova', default=False, type=bool)
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r'path/to/data' # 



# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print(data)
print('Decode Dict: \n',decode_dict)
trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
if args.percent < 100:
    trial_name = trial_name + '_percent_'+str(args.percent)+'_'
    data = utils.reduce_dataframes_size(data, percentage=args.percent)
if args.finetune:
    n_iterations = args.trials or 1000    
    trial_name= trial_name+'finetuning' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, trials= n_iterations, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()

elif args.rfe:
   
    trial_name= trial_name+'_rfe' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, decode_dict=decode_dict)
    features, features3 = xgboost_model.run_rfe_number_kfold()
elif args.initial:
    trial_name= trial_name+'_init' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=True)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
elif args.final:
    print('running final version of the model:')
    trial_name= trial_name+'_final' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=False)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
elif args.final_selected_features:
    trial_name= trial_name+'_selected_features' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
elif args.annova:
   
    trial_name= trial_name+'_with_annova' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, decode_dict=decode_dict)
    features = xgboost_model.run_anova_feature_selection(
        
    )         
                  
         
else: 
    trial_name= trial_name+'_normal' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
    
