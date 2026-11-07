import torch
import platform
import pandas as pd 
import os
import matplotlib.pyplot as plt
from typing import Optional
import inspect
import pickle
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR
import xgboost
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import math
from sklearn.feature_selection import f_classif
import inspect
import pickle
from sklearn.model_selection import KFold
from sklearn.base import clone
import utils
import copy
import gc
from SVCRFE import CustomSVC
from NuSVCRFE import CustomNuSVC
class svm:
    def __init__(self, data: dict, 
                 path:str=os.path.abspath(os.getcwd()),
                 optuna:bool=False, 
                 name:str='',
                 trial_name: str='classification_cognitve_load', 
                 decode_dict:dict = None, 
                 trials:int =1000, 
                 tsne:bool=False, 
                 init=False,
                 kfold:bool=False,
                 number_fold:int=5,
                 select_features:bool=False,
                 features:list=[]) -> None:
        """Initialization of the svm class. This class contains the training, and testing of/with SVM. The model will be validated with the LOOCV (currently a 4-fold CV). \
            The class also contains multiple plot functions to visualize the predictions.


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
        """
        # finetuned parameters 
       
        self.selected_features = select_features
        self.features = features or []
        self.kfold=kfold
        self.number_folds = number_fold if not np.nan or 0 else 5
        #finetuned parameters endgültig
        self.parameters = {'svm': 'SVC'}
        self.name = trial_name
        if 'percent' in trial_name and '50' in trial_name:
            self.parameters={'svm': 'NuSVC'}
        elif  'percent' in trial_name and '75' in trial_name:
             self.parameters={'svm': 'NuSVC'}
        if init: 
            # empty dict for no parameters -> intitial run with default params
            self.parameters = {'max_iter':1000}
        self.decode_dict = decode_dict
        path = path +'/svm_'+trial_name+'_LOOCV'
        self.data = data
        self.tsne = tsne or False
        self.N_ITER = trials or 1000
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.optuna = optuna


        self.path_save_preds =r''

    def run_anova_feature_selection(
        self,
        n_splits: int = 5
    ) -> pd.DataFrame:
        """
        Determine the optimal feature subset using univariate F-tests.

        The method automatically determines whether the configured estimator
        solves a classification or regression problem. The estimator type is
        read from `self.parameters`.

        Supported classification estimators are SVC, NuSVC, LinearSVC, and
        XGBClassifier. Classification features are ranked using the ANOVA
        F-value returned by `f_classif`, and feature subsets are evaluated
        using the F1 score.

        Supported regression estimators are SVR, NuSVR, LinearSVR, and
        XGBRegressor. Regression features are ranked using the univariate
        regression F-value returned by `f_regression`, and feature subsets
        are evaluated using the mean absolute error.

        If `self.parameters` contains the key `"svm"`, its value determines
        the estimator type. If the key is absent, the method uses the value
        of `"model"` when available. Otherwise, XGBRegressor is used as the
        default estimator.

        Within each cross-validation fold, the F-scores are calculated using
        only the training data. Features are ranked from the highest to the
        lowest F-score. Separate models are subsequently fitted using the
        first 1, 2, 3, up to all ranked features.

        For classification, the subset with the highest F1 score is selected.
        For regression, the subset with the lowest mean absolute error is
        selected. If multiple subsets have the same score, the subset with
        fewer features is preferred.

        Results are stored in the following instance attributes:

        `self.anova_fold_results`
            DataFrame containing the best result for every fold.

        `self.anova_best_features`
            Dictionary mapping every fold to its best feature subset.

        `self.anova_feature_rankings`
            Dictionary mapping every fold to its complete feature ranking.

        `self.anova_score_tables`
            Dictionary containing F-scores and p-values for every fold.

        `self.anova_all_results`
            Dictionary containing the evaluation results for all tested
            feature counts.

        Parameters
        ----------
        n_splits : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        pd.DataFrame
            Fold-level results containing the detected estimator type,
            problem type, optimal number of features, selected features,
            and best evaluation score.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If the estimator type, cross-validation configuration, target
            values, or feature matrix is invalid.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        The feature-ranking methods used here are univariate. Each feature is
        evaluated independently, and interactions between features are not
        considered.

        The optimal feature count is selected using the test data of the
        respective fold. Therefore, the best fold score is not an independent
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
            raise ValueError("n_splits must be at least 2.")

        if n_splits > len(self.data):
            raise ValueError(
                "n_splits cannot exceed the number of elements "
                "in self.data."
            )


        model_parameters = copy.deepcopy(self.parameters)

        if "svm" in model_parameters:
            model_type = model_parameters.pop("svm")
        elif "model" in model_parameters:
            model_type = model_parameters.pop("model")
        else:
            model_type = "XGBRegressor"

        classification_models = {
            "SVC",
            "NuSVC",
            "LinearSVC",
            "XGBClassifier"
        }

        regression_models = {
            "SVR",
            "NuSVR",
            "LinearSVR",
            "XGBRegressor"
        }

        supported_models = (
            classification_models
            | regression_models
        )

        if model_type not in supported_models:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported model types are: "
                f"{sorted(supported_models)}."
            )

        if model_type in classification_models:
            problem_type = "classification"
            selection_method = "f_classif"
            optimization_metric = "f1_score"
        else:
            problem_type = "regression"
            selection_method = "f_regression"
            optimization_metric = "mean_absolute_error"

        model_parameters.pop("early_stopping_rounds", None)

        print(f"Automatically detected model: {model_type}")
        print(f"Detected problem type: {problem_type}")
        print(f"Feature-selection method: {selection_method}")
        print(f"Optimization metric: {optimization_metric}")

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

        self.anova_fold_results = None
        self.anova_best_features = {}
        self.anova_feature_rankings = {}
        self.anova_score_tables = {}
        self.anova_all_results = {}

        all_fold_results = []


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

            decoded_ids = utils.decode_string_columns(
                pd.DataFrame(
                    test_set["id"],
                    columns=["id"]
                ),
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
            print(f"Model: {model_type}")
            print(f"Problem type: {problem_type}")
            print("=" * 70)

            X_train = utils.clean_up_features(
                train_set,
                extra_columns_to_drop=columns_to_drop
            )

            X_test = utils.clean_up_features(
                test_set,
                extra_columns_to_drop=columns_to_drop
            )

            X_test = X_test.reindex(
                columns=X_train.columns
            )

        
            if problem_type == "classification":
                y_train = np.asarray(
                    utils.map_labels(labels=label_train)
                ).ravel()

                y_test = np.asarray(
                    utils.map_labels(labels=label_test)
                ).ravel()

            else:
                y_train = np.asarray(
                    label_train,
                    dtype=float
                ).ravel()

                y_test = np.asarray(
                    label_test,
                    dtype=float
                ).ravel()


            if X_train.empty:
                raise ValueError(
                    f"Fold {fold}: The training set contains no features."
                )

            if X_train.isna().any().any():
                missing_columns = X_train.columns[
                    X_train.isna().any()
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: The training features contain "
                    f"missing values in columns: {missing_columns}."
                )

            if X_test.isna().any().any():
                missing_columns = X_test.columns[
                    X_test.isna().any()
                ].tolist()

                raise ValueError(
                    f"Fold {fold}: The test features contain missing "
                    f"values or unavailable columns: {missing_columns}."
                )

            if not np.isfinite(y_train).all():
                raise ValueError(
                    f"Fold {fold}: The training targets contain "
                    "NaN or infinite values."
                )

            if not np.isfinite(y_test).all():
                raise ValueError(
                    f"Fold {fold}: The test targets contain "
                    "NaN or infinite values."
                )

            if problem_type == "classification":
                unique_train_labels = np.unique(y_train)
                unique_test_labels = np.unique(y_test)

                if not set(unique_train_labels).issubset({0, 1}):
                    raise ValueError(
                        f"Fold {fold}: Training labels must be encoded "
                        f"as zero and one. Received: "
                        f"{unique_train_labels.tolist()}."
                    )

                if not set(unique_test_labels).issubset({0, 1}):
                    raise ValueError(
                        f"Fold {fold}: Test labels must be encoded "
                        f"as zero and one. Received: "
                        f"{unique_test_labels.tolist()}."
                    )

                if len(unique_train_labels) < 2:
                    raise ValueError(
                        f"Fold {fold}: The training data contain only "
                        "one class."
                    )

            else:
                if np.unique(y_train).size < 2:
                    raise ValueError(
                        f"Fold {fold}: The regression target is constant "
                        "in the training data."
                    )

            total_number_of_features = X_train.shape[1]

            print(
                f"Number of available features: "
                f"{total_number_of_features}"
            )


            print(
                f"Calculating {selection_method} scores..."
            )

            with np.errstate(
                divide="ignore",
                invalid="ignore"
            ):
                if problem_type == "classification":
                    f_scores, p_values = f_classif(
                        X_train,
                        y_train
                    )
                else:
                    f_scores, p_values = f_regression(
                        X_train,
                        y_train,
                        center=True,
                        force_finite=True
                    )

            score_table = pd.DataFrame({
                "feature": X_train.columns,
                "f_score": f_scores,
                "p_value": p_values,
                "original_position": np.arange(
                    total_number_of_features
                )
            })

            score_table["f_score"] = (
                score_table["f_score"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            score_table["p_value"] = (
                score_table["p_value"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
            )

            score_table = (
                score_table
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

            feature_ranking = score_table[
                "feature"
            ].tolist()

            self.anova_feature_rankings[fold] = (
                feature_ranking.copy()
            )

            self.anova_score_tables[fold] = (
                score_table.copy()
            )

            print("\nFeature ranking:")

            for rank, score_row in score_table.iterrows():
                print(
                    f"{rank + 1:3d}. "
                    f"{score_row['feature']} | "
                    f"F-score: {score_row['f_score']:.4f} | "
                    f"p-value: {score_row['p_value']:.6f}"
                )


            feature_counts = []
            metric_values = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for current_number_of_features in range(
                1,
                total_number_of_features + 1
            ):
                selected_features = feature_ranking[
                    :current_number_of_features
                ]

                if model_type == "SVC":
                    estimator = SVC(
                        **model_parameters
                    )

                elif model_type == "NuSVC":
                    estimator = NuSVC(
                        **model_parameters
                    )

                elif model_type == "LinearSVC":
                    estimator = LinearSVC(
                        **model_parameters
                    )

                elif model_type == "SVR":
                    estimator = SVR(
                        **model_parameters
                    )

                elif model_type == "NuSVR":
                    estimator = NuSVR(
                        **model_parameters
                    )

                elif model_type == "LinearSVR":
                    estimator = LinearSVR(
                        **model_parameters
                    )

                elif model_type == "XGBClassifier":
                    estimator = xgb.XGBClassifier(
                        **model_parameters
                    )

                else:
                    estimator = xgb.XGBRegressor(
                        **model_parameters
                    )

                try:
                    estimator.fit(
                        X_train.loc[:, selected_features],
                        y_train
                    )

                    predictions = estimator.predict(
                        X_test.loc[:, selected_features]
                    )

                    if problem_type == "classification":
                        metric_value = f1_score(
                            y_test,
                            predictions,
                            zero_division=0
                        )
                    else:
                        metric_value = mean_absolute_error(
                            y_test,
                            predictions
                        )

                    feature_counts.append(
                        current_number_of_features
                    )

                    metric_values.append(
                        float(metric_value)
                    )

                    selected_feature_sets.append(
                        selected_features.copy()
                    )

                    if problem_type == "classification":
                        print(
                            f"Number of features: "
                            f"{current_number_of_features:3d} | "
                            f"F1: {metric_value:.4f}"
                        )
                    else:
                        print(
                            f"Number of features: "
                            f"{current_number_of_features:3d} | "
                            f"MAE: {metric_value:.4f}"
                        )

                except Exception as error:
                    print(
                        f"Evaluation failed for "
                        f"{current_number_of_features} features."
                    )
                    print(f"Error: {error}")

            if not metric_values:
                raise RuntimeError(
                    f"Fold {fold}: No feature subset could be evaluated."
                )


            if problem_type == "classification":
                
                ordered_result_indices = sorted(
                    range(len(metric_values)),
                    key=lambda index: (
                        -metric_values[index],
                        feature_counts[index]
                    )
                )
            else:
                # Lowest MAE wins. Fewer features are preferred when
                # multiple subsets have the same error.
                ordered_result_indices = sorted(
                    range(len(metric_values)),
                    key=lambda index: (
                        metric_values[index],
                        feature_counts[index]
                    )
                )

            best_result_index = ordered_result_indices[0]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_metric_value = metric_values[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ]

            self.anova_best_features[fold] = (
                best_features.copy()
            )

            metric_column = (
                "f1_score"
                if problem_type == "classification"
                else "mean_absolute_error"
            )

            fold_evaluation_results = pd.DataFrame({
                "number_of_features": feature_counts,
                metric_column: metric_values,
                "selected_features": selected_feature_sets
            })

            fold_evaluation_results = (
                fold_evaluation_results
                .sort_values(
                    by=[
                        metric_column,
                        "number_of_features"
                    ],
                    ascending=[
                        problem_type == "regression",
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            self.anova_all_results[fold] = (
                fold_evaluation_results
            )

            fold_result = {
                "fold": fold,
                "participants": participants,
                "model_type": model_type,
                "problem_type": problem_type,
                "selection_method": selection_method,
                "optimization_metric": optimization_metric,
                "total_number_of_features": (
                    total_number_of_features
                ),
                "best_number_of_features": (
                    best_number_of_features
                ),
                "best_features": best_features,
                "best_score": best_metric_value,
                "feature_ranking": feature_ranking
            }

            if problem_type == "classification":
                fold_result["best_f1_score"] = best_metric_value
                fold_result["best_mae"] = np.nan
            else:
                fold_result["best_f1_score"] = np.nan
                fold_result["best_mae"] = best_metric_value

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

            if problem_type == "classification":
                print(
                    f"Best F1 score: "
                    f"{best_metric_value:.4f}"
                )
            else:
                print(
                    f"Best MAE: "
                    f"{best_metric_value:.4f}"
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

        print(
            self.anova_fold_results[
                [
                    "fold",
                    "participants",
                    "model_type",
                    "problem_type",
                    "best_number_of_features",
                    "best_f1_score",
                    "best_mae"
                ]
            ].to_string(index=False)
        )

        return self.anova_fold_results

    
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





  
    def train(self,data:dict,parameters:dict)->tuple[dict, dict]:
        """Trains the SVM model, it will split the data into test and trainings data according the number of folds. 
        Will call also plotting functions to viszualize the the predictions the ground truth.

        Parameters
        ----------
        data : dict
            contains the training data
        num_folds : int
            mumber of folds for the K-Fold
        parameters : dict
            specified parameters of the XGBoost, are defined in the Intialization of this class or through Optuna
        path : str
            where to save the images and data to
        name_label : str
            name of the label column

        Returns
        -------
        
        tuple[dict,dict]
            predictions, scores of the best model, scores
        """

        kfold = KFold(n_splits=len(data))
        if self.kfold: kfold =KFold(n_splits=self.number_folds)
        predictions = {}
        best_model = {}
        trainings_data = []
        testing_data = []
        modeling = []
        for fold,  (train_idx, test_idx) in enumerate(kfold.split(data)):
            gc.collect()
           
            
            
            data2 = copy.deepcopy(data)
            train_data = np.array(data2, dtype=object)[train_idx].tolist()
            test_data = np.array(data2, dtype=object)[test_idx].tolist()
            train_set, test_set,label_train,label_test = utils.loso_datasets_ml(train_data=train_data, test_data=test_data)
            train_set_ids = utils.decode_string_columns(pd.DataFrame(train_set['id'], columns=['id']), decode_dict=self.decode_dict)
            
            id = utils.decode_string_columns(pd.DataFrame(test_set['id'], columns=['id']), decode_dict=self.decode_dict)
            participant = id.iloc[0].item() #sometimes shit cannot be calculated sadly :(
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡'.format(fold,participant))
            train_set = utils.clean_up_features(train_set, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])
            test_set = utils.clean_up_features(test_set, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])
            label_train_cl = utils.map_labels(labels=label_train)
            label_test_cl = utils.map_labels(label_test)
            label_compl = pd.concat([pd.DataFrame(label_test_cl), pd.DataFrame(label_train_cl)], axis='index')
            
            # adapt features here.´
            self.features = svc_rfe = [
            
            ]
            if isinstance(self.features, list) and all(isinstance(x, list) for x in self.features): # for each of the folds get a differnt feature set.
                 train_set = train_set[self.features[fold]]
                 test_set = test_set[self.features[fold]]
            elif self.selected_features:
                 train_set = train_set[self.features]
                 test_set = test_set[self.features]
            if self.tsne:
                utils.plot_tsne(X_train=train_set, y_train=pd.DataFrame({'labels':label_train_cl}), person_labels_train=train_set_ids['id'],X_test=test_set, y_test=pd.DataFrame({'labels':label_test_cl}), person_labels_test=id['id'], name='Set of '+str(fold),path= self.path)
            try:
                if parameters['svm'] == 'SVC':
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVC(**param)
                elif parameters['svm'] == 'NuSVC': 
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVC(**param)
                else:
                    param = copy.deepcopy(parameters)
                    del param['svm']
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = LinearSVC(**param)  
            except KeyError:     
                model = SVC()    
            model.fit(train_set,label_train_cl)
            compare = pd.DataFrame()
            y_proba=model.predict(test_set)
            modeling.append(model)


        
            
            
            comp = {}
            mae_expected = mean_absolute_error(label_test_cl, y_proba)
            pre = y_proba
            comp['acc'] = accuracy_score(label_test_cl, pre)
            comp['f1'] = f1_score(label_test_cl, pre)
            comp['mae'] = mean_absolute_error(label_test_cl, pre)
            comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
            acc = accuracy_score(label_test_cl, pre)
            print('ACC: ', accuracy_score(label_test_cl, pre))
            f1s = f1_score(label_test_cl, pre)
            print('F1S: ', f1s)
            mae = mean_absolute_error(label_test_cl, pre)
            print('MAE: ', mae)
            print('MAE Exp.: ', mae_expected)

            compare['mae'] = mae
            compare['f1s'] = f1s
            compare['acc'] = acc            
            name = 'Fold_'+str(fold)
            predictions[name]=compare
            predictions[name+'pred_org'] = pre
            predictions[name+'_truth'] = label_test_cl
            predictions[name + "_truth_org"] = label_test
            # Speichern des Models, nutzlos ich weiß aber nun gut
            if fold ==0:
                
                best_model['mae'] = mae
                best_model['f1s'] =f1s
                best_model['acc']=acc
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = model

            elif f1s > best_model['f1s']:
                best_model['mae'] = mae
                best_model['f1s'] =f1s
                best_model['acc']=acc
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = model
            self.dump_files([testing_data, trainings_data, modeling], self.path)    
            self.visualize_results(ground_truth=label_test, pred=pre, nombre='_svc', compare=comp, fold=fold, participant=participant)
            self.visualize_results_frederic(ground_truth=label_test, pred=pre, nombre='_svc', compare=comp, fold=fold, participant=participant)
        self.dump_files_for_shap([predictions], self.path_save_preds, suffix=self.trial_name+'svm_loocv')    
        return predictions, best_model


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
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#e84a7f', label='True Cognitive Load', marker='o')
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
        # close any open figures
        plt.close('all')



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

        fig323cc = plt.figure(figsize=(40,7))
        sns.lineplot(data=temp['True label'], color='#377eb8', label='True Cognitive Load', marker='o',linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#4daf4a', label='Predicted Cognitive load',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_with_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323cc)
        # close any open figures
        plt.close('all')

    
    def dump_files(self,files_list, path) -> None:
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

            
    

   
    
    def svm_trial(self)->dict:
        """Will execute the class and train, test the SVM.

        Returns
        -------
        dict
           predicitions
        """
        device = 'cuda'
        
        preds_svc, bestmodel =  self.train(self.data, self.parameters)
        
        self.dump_files([preds_svc], self.path)

       
        
        return preds_svc
    

    def objective(self, trial):
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
                
        svm = trial.suggest_categorical('svm', ['LinearSVC', 'SVC', 'NuSVC'])
        model = None

        if svm == 'LinearSVC':
           
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),  # 1e2 == 100
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
                # penalty bleibt auf 'l2' (Default), da Kombis mit 'l1' schnell invalid werden
                'dual': trial.suggest_categorical('dual', [True, False]),  # Achtung: bei loss='hinge' dual=True nötig
                'fit_intercept': True,
                'intercept_scaling': trial.suggest_float('intercept_scaling', 0.5, 2.0),
                'class_weight': 'balanced',
                'max_iter': trial.suggest_int('max_iter', 1000, 200000),
            }
            
            if params['loss'] == 'hinge':
                params['dual'] = True
            model = LinearSVC(**params)

        elif svm == 'NuSVC':
           
            params = {
                'nu': trial.suggest_float('nu', 1e-3, 0.9, log=True),  # (0,1], praktikabel: 0.01–0.5
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12),  # relevant nur bei 'poly', sonst ignoriert
                'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                'class_weight': 'balanced',
                'probability': trial.suggest_categorical('probability', [True, False]),
                'decision_function_shape': trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr']),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 200000),
                'random_state': 77,  
            }
            model = NuSVC(**params)

        else:  # SVC
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12),  # nur für 'poly' relevant
                'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                'class_weight': 'balanced',
                'probability': trial.suggest_categorical('probability', [True, False]),
                'decision_function_shape': trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr']),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 200000),
                'random_state': 77,
            }
            model = SVC(**params)


        label_train_cl = utils.map_labels(labels=self.label_train)
        label_test_cl = utils.map_labels(self.label_test)
        model.fit(self.X_train,label_train_cl)
        pred = model.predict(self.X_test)
        

        mae = mean_absolute_error(label_test_cl, pred)
        f1 =f1_score(label_test_cl, pred)
        return f1



    def optimizing_with_optuna(self, n_iterations:int , sampler, 
                            warmup_steps:int, study_name:str='SVM regression',
                            direction:str='maximize', test_size:float=0.2, val_size: Optional[float]=None)-> dict: 
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
            name of the optuna's study, by default 'SVM regression'
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
        results_tpe={}
        # Create the study
        study = optuna.create_study(
            sampler=sampler,
            direction=direction, study_name=study_name)
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
        # best Model
        
        results_tpe['best_params'] = study.best_params
        results_tpe['best_value'] = study.best_value
        results_tpe['best_trial'] = study.best_trial
        results_tpe['study'] = study
        return results_tpe
     
    def rfe_svm_non_linear(
              self,
              svm:SVC|SVR,
              Q:int=20,
              data:pd.DataFrame=None, 
              test:pd.Series=None,
              k:int=6
    )->list:
         svm.fit(data, test)
         p = len(data.columns)
         samples = len(data.index)
         min_extreme_index = round(samples*0.02)
         max_extreme_index = round(samples*0.98)
         columns = data.columns.values.tolist()
         data_new_pd = data.loc[min_extreme_index:max_extreme_index]
         quantiles = round(np.linspace(0.02,0.98, num=Q)*samples)
         quantiles_pd = data_new_pd.loc[quantiles]
         mad_values = {}
         for i in quantiles_pd.columns:
              masked_pd = copy.deepcopy(quantiles_pd)
              masked_pd = 0
              masked_pd[i] = quantiles_pd[i]
              l = svm._decision_function(masked_pd)
              MAD = np.median(np.abs(l-quantiles_pd[i].median()))*1.4826
              mad_values[i] = MAD
         mad_pd = pd.DataFrame(mad_values, index=[0])
         while len(mad_pd.columns)>k:
              mad_pd.drop(mad_pd.min(axis='columns').name, inplace=True)
              
         return mad_pd.columns



    def quantile_grid(
        self,
        col_values: np.ndarray,
        q_min: float,
        q_max: float,
        Q: int
    ) -> np.ndarray:
        """
        Calculate feature values on an evenly spaced quantile grid.

        The method creates `Q` evenly spaced quantile positions between
        `q_min` and `q_max` and calculates the corresponding values for
        the supplied feature column.

        Parameters
        ----------
        col_values : np.ndarray
            One-dimensional array containing the values of a feature.
        q_min : float
            Lower quantile boundary between 0 and 1.
        q_max : float
            Upper quantile boundary between 0 and 1.
        Q : int
            Number of quantile positions.

        Returns
        -------
        np.ndarray
            Feature values corresponding to the specified quantile positions.

        Raises
        ------
        ValueError
            If the quantile boundaries or the number of quantiles are invalid.
        """
        if not 0 <= q_min < q_max <= 1:
            raise ValueError(
                "q_min and q_max must satisfy "
                "0 <= q_min < q_max <= 1."
            )

        if Q < 2:
            raise ValueError("Q must be at least 2.")

        quantiles = np.linspace(q_min, q_max, Q)

        return np.quantile(
            np.asarray(col_values, dtype=float),
            quantiles
        )
    

    def rfe_svm_non_linear_pseudosamples_by_name(
        self,
        svm,
        X: pd.DataFrame,
        y,
        K: int,
        Q: int = 21,
        q_min: float = 0.02,
        q_max: float = 0.98,
        baseline_value: float = 0.0
    ) -> tuple:
        """
        Rank features using pseudo-sample RFE for nonlinear SVM models.

        The method implements a pseudo-sample-based recursive feature
        elimination procedure for nonlinear support vector machines.

        During each iteration, the SVM is fitted using the currently
        available features. For every remaining feature, `Q` pseudo-samples
        are created. The feature under consideration is varied across its
        quantile values, while every other feature is fixed at
        `baseline_value`.

        Feature relevance is quantified as the scaled median absolute
        deviation of the resulting decision-function values. The feature
        with the smallest relevance score is removed. The procedure is
        repeated until `K` features remain.

        Parameters
        ----------
        svm : SVC or NuSVC
            Nonlinear support vector classifier used for feature ranking.
        X : pd.DataFrame
            Feature matrix with samples in rows and feature names in columns.
            The features should be standardized when `baseline_value=0.0`.
        y : pd.Series or np.ndarray
            Binary target labels.
        K : int
            Number of features to retain.
        Q : int, default=21
            Number of pseudo-samples generated for each feature.
        q_min : float, default=0.02
            Lower quantile boundary used to exclude extreme values.
        q_max : float, default=0.98
            Upper quantile boundary used to exclude extreme values.
        baseline_value : float, default=0.0
            Value assigned to features that are not varied in a pseudo-sample.
            For standardized features, zero usually represents the mean.

        Returns
        -------
        selected_features : list of str
            Names of the `K` retained features.
        feature_ranking : list of str
            Complete feature ranking from most important to least important.
        elimination_history : list of dict
            Information about the removed feature and MAD scores during each
            elimination iteration.

        Raises
        ------
        TypeError
            If `X` is not a DataFrame or `svm` is not an SVC or NuSVC model.
        ValueError
            If `K`, `Q`, or the input feature matrix is invalid.

        Notes
        -----
        The pseudo-sample construction and the MAD-based feature elimination
        are kept intact. The complete ranking is derived from the elimination
        order, allowing all feature counts to be evaluated without rerunning
        the entire elimination procedure.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(svm, (SVC, NuSVC)):
            raise TypeError(
                "svm must be an instance of SVC or NuSVC."
            )

        if X.empty:
            raise ValueError("X must contain at least one feature.")

        if K < 1 or K > X.shape[1]:
            raise ValueError(
                "K must be between 1 and the number of features."
            )

        if Q < 2:
            raise ValueError("Q must be at least 2.")

        y = np.asarray(y).ravel()

        remaining_features = X.columns.tolist()
        removed_features = []
        elimination_history = []

        while len(remaining_features) > K:
            current_model = clone(svm)

            X_subset = X.loc[:, remaining_features]

            current_model.fit(
                X_subset,
                y
            )

            number_of_features = len(remaining_features)
            mad_scores = np.zeros(number_of_features, dtype=float)
            current_feature_scores = {}

            for feature_position, feature_name in enumerate(
                remaining_features
            ):
                quantile_values = self.quantile_grid(
                    col_values=X[feature_name].to_numpy(),
                    q_min=q_min,
                    q_max=q_max,
                    Q=Q
                )

                pseudo_samples = np.full(
                    shape=(Q, number_of_features),
                    fill_value=baseline_value,
                    dtype=float
                )

                pseudo_samples[
                    :,
                    feature_position
                ] = quantile_values

                decision_values = current_model.decision_function(
                    pseudo_samples
                )

                decision_values = np.asarray(
                    decision_values
                ).ravel()

                median_decision_value = np.median(
                    decision_values
                )

                mad_score = 1.4826 * np.median(
                    np.abs(
                        decision_values - median_decision_value
                    )
                )

                mad_scores[feature_position] = mad_score
                current_feature_scores[feature_name] = float(
                    mad_score
                )

            removal_position = int(
                np.argmin(mad_scores)
            )

            removed_feature = remaining_features[
                removal_position
            ]

            elimination_history.append({
                "number_of_features_before_removal": (
                    number_of_features
                ),
                "removed_feature": removed_feature,
                "removed_feature_score": current_feature_scores[
                    removed_feature
                ],
                "feature_scores": current_feature_scores.copy()
            })

            remaining_features.pop(
                removal_position
            )

            removed_features.append(
                removed_feature
            )

        selected_features = remaining_features.copy()

        feature_ranking = (
            remaining_features
            + removed_features[::-1]
        )

        return (
            selected_features,
            feature_ranking,
            elimination_history
        )


    def run_rfe(self):
        """
        Determine the optimal SVM feature subset for each cross-validation fold.

        The method performs cross-validation over the elements in `self.data`.
        For nonlinear SVC and NuSVC models, the paper-based pseudo-sample RFE
        method is executed once per fold to obtain a complete feature ranking.
        The first 1, 2, 3, up to all ranked features are then evaluated using
        separate classification models.

        For LinearSVC, standard recursive feature elimination is executed once
        per fold to obtain the complete feature ranking.

        The feature subset with the highest F1 score is selected as the best
        subset for each fold. If multiple subsets have the same F1 score, the
        subset with fewer features is preferred.

        Results are stored in the following instance attributes:

        `self.svm_rfe_fold_results`
            DataFrame containing the best result for each fold.

        `self.svm_rfe_best_features`
            Dictionary mapping each fold to its best feature subset.

        `self.svm_rfe_feature_rankings`
            Dictionary mapping each fold to its complete feature ranking.

        `self.svm_rfe_all_results`
            Dictionary containing the F1 score and selected features for every
            tested feature count in each fold.

        `self.svm_rfe_elimination_history`
            Dictionary containing the pseudo-sample elimination history for
            each nonlinear SVM fold.

        Returns
        -------
        pd.DataFrame
            Fold-level results containing the best number of features, best
            feature names, and best F1 score.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If the model type, labels, feature matrix, or cross-validation
            configuration is invalid.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        The pseudo-sample construction and MAD-based feature elimination are
        preserved for SVC and NuSVC. Efficiency is improved by performing
        feature elimination once per fold and reusing the resulting ranking
        for every feature count.

        The best feature count is selected using the test set of the respective
        fold. Consequently, the reported maximum F1 score should not be
        interpreted as an independent estimate of generalization performance.
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

        if len(self.data) < 2:
            raise ValueError(
                "At least two data elements are required for "
                "cross-validation."
            )

        cross_validator = KFold(
            n_splits=5,
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

        svm_type = model_parameters.pop(
            "svm",
            "SVC"
        )

       
        if svm_type == "NuSVC":
            model_parameters.pop(
                "random_state",
                None
            )
            del model_parameters['random_state']

        self.svm_rfe_fold_results = None
        self.svm_rfe_best_features = {}
        self.svm_rfe_feature_rankings = {}
        self.svm_rfe_all_results = {}
        self.svm_rfe_elimination_history = {}

        all_fold_results = []

        for fold, (train_indices, test_indices) in enumerate(
            cross_validator.split(self.data)
        ):
            data_copy = copy.deepcopy(self.data)
            gc.collect()
            
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

            decoded_ids = utils.decode_string_columns(
                pd.DataFrame(
                    test_set["id"],
                    columns=["id"]
                ),
                decode_dict=self.decode_dict
            )

            participant = decoded_ids.iloc[0].item()

            print("\n" + "=" * 70)
            print(
                f"Fold {fold} | Test participant: {participant}"
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

            
            X_test = X_test.reindex(
                columns=X_train.columns
            )

            y_train = np.asarray(
                utils.map_labels(labels=label_train)
            ).ravel()

            y_test = np.asarray(
                utils.map_labels(labels=label_test)
            ).ravel()

            if X_train.empty:
                raise ValueError(
                    f"Fold {fold}: The training set contains no features."
                )

            if X_train.isna().any().any():
                raise ValueError(
                    f"Fold {fold}: The training features contain "
                    "missing values."
                )

            if X_test.isna().any().any():
                raise ValueError(
                    f"Fold {fold}: The test features contain missing "
                    "values or do not match the training features."
                )

            if not set(np.unique(y_train)).issubset({0, 1}):
                raise ValueError(
                    f"Fold {fold}: Training labels must be encoded as "
                    "zero and one."
                )

            if not set(np.unique(y_test)).issubset({0, 1}):
                raise ValueError(
                    f"Fold {fold}: Test labels must be encoded as "
                    "zero and one."
                )

            number_of_available_features = X_train.shape[1]

            print(f"SVM type: {svm_type}")
            print(
                f"Number of available features: "
                f"{number_of_available_features}"
            )



            if svm_type == "SVC":
                ranking_model = SVC(
                    **model_parameters
                )

                print(
                    "Calculating the pseudo-sample feature ranking..."
                )

                (
                    _,
                    feature_ranking,
                    elimination_history
                ) = self.rfe_svm_non_linear_pseudosamples_by_name(
                    svm=ranking_model,
                    X=X_train,
                    y=y_train,
                    K=1,
                    Q=21,
                    q_min=0.02,
                    q_max=0.98,
                    baseline_value=0.0
                )

            elif svm_type == "NuSVC":
                ranking_model = NuSVC(
                    **model_parameters
                )

                print(
                    "Calculating the pseudo-sample feature ranking..."
                )

                (
                    _,
                    feature_ranking,
                    elimination_history
                ) = self.rfe_svm_non_linear_pseudosamples_by_name(
                    svm=ranking_model,
                    X=X_train,
                    y=y_train,
                    K=1,
                    Q=21,
                    q_min=0.02,
                    q_max=0.98,
                    baseline_value=0.0
                )

            elif svm_type == "LinearSVC":
                ranking_model = LinearSVC(
                    **model_parameters
                )

                print(
                    "Calculating the standard RFE feature ranking..."
                )

                ranking_selector = RFE(
                    estimator=ranking_model,
                    n_features_to_select=1,
                    step=1
                )

                ranking_selector.fit(
                    X_train,
                    y_train
                )

                ranking_table = pd.DataFrame({
                    "feature": X_train.columns,
                    "rfe_rank": ranking_selector.ranking_,
                    "original_position": np.arange(
                        number_of_available_features
                    )
                })

                ranking_table = ranking_table.sort_values(
                    by=[
                        "rfe_rank",
                        "original_position"
                    ],
                    ascending=[
                        True,
                        True
                    ]
                ).reset_index(drop=True)

                feature_ranking = ranking_table[
                    "feature"
                ].tolist()

                elimination_history = []

            else:
                raise ValueError(
                    "Supported SVM types are 'SVC', 'NuSVC', and "
                    f"'LinearSVC'. Received: '{svm_type}'."
                )

            self.svm_rfe_feature_rankings[fold] = (
                feature_ranking.copy()
            )

            self.svm_rfe_elimination_history[fold] = (
                elimination_history
            )

            print("\nFeature ranking:")

            for rank, feature_name in enumerate(
                feature_ranking,
                start=1
            ):
                print(f"{rank:3d}. {feature_name}")

           
            feature_counts = []
            f1_values = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for number_of_features in range(
                1,
                number_of_available_features + 1
            ):
                selected_features = feature_ranking[
                    :number_of_features
                ]

                if svm_type == "SVC":
                    classifier = SVC(
                        **model_parameters
                    )

                elif svm_type == "NuSVC":
                    classifier = NuSVC(
                        **model_parameters
                    )

                else:
                    classifier = LinearSVC(
                        **model_parameters
                    )

                try:
                    classifier.fit(
                        X_train.loc[:, selected_features],
                        y_train
                    )

                    predictions = classifier.predict(
                        X_test.loc[:, selected_features]
                    )

                    f1_value = f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )

                    feature_counts.append(
                        number_of_features
                    )

                    f1_values.append(
                        float(f1_value)
                    )

                    selected_feature_sets.append(
                        selected_features.copy()
                    )

                    print(
                        f"Number of features: "
                        f"{number_of_features:3d} | "
                        f"F1: {f1_value:.4f}"
                    )

                except Exception as error:
                    print(
                        f"Evaluation failed for {number_of_features} "
                        "features."
                    )
                    print(f"Error: {error}")

            if not f1_values:
                raise RuntimeError(
                    f"Fold {fold}: No feature subset could be evaluated."
                )

           
            result_order = sorted(
                range(len(f1_values)),
                key=lambda result_index: (
                    -f1_values[result_index],
                    feature_counts[result_index]
                )
            )

            best_result_index = result_order[0]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_f1_score = f1_values[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ]

            self.svm_rfe_best_features[fold] = (
                best_features.copy()
            )

            fold_evaluation_results = pd.DataFrame({
                "number_of_features": feature_counts,
                "f1_score": f1_values,
                "selected_features": selected_feature_sets
            })

            fold_evaluation_results = (
                fold_evaluation_results
                .sort_values(
                    by=[
                        "f1_score",
                        "number_of_features"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            self.svm_rfe_all_results[fold] = (
                fold_evaluation_results
            )

            fold_result = {
                "fold": fold,
                "participant": participant,
                "svm_type": svm_type,
                "total_number_of_features": (
                    number_of_available_features
                ),
                "best_number_of_features": (
                    best_number_of_features
                ),
                "best_features": best_features,
                "best_f1_score": best_f1_score,
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
                f"Best F1 score: {best_f1_score:.4f}"
            )

            print("Best features:")

            for feature_name in best_features:
                print(f"  - {feature_name}")

            print(
                f"\nResults ranked by F1 score for fold {fold}:"
            )

            for _, result_row in fold_evaluation_results.iterrows():
                print(
                    f"Number of features: "
                    f"{result_row['number_of_features']}, "
                    f"F1: {result_row['f1_score']:.4f}"
                )

                print(
                    f"Features: {result_row['selected_features']}"
                )

            self.dump_files_for_shap(
                [
                    fold_result,
                    fold_evaluation_results,
                    feature_ranking,
                    elimination_history
                ],
                path=self.path,
                suffix=f"Fold_{fold}_{svm_type}_RFE"
            )

        
        self.svm_rfe_fold_results = pd.DataFrame(
            all_fold_results
        )

        self.dump_files_for_shap(
            [
                self.svm_rfe_fold_results,
                self.svm_rfe_best_features,
                self.svm_rfe_feature_rankings,
                self.svm_rfe_all_results,
                self.svm_rfe_elimination_history
            ],
            path=self.path,
            suffix=f"all_folds_{svm_type}_selected_features"
        )

        print("\n" + "=" * 70)
        print("Best feature count and feature subset per fold")
        print("=" * 70)

        print(
            self.svm_rfe_fold_results[
                [
                    "fold",
                    "participant",
                    "svm_type",
                    "total_number_of_features",
                    "best_number_of_features",
                    "best_f1_score"
                ]
            ].to_string(index=False)
        )

        return self.svm_rfe_fold_results


    def finetune (self):
        """Finetunes the SVC using optuna.

        Returns
        -------
        dict
            Results of the hyper-parameter-finetuing with optuna, contains best parameter, best run and so on
        """
        sampler = TPESampler()
        warmup_steps = 7
        self.name= self.name+'_optuna'
        self.direction = 'maximize'
        X_train, X_test,self.label_train, self.label_test = utils.loso_finetuning_ml(self.data)
        self.X_train = utils.clean_up_features(feature_data=X_train, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant', 'PPG_fixed_peaks','PPG_Hr','EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Recovery', 'EDA_SCR_Onsets', 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'HRV_TINN', 'HRV_SDRMSSD'])
        self.X_test = utils.clean_up_features(feature_data=X_test, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant', 'PPG_fixed_peaks','PPG_Hr','EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Recovery', 'EDA_SCR_Onsets', 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'HRV_TINN', 'HRV_SDRMSSD'])
        optuna_results = self.optimizing_with_optuna(self.N_ITER,sampler, warmup_steps=warmup_steps,study_name=self.name, direction=self.direction)
        self.dump_files([optuna_results],self.path)

        return optuna_results



    
import argparse
import datetime

parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')    
parser.add_argument('--final_selected_features', default=False, type=bool, help='Run the final models')   
parser.add_argument('--kfold', default=False, type=bool, help='Runs the model with kfold using by default 5 folds unless specified with --folds')
parser.add_argument('--folds', default=5, type=int, help='Number of folds which will be used for kfold.')
parser.add_argument('--tsne', default=False, type=bool)
parser.add_argument('--annova', default=False, type=bool)

args = parser.parse_args()
folds = args.folds or 5
features = [] # copy the features here if there are any.



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = r'' # 

# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print(data)
print('Decode Dict: \n',decode_dict)
trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
if args.kfold: trial_name=trial_name+'_kfold'

if args.percent < 100:
    trial_name = trial_name + '_percent_'+str(args.percent)+'_'
    data = utils.reduce_dataframes_size(data, percentage=args.percent)
if args.finetune:

    n_iterations = args.trials or 1000
    trial_name= trial_name+'during_tuning' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=True, trials= n_iterations, decode_dict=decode_dict)
    optuna_results= svm_model.finetune()

elif (args.tsne and args.initial):
    trial_name= trial_name+'_init_with_tsne' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict,init=True, tsne=True)
    preds= svm_model.svm_trial()
elif args.initial:
    trial_name= trial_name+'_init' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=False, init=True)
    preds= svm_model.svm_trial()
elif args.tsne:
    trial_name= trial_name+'_with_tsne' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    preds= svm_model.svm_trial()
elif args.rfe:
    trial_name= trial_name+'_with_rfe' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    col_list, col = svm_model.run_rfe()
elif args.final and not args.final_selected_features:
    trial_name= trial_name+'_endgultig_final_run_svc_loocv_with_tsne_allerletzterrun' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True,kfold=args.kfold, number_fold=args.folds)
    preds= svm_model.svm_trial()  
elif args.final_selected_features:
    trial_name= trial_name+'_final_run_svc_loocv_with_tsne_and_rfe_features_allerletzterrun' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True, kfold=args.kfold, number_fold=args.folds, select_features=True, features=features)
    preds= svm_model.svm_trial()     
elif args.annova:
    trial_name= trial_name+'_with_annove' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    col_list, col = svm_model.run_anova_feature_selection()     
else: 
    trial_name= trial_name+'_normal' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=False)
    preds= svm_model.svm_trial()
        
