# rocket and its variations
# Imports
import copy
import datetime
import gc
import inspect
import math
import multiprocessing
import os
import pickle
import platform
import statistics
from statistics import mean
from typing import Any
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import multirocket
import multiRocketHydra
import numpy as np
import optuna
import pandas as pd
import sklearn
import utils
from optuna.samplers import TPESampler
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import argparse
import datetime
import torch


class rocket_trial:
   
    def __init__(
        self,
        data: dict,
        path: str = os.path.abspath(os.getcwd()),
        name: str = "",
        name_label: str = "label",
        num_folds: int = 5,
        trial_name: str = "regression_cognitve_load",
        optuna: bool = True,
        decode_dict:dict=None,
        trials: int = 1000,
        init: bool = False,
        percentage:int = 100

    ):
        self.parameters = {
            "rocket_version": "multirocket",
            "n_kernels": 10000,
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": 32,
            "n_features_per_kernel": 4,
            "estimator": None,
            "random_state": 77,
            "n_jobs": 1,
            "xgb_params": {
                "objective": "reg:squarederror",
                "eval_metric": ["mape", "rmse", "mae"],
                "device": "cuda",
            },
            "callback_early_stopping": {
                "rounds": 30,
                "min_delta": 7e-4,
                "save_best": False,
                "maximize": False,
                "data_name": "validation_1",
                "metric_name": "mae",
            },
            "use_optuna": False,
            "optuna_callback": None,
        }

        if init:
            self.parameters = [
                {
                    "objective": "reg:squarederror",
                    "eval_metric": ["mape", "rmse", "mae"],
                    "device": "cuda",
                }
            ]
           
        self.optuna = optuna
        self.num_folds = num_folds
        self.name = name
        self.name_label = name_label
        self.percentage = percentage
        self.decode_dict = decode_dict     
        self.trials = trials or 1000
        self.data = data
        path = path + "/rocket_boost_" + trial_name + "personalized"
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.parameters_list = [

            {
                "rocket_version": "multirocket",
                "n_kernels": 10000,
                "rocket_transform": "rocket",
                "max_dilations_per_kernel": 32,
                "n_features_per_kernel": 4,
                "estimator": None,
                "random_state": 77,
                "n_jobs": 1,
                "xgb_params": {
                    "objective": "reg:squarederror",
                    "eval_metric": ["mape", "rmse", "mae"],
                    "device": "cuda",
                },
                "callback_early_stopping": {
                    "rounds": 30,
                    "min_delta": 7e-4,
                    "save_best": False,
                    "maximize": False,
                    "data_name": "validation_1",
                    "metric_name": "mae",
                },
                "use_optuna": False,
                "optuna_callback": None,
            }
        ]

        self.path_save_preds =r'/path/to/folder'


    def get_params(self, params: dict) -> dict:
        rocket_version = params.get("rocket_version")

        rocket_keys = {
            "booster",
            "n_estimators",
            "min_child_weight",
            "lambda",
            "alpha",
            "eta",
            "gamma",
            "max_depth",
            "grow_policy",
            "sampling_method",
            "sample_type",
            "normalize_type",
            "rate_drop",
            "skip_drop",
            "updater",
            "feature_selector",
            "top_k",
        }

        rocket_params_xgb = {k: v for k, v in params.items() if k in rocket_keys}
        rocket_params_xgb.update(
            {
                "objective": "reg:squarederror",
                "eval_metric": ["mape", "rmse", "mae"],
                "seed": 77,
                "verbosity": 0,
                "device": "cuda",
            }
        )

        # Early-Stopping-Parameter
        early_stopping = {
            "rounds": params.get("early_stopping_rounds"),
            "min_delta": params.get("min_delta"),
            "save_best": False,
            "maximize": False,
            "data_name": "validation_1",
            "metric_name": "mae",
        }

        # Rocket-Parameter
        
        if rocket_version == "hydra":
            rocket_params = {
                "n_kernels": params.get("h_n_kernels"),
                "n_groups": params.get("h_n_group"),
                "callback_early_stopping": early_stopping,
            }
        elif rocket_version == "multirockethydra":
            rocket_params = {
                "n_kernels": params.get("mh_n_kernels"),
                "n_groups": params.get("mh_n_group"),
                "callback_early_stopping": early_stopping,
            }
        elif rocket_version == "multirocket":
            rocket_params = {
                "n_kernels": params.get("mr_n_kernels"),
                "rocket_transform": "rocket",
                "max_dilations_per_kernel": params.get("mr_max_dilations_per_kernel"),
                "n_features_per_kernel": params.get("mr_n_features_per_kernel"),
                "callback_early_stopping": early_stopping,
            }

        return rocket_version, rocket_params, rocket_params_xgb, early_stopping

    def train(self, num_folds: int, parameters: list[dict] | dict):
        """Trains the rocketoost model, it will split the data into test and trainings data according the number of folds.
        Will call also plotting functions to viszualize the the predictions the ground truth.

        Parameters
        ----------
        data : dict
            contains the training data
        num_folds : int
            mumber of folds for the K-Fold
        parameters : dict
            specified parameters of the rocketoost, are defined in the Intialization of this class or through Optuna
        path : str
            where to save the images and data to
        name_label : str
            name of the label column
        top_k : int, optional
            number of the most important features, by default 30

        Returns
        -------

        tuple[list, dict, rocket.rocketRegressor, pd.DataFrame, pd.DataFrame]
            list of scores, predictions, (feature importance), best performing model, scores of the best model, scores
        """

        predictions = {}
        scores_participant= {}
        score_pd_part = {}
        best_models ={}
        

        
        c = 0
        for participant in self.data:
           
            splitter = KFold(n_splits=num_folds)
            best_model = {}
            pred = {}
            id = utils.decode_string_columns(pd.DataFrame(participant[0]['participant'], columns=['participant']), decode_dict=decode_dict)
            name = id.iloc[0].item()
            participant2 = copy.deepcopy(participant)
            print('Creating model for participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(name))
            for fold,(train_idx, test_idx) in enumerate(splitter.split(participant2)):
                participant1 =copy.deepcopy(participant2)
                train_set = np.array(participant1, dtype=object)[train_idx]
                test_set = np.array(participant1, dtype=object)[test_idx]   
                test_set, train_set, label_test, label_train, auto_test, auto_train = utils.personal_datasets_dl(train_data=train_set, test_data=test_set) # TODO
                print('------------------------------- ✧ ✶𐙚⋆°｡⋆♡ Fold No.: {} ♡⋆｡°⋆𐙚✶ ✧ -------------------------------'.format(str(fold)))
                
                
                test_set = self.dataframe_to_3d_numpy(test_set)
                train_set = self.dataframe_to_3d_numpy(train_set)

                if len(parameters) > 1 and isinstance(parameters, dict):
                    param = copy.deepcopy(parameters[name])
                    rocketversion, rocket_params, xgb_params, earlystoppingparams = (
                        self.get_params(params=param)
                    )
                    
                   
                    rocket_params['xgb_params'] = xgb_params
                    rocket_params["eval_set"] = [
                            (train_set, label_train),
                            (test_set, label_test),
                        ]
                    rocket_params[ "callback_early_stopping"] = earlystoppingparams
                   
                    if rocketversion == "hydra":
                        model = hydra.HydraRegressor(**rocket_params)
                    elif rocketversion == "multirocket":
                        model = multirocket.MultiRocketRegressor(**rocket_params)
                    else:
                        model = multiRocketHydra.MultiRocketHydraRegressor(
                            **rocket_params
                        )
                else:
                    try:
                        rocketversion = parameters["rocket_version"]
                        rocket_params = copy.deepcopy(parameters)
                      
                        rocket_params["eval_set"] = [
                            (train_set, label_train),
                            (test_set, label_test),
                        ]
                        
                        rocket_params[xgb_params]=xgb_params
                        rocket_params['early_stopping_dict'] = earlystoppingparams
                        del rocket_params["rocket_version"]
                        if rocketversion == "hydra":
                            model = hydra.HydraRegressor(**rocket_params)
                        elif rocketversion == "multirocket":
                            model = multirocket.MultiRocketRegressor(**rocket_params)
                        else:
                            model = multiRocketHydra.MultiRocketHydraRegressor(
                                **rocket_params
                            )
                    except KeyError:
                        print(
                            "Incorrect parameter configuration, missing rocket version"
                        )
                        model = multirocket.MultiRocketRegressor()
                


                model.fit(
                    train_set,
                    label_train,
                )
                compare = pd.DataFrame()
                pre = model.predict(test_set)

                mse = mean_squared_error(label_test, pre)
                mae = mean_absolute_error(label_test, pre)
                mape = mean_absolute_percentage_error(label_test, pre)
                r = r2_score(label_test, pre)
                print("MSE: ", mse)
                print("RMSE: ", math.sqrt(mse))
                print("MAE: ", mae)

                compare["rmse"] = math.sqrt(mse)
                compare["mape"] = mape
                compare["mae"] = mae
                compare["r2"] = r
                comp = {}

                comp["rmse"] = math.sqrt(mean_squared_error(label_test, pre))
                comp["mse"] = mean_squared_error(label_test, pre)
                comp["mape"] = mean_absolute_percentage_error(label_test, pre)
                comp["mae"] = mean_absolute_percentage_error(label_test, pre)
                comp["r2"] = r2_score(label_test, pre)
                comp = pd.DataFrame(comp, index=["Fold " + str(fold)])
                name_fold = name + "_Fold_" + str(fold)
                pred[name_fold] = compare
                pred[name_fold + "_pred"] = pre
                pred[name_fold + "_truth"] = label_test
                # Speichern des Models, nutzlos ich weiß aber nun gut
                if fold == 0:
                    best_model["rmse"] = math.sqrt(mse)
                    best_model["mape"] = mape
                    best_model["mae"] = mae

                    best_model["model"] = model

                elif mae < best_model["mae"]:
                    best_model["rmse"] = math.sqrt(mse)
                    best_model["mape"] = mape
                    best_model["mae"] = mae
                    best_model["model"] = model

                self.visualize_results(
                    ground_truth=label_test,
                    pred=pre,
                    nombre="_svm",
                    compare=comp,
                    fold=fold,
                    participant=name,
                )
                self.visualize_results_color(
                    ground_truth=label_test,
                    pred=pre,
                    nombre="_svm",
                    compare=comp,
                    fold=fold,
                    participant=name,
                )
            c = c + 1
            predictions[str(name)] = pred
            best_models[name] = best_model
        self.dump_files_for_shap(
            [predictions],
            path=self.path_save_preds,
            suffix=self.trial_name + "rocket_personalized",
        )
        return predictions, best_model

    def dataframe_to_3d_numpy(self, df):
        """
        Konvertiert einen pandas DataFrame, bei dem jede Zelle eine Zeitreihe enthält
        (z. B. Liste oder numpy Array), in ein 3D numpy Array mit der Form
        (n_instances, n_features, n_timepoints), geeignet für AEON-Modelle.

        Parameter:
        df (pd.DataFrame): Eingabe-DataFrame mit Shape (n_instances, n_features)

        Rückgabe:
        np.ndarray: 3D numpy Array mit Shape (n_instances, n_features, n_timepoints)
        """
        n_instances = df.shape[0]
        n_features = df.shape[1]
        n_timepoints = len(
            df.iloc[0, 0]
        )  

        X = np.empty((n_instances, n_features, n_timepoints), dtype=np.float32)

        for i in range(n_instances):
            for j in range(n_features):
                X[i, j, :] = np.array(df.iloc[i, j], dtype=np.float32)

        return X

    def visualize_results(
        self,
        ground_truth: np.array,
        pred: np.array,
        nombre: str,
        compare: pd.DataFrame,
        fold: int,
        participant: str,
    ) -> None:
        """Viszualizing the results. This will plot a point plot of the predicitions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.

        Parameters
        ----------
        ground_truth : np.array
            true values of the cl
        pred : np.array
            predicited values of the cl
        nombre : str
            name of the plot, e.g., name of the model
        compare : pd.DataFrame
            dataframe of the error metriken only on row
        fold : int
            The current fold number
        participant : str
            Name of the participant
        """
        fig12 = plt.figure(figsize=(40, 7))
        colors = [
            "#E6BFCC",
            "#E98DAD",
            "#C74777",
            "#7A042A",
            "#4F021B",
            "#663845",
            "#8B4E5E",
        ]
        compare.plot(kind="bar", color=colors, rot=30)
        plt.xlabel("Error Metrics")
        plt.ylabel("Error")
        plt.title(
            "Error for Fold " + str(fold) + " of Participant " + str(participant),
            wrap=True,
        )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        name = (
            "Error_metrics_fold_"
            + nombre
            + str(fold)
            + "_participant_"
            + str(participant)
            + ".png"
        )
        plt.savefig(self.path + "/" + name, bbox_inches="tight")
        plt.close(fig12)

        temp = pd.DataFrame({"True label": ground_truth, "Predicted label": pred})
        fig22 = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        import seaborn as sns

        sns.regplot(
            data=temp,
            x="True label",
            y="Predicted label",
            marker="o",
            scatter_kws={"color": "#E98DAD"},
            line_kws={"color": "#c74777"},
        )
        plt.xlabel("True Cognitive Load")
        plt.ylabel("Predicted Cognitive Load")
        plt.title(
            "Cognitive load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )
        name_scatter = (
            "cl_true_pred_participant_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        sns.scatterplot(
            data=temp,
            x=temp.index,
            y="True label",
            color="#e84a7f",
            label="True Cognitive Load",
            marker="*",
        )
        sns.scatterplot(
            data=temp,
            x=temp.index,
            y="Predicted label",
            color="#E98DAD",
            label="Predicted Cognitive Load",
            marker="o",
        )
        plt.xlabel("Time windows")
        plt.ylabel("Cognitive Load")
        plt.title(
            "Cognitive load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )
        name_scatter = (
            "cl_true_pred_participant_line_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig323c)
        fig323cc = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        sns.lineplot(
            data=temp["True label"],
            color="#E84a7f",
            label="True True Cognitive Load",
            marker="o",
            linestyle="dotted",
        )
        sns.lineplot(
            data=temp["Predicted label"],
            color="#E98DAD",
            label="Predicted Cognitive load",
            marker="o",
            linestyle="",
        )

        plt.xlabel("Time windows")
        plt.ylabel("True Cognitive Load")
        plt.title(
            "Cognitve load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )

        name_scatter = (
            "cl_true_pred_participant_with_line_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig323cc)

    def visualize_results_color(
        self,
        ground_truth: np.array,
        pred: np.array,
        nombre: str,
        compare: pd.DataFrame,
        fold: int,
        participant: str,
    ) -> None:
        """Viszualizing the results. This will plot a point plot of the predicitions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.

        Parameters
        ----------
        ground_truth : np.array
            true values of the cl
        pred : np.array
            predicited values of the cl
        nombre : str
            name of the plot, e.g., name of the model
        compare : pd.DataFrame
            dataframe of the error metriken only on row
        fold : int
            The current fold number
        participant : str
            Name of the participant
        """
        fig12 = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        colors = [
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
        ]
        compare.plot(kind="bar", color=colors, rot=30)
        plt.xlabel("Error Metrics")
        plt.ylabel("Error")
        plt.title(
            "Error for Fold " + str(fold) + " of Participant " + str(participant),
            wrap=True,
        )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        name = (
            "Error_metrics_foldf_"
            + nombre
            + str(fold)
            + "_participant_"
            + str(participant)
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name, bbox_inches="tight")
        plt.close(fig12)

        temp = pd.DataFrame({"True label": ground_truth, "Predicted label": pred})
        fig22 = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        import seaborn as sns

        sns.regplot(
            data=temp,
            x="True label",
            y="Predicted label",
            marker="o",
            scatter_kws={"color": "#377eb8"},
            line_kws={"color": "#4daf4a"},
        )
        plt.xlabel("True Cognitive Load")
        plt.ylabel("Predicted Cognitive Load")
        plt.title(
            "Cognitive load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )
        name_scatter = (
            "cl_true_pred_participantf_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        sns.scatterplot(
            data=temp,
            x=temp.index,
            y="True label",
            color="#377eb8",
            label="True Cognitive Load",
            marker="o",
        )
        sns.scatterplot(
            data=temp,
            x=temp.index,
            y="Predicted label",
            color="#4daf4a",
            label="Predicted Cognitive Load",
            marker="o",
        )
        plt.xlabel("Time windows")
        plt.ylabel("Cognitive Load")
        plt.title(
            "Cognitive load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )
        name_scatter = (
            "cl_true_pred_participant_linef_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig323c)

        fig323cc = plt.figure(figsize=(50,10))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 25})
        sns.lineplot(
            data=temp["True label"],
            color="#377eb8",
            label="True Cognitive Load",
            marker="o",
            linestyle="dotted",
        )
        sns.lineplot(
            data=temp["Predicted label"],
            color="#E98DAD",
            label="Predicted Cognitive load",
            marker="o",
            linestyle="",
        )

        plt.xlabel("Time windows")
        plt.ylabel("True Cognitive Load")
        plt.title(
            "Cognitive load estimated from psycho-physiological data vs. subjective perception of participant "
            + str(participant)
            + " from fold "
            + str(fold),
            wrap=True,
        )
        name_scatter = (
            "cl_true_pred_participant_with_linef_"
            + str(participant)
            + "_Fold_"
            + str(fold)
            + nombre
            + ".pdf"
        )
        plt.savefig(self.path + "/" + name_scatter)
        plt.close(fig323cc)

        plt.close("all")

    

    def dump_files_for_shap(
        self, files_list: list, path: str, suffix: str = ""
    ) -> None:
        """
        This function will take the variable name and use it as a file name to dump it in pickle files.

        :param files_list: list of variable names which should be dumped, just copy and paste the names
        :param path: path which you want to dump the files
        :param suffix: optional string to be added to the file name
        :return: None
        """
        for file in files_list:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            name = [
                var_name for var_name, var_val in callers_local_vars if var_val is file
            ]
            if name[0].startswith("_") and name[0].endswith("_") and len(name) > 1:
                name[0] = name[1]
            with open(f"{path}/{name[0]}{suffix}.pickle", "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save all for later
    def dump_files(self, files_list, path):
        """
        This function will take the variable name and use it as a file name to dump it in pickle files.

        :param files_list: list of variable names which should be dumped, just copy and paste the names
        :param path: path which you want to dump the files
        :return: None
        """

        for file in files_list:
            callers_local_vars = inspect.currentframe().f_back.f_locals.items()
            name = [
                var_name for var_name, var_val in callers_local_vars if var_val is file
            ]
            with open(path + "/" + name[0] + ".pickle", "wb") as handle:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    def split_participant_data(
        self, data_dict: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits each DataFrame of each participant in a nested dictionary into 70% training and 30% testing sets,
        and returns concatenated DataFrames for training and testing.

        Parameters
        ----------
        data_dict : dict
            A nested dictionary where the first-level keys are participant identifiers and the second-level values
            are dictionaries mapping task names to pandas DataFrames. Each DataFrame represents time-series data
            for a specific task of a participant.

        Returns
        -------
        train_df : pandas.DataFrame
            A concatenated DataFrame containing the 70% training splits from all participants and tasks.

        test_df : pandas.DataFrame
            A concatenated DataFrame containing the 30% testing splits from all participants and tasks.

        Notes
        -----
        - Each DataFrame is cleaned by removing duplicated columns, dropping columns with excessive NaNs,
        and interpolating missing values using PCHIP interpolation.
        - The split is performed in temporal order (i.e., first 70% rows for training, remaining 30% for testing).
        - This function assumes that each task DataFrame is already preprocessed to be suitable for interpolation.
        """
        train_data = []
        test_data = []

        for participant in data_dict:
            task_dfs = data_dict[participant].values()

            for df in task_dfs:
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.dropna(thresh=60, axis="columns")
                df = df.interpolate("pchip", limit_direction="both")

                split_index = int(len(df) * 0.7)
                train_split = df.iloc[:split_index].copy()
                test_split = df.iloc[split_index:].copy()

                train_data.append(train_split)
                test_data.append(test_split)

        train_df = pd.concat(train_data, axis="index", ignore_index=True)
        test_df = pd.concat(test_data, axis="index", ignore_index=True)

        return train_df, test_df

    def cleanup_callback(self, study, trial):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def trial(self):
        """Will execute the class and train, test and if specified fine-tune the rocket_model

        Returns
        -------
        tuple[dict, dict,pd.DataFrame]
           Errormetrics, predicitions, errors for the best model
        """
     

        if self.optuna:

            DeprecationWarning

        else:

            best_model, predictions =  (
                self.train(self.num_folds, self.parameters_list)
            )
            







parser = argparse.ArgumentParser(description='rocket Training and finetuning')
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')  
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'/path/to/data' # 
saving_path = r'/path/to/folder'
# 
data, decode_dict = utils.preprare_data(path, 'dl_features')
# TODO change it for the rocket versions.


n_iterations = args.trials or 800    

print('Decode Dict: \n',decode_dict)
trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
if args.percent < 100:
    trial_name = trial_name + '_percent_'+str(args.percent)+'_'
    data = utils.reduce_dataframes_size(data, percentage=args.percent, path=saving_path, decode_dict=decode_dict)



if args.initial:
    trial_name= trial_name+'_init_perso_rocket' # for the name
    rocket = rocket_trial(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=True, percentage= args.percent)
    scores_table,preds,score_pd = rocket.trial()
elif args.final:
    trial_name= trial_name+'_final_endgültig_run_rocket_personalized_allerletzerrun' # for the name
    rocket = rocket_trial(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=False, percentage= args.percent)
    scores_table,preds,score_pd = rocket.trial()         
else: 
    trial_name= trial_name+'_normal_perso_rocket' # for the name
    rocket = rocket_trial(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, percentage= args.percent)
    scores_table,preds,score_pd = rocket.trial()
    