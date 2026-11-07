"""

Script for finetuning the rocket variants in combination with xgboost. To make sure the vram/ram is clear, every trial
from optuna is created in a new thread.



"""
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


path_save_preds = r"/path/to/folder"


def get_params(params: dict) -> dict:
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

    rocket_params = {k: v for k, v in params.items() if k in rocket_keys}
    rocket_params.update(
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
    rocket_params = {}
    if rocket_version == "hydra":
        rocket_params = {
            "n_kernels": params.get("h_n_kernels"),
            "n_groups": params.get("h_n_group"),
            "rocket_params": rocket_params,
            "callback_early_stopping": early_stopping,
        }
    elif rocket_version == "multirockethydra":
        rocket_params = {
            "n_kernels": params.get("mh_n_kernels"),
            "n_groups": params.get("mh_n_group"),
            "rocket_params": rocket_params,
            "callback_early_stopping": early_stopping,
        }
    elif rocket_version == "multirocket":
        rocket_params = {
            "n_kernels": params.get("mr_n_kernels"),
            "rocket_transform": "rocket",
            "max_dilations_per_kernel": params.get("mr_max_dilations_per_kernel"),
            "n_features_per_kernel": params.get("mr_n_features_per_kernel"),
            "rocket_params": rocket_params,
            "callback_early_stopping": early_stopping,
        }

    return rocket_version, rocket_params, rocket_params, early_stopping


def dataframe_to_3d_numpy(df):
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
    )  # Annahme: alle Zellen haben gleich viele Zeitpunkte

    X = np.empty((n_instances, n_features, n_timepoints), dtype=np.float32)

    for i in range(n_instances):
        for j in range(n_features):
            X[i, j, :] = np.array(df.iloc[i, j], dtype=np.float32)

    return X


def run_trial(
    trial_number, study_name, storage_url, path,sampler, direction
    
):  
    
    def objective(trial) -> int:
        import torch
        import gc
        import math
        from sklearn.metrics import mean_squared_error
        import joblib
        X_train, label_train, X_test, label_test= joblib.load(path)
       
        torch.cuda.empty_cache()
       
        gc.collect()
        param_rocket = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 77,
            "verbosity": 0,
            "device": "cuda",}
        # XGBoost-Parameter
        param_rocket = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 77,
            "verbosity": 0,
            "device": "cuda",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "dart", "gblinear"]
            ),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        }

        booster = param_rocket["booster"]

        if booster in ["gbtree", "dart"]:
            param_rocket.update(
                {
                    "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                    "gamma": trial.suggest_float("gamma", 0, 10.0),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "grow_policy": trial.suggest_categorical(
                        "grow_policy", ["depthwise"]
                    ),
                }
            )

        if booster == "gbtree":
            param_rocket["sampling_method"] = trial.suggest_categorical(
                "sampling_method", ["uniform"]
            )

        if booster == "dart":
            param_rocket.update(
                {
                    "sample_type": trial.suggest_categorical(
                        "sample_type", ["uniform", "weighted"]
                    ),
                    "normalize_type": trial.suggest_categorical(
                        "normalize_type", ["tree", "forest"]
                    ),
                    "rate_drop": trial.suggest_float("rate_drop", 0.0, 0.5),
                    "skip_drop": trial.suggest_float("skip_drop", 0.0, 0.5),
                }
            )

        if booster == "gblinear":
            param_rocket.update(
                {
                    "updater": trial.suggest_categorical("updater", ["coord_descent"]),
                    "feature_selector": trial.suggest_categorical(
                        "feature_selector", ["shuffle", "cyclic"]
                    ),
                    "top_k": trial.suggest_int("top_k", 0, 100),
                    "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                }
            )

        early_stopping = {
            "rounds": trial.suggest_int("early_stopping_rounds", 10, 80),
            "min_delta": trial.suggest_float("min_delta", 1e-5, 1e-1, log=True),
            "save_best": False,
            "maximize": False,
            "data_name": "validation_0",
            "metric_name": "rmse",
        }

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "validation_0-rmse"
        )
        eval_set = [(X_test, label_test)]

        

       #

        rocket_version = trial.suggest_categorical(
            "rocket_version", ["hydra", "multirockethydra", "multirocket"]
        )

        
        if rocket_version == "hydra":
            h_n_kernels = trial.suggest_int("h_n_kernels", 5000, 10000)
            h_n_groups = trial.suggest_int("h_n_group", 32, 64)

            params = {
                "n_kernels": h_n_kernels,
                "n_groups": h_n_groups,
               "xgb_params": param_rocket,
                "callback_early_stopping": early_stopping,
                "use_optuna": True,
                "optuna_callback": pruning_callback,
                "eval_set": eval_set,
            }

            model = hydra.HydraRegressor(**params)

        elif rocket_version == "multirockethydra":
            mh_n_kernels = trial.suggest_int("mh_n_kernels", 5000, 10000)
            mh_n_groups = trial.suggest_int("mh_n_group", 32, 64)

            params = {
                "n_kernels": mh_n_kernels,
                "n_groups": mh_n_groups,
                "xgb_params": param_rocket,
                "callback_early_stopping": early_stopping,
                "use_optuna": True,
                "optuna_callback": pruning_callback,
                "eval_set": eval_set,
            }

            model = multiRocketHydra.MultiRocketHydraRegressor(**params)

        else:
            mr_n_kernels = trial.suggest_int("mr_n_kernels", 5000, 10000)
            rocket_transform = "rocket"
            mr_max_dilations_per_kernel = trial.suggest_int(
                "mr_max_dilations_per_kernel", 16, 32
            )
            mr_n_features_per_kernel = trial.suggest_int(
                "mr_n_features_per_kernel", 4, 16
            )

            params = {
                "n_kernels": mr_n_kernels,
                "rocket_transform": rocket_transform,
                "max_dilations_per_kernel": mr_max_dilations_per_kernel,
                "n_features_per_kernel": mr_n_features_per_kernel,
                "random_state": 77,
                "n_jobs": 1,
               "xgb_params": param_rocket,
               "callback_early_stopping": early_stopping,
                "use_optuna": True,
                "optuna_callback": pruning_callback,
                "eval_set": eval_set,
            }

            model = multirocket.MultiRocketRegressor(**params)


        del params, param_rocket, eval_set, rocket_version, pruning_callback
        
        torch.cuda.empty_cache()
       
        gc.collect()

        model.fit(X_train, label_train)
        del X_train, label_train
        torch.cuda.empty_cache()
       
        gc.collect()
        pred = model.predict(X_test)
        del X_test, model
        torch.cuda.empty_cache()
       
        gc.collect()

        mse = mean_squared_error(label_test, pred)
        rmse = math.sqrt(mse)

        del mse, pred
        del label_test
        torch.cuda.empty_cache()
       
        gc.collect()

        return rmse
    

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    import joblib
    
    study.optimize(
        objective,
        n_trials=1,
        catch=(NameError, AttributeError, ModuleNotFoundError, TypeError, ValueError),
        gc_after_trial=True,
        callbacks=[cleanup_callback],
    )
   
# Save all for later
def dump_files(files_list, path):
    """
    This function will take the variable name and use it as a file name to dump it in pickle files.

    :param files_list: list of variable names which should be dumped, just copy and paste the names
    :param path: path which you want to dump the files
    :return: None
    """

    for file in files_list:
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        name = [var_name for var_name, var_val in callers_local_vars if var_val is file]
        with open(path + "/" + name[0] + ".pickle", "wb") as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_participant_data(data_dict: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def cleanup_callback(study, trial):
    gc.collect()
    torch.cuda.empty_cache()
   


import argparse
import datetime
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import utils
import datetime
import os


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description='rocket Training and finetuning')
    parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
    parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
    parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
    parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
    parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
    parser.add_argument('--final', default=False, type=bool, help='Run the final models')  
    parser.add_argument('--part', default=0, type=int, help='Which finetuning run to start')
    args = parser.parse_args()
    sampler = TPESampler()
    direction = 'minimize'




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = r'/path/to/data'
    saving_path = r'path/to/folder'
    data, decode_dict = utils.preprare_data(path, 'dl_feat')



    n_iterations = args.trials or 800    

    print('Decode Dict: \n',decode_dict)
    trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
    if args.percent < 100:
        trial_name = trial_name + '_percent_'+str(args.percent)+'_'
        data = utils.reduce_dataframes_size(data, percentage=args.percent, path=saving_path, decode_dict=decode_dict)

    

    trial_name = datetime.datetime.strftime(
        datetime.datetime.today(), "%Y-%m-%d %H-%M-%S"
    )

    if args.finetune:
        # train the  rocket model
        n_iterations =  100

        sampler = TPESampler(multivariate=True, n_startup_trials=4)
        warmup_steps = 7
        name = "regression"
        direction = "minimize"
        device = "cuda"

    participants_tpe = {}
    best_params_list = []

    for participant in data:
        
        train_sets, test_sets = sklearn.model_selection.train_test_split(
            participant, shuffle=False, test_size=0.3
        )
        
        id = utils.decode_string_columns(pd.DataFrame(train_sets[0]['participant'], columns=['participant']), decode_dict=decode_dict)
        
       
        
        X_test, X_train, label_test, label_train, _, _ = utils.personal_datasets_dl(
            train_data=train_sets, test_data=test_sets
        )
        
        


        participants_name = id.iloc[0].item()
        
        

        X_train = dataframe_to_3d_numpy(X_train)
    
        X_test = dataframe_to_3d_numpy(X_test)
       
                
        
        
        import joblib
        joblib.dump((X_train, label_train, X_test, label_test), f"{participants_name}_data.pkl")
        del X_train, label_train, X_test, label_test
        torch.cuda.empty_cache()
       
        gc.collect()


        results_tpe = {}
        storage = "sqlite:///optuna_study_" + participants_name + ".db"
        study = optuna.create_study(
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            direction='maximize',
            study_name=participants_name,
        )
        for i in range(n_iterations):
            p = multiprocessing.Process(
                target=run_trial,
                args=(
                    i,
                    participants_name,
                    storage,
                    f"{participants_name}_data.pkl",
                    sampler,
                    direction
                   
                ),
            )
            p.start()
            p.join()

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial of " + participants_name + ": ")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("Best parameters", study.best_params)
        print("Best value", study.best_value)
        print("Best trial", study.best_trial)
        # best Model
        date = datetime.datetime.strftime(
            datetime.datetime.today(), "%Y-%m-%d-%H-%M-%S"
        )

        with open(
            "best_parameters_kfold_rocket" + participants_name + "_" + date + ".txt",
            "w",
        ) as f:
            print(study.best_params, file=f)
        results_tpe["best_params"] = study.best_params
        results_tpe["best_value"] = study.best_value
        results_tpe["best_trial"] = study.best_trial
        results_tpe["study"] = study
        participants_tpe[participants_name] = results_tpe
        best_params_list.append(study.best_params)    
        

    dump_files([participants_tpe, best_params_list], path)


if __name__ == "__main__":
    main()
