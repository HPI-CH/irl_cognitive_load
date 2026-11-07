import platform
import optuna
import pandas as pd 
import os
import matplotlib.pyplot as plt
from typing import Optional
import inspect
import pickle
import sklearn
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import mean_absolute_error, zero_one_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import math
import inspect
import pickle
from sklearn.model_selection import KFold
import utils
import copy
import gc
import torch


class svm:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), name:str='',name_label:str='label', num_folds:int=4,trial_name: str='regression_cognitve_load',trials:int=1000, decode_dict:dict=None, optuna:bool=False, init:bool=False) -> None:
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



        
       
        self.num_folds = num_folds
        if init:self.parameters = [{}]
        self.name = name
        self.name_label = name_label
        self.decode_dict = decode_dict
        self.trials = trials or 1000
        path = path +'/svm_'+trial_name+'personalized'
        self.data = data
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.optuna = optuna
        self.parameters=[{}]
        if init: self.parameters = [{}]
       
                
     
        self.path_save_preds =r'/path/to/folder'




  
    def train(self,data,num_folds,parameters)->tuple[dict, dict]:
        """Trains the SVM model, it will split the data into test and trainings data according the number of folds. 
        Will call also plotting functions to viszualize the the predictions the ground truth.

        Parameters
        ----------
        data : dict
            contains the training data
        num_folds : int
            mumber of folds for the K-Fold
        parameters : dict
            specified parameters of the svm, are defined in the Intialization of this class or through Optuna
        path : str
            where to save the images and data to
        name_label : str
            name of the label column

        Returns
        -------
        
        tuple[dict,dict]
            predictions, scores of the best model, scores
        """


       
        predictions = {}
        best_models ={}
        
        skip =  False
        c = 0
        for participant in data:

            splitter = KFold(n_splits=num_folds)
            
            pred = {}
            best_model = {}
            scores = []
            id = utils.decode_string_columns(pd.DataFrame(participant['id'], columns=['id']), decode_dict=self.decode_dict)
            name = id.iloc[0].item()
            print('Creating model for participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(name))
            for fold,(train_idx, test_idx) in enumerate(splitter.split(participant)):
                train_set = participant.iloc[train_idx]
                test_set = participant.iloc[test_idx]   
                test_set, train_set, label_test1, label_train1 = utils.personal_datasets_ml(train_data=train_set, test_data=test_set)
                label_train = copy.deepcopy(label_train1)
                label_test = copy.deepcopy(label_test1)
                if len(np.unique(label_train))<=2:
                     label_train_cl = label_train
                     label_test_cl = label_test
                else:     
                    label_train_cl = utils.map_labels(labels=label_train)
                    label_test_cl = utils.map_labels(label_test)
                label_compl = pd.concat([pd.DataFrame(label_test_cl), pd.DataFrame(label_train_cl)], axis='index')
                u, oc = utils.occurence_count(label_compl)
                if len(u)==1: 
                     skip = True
                     continue
                print('------------------------------- ✧ ✶𐙚⋆°｡⋆♡ Fold No.: {} ♡⋆｡°⋆𐙚✶ ✧ -------------------------------'.format(str(fold)))
                
                if (len(parameters)>1 and isinstance(parameters, list)):
                    if parameters[c]['svm'] == 'SVC':
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = SVC(**param)
                    elif parameters[c]['svm'] == 'NuSVC': 
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        try:
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                        except KeyError:pass    
                        model = NuSVC(**param)
                    else:
                        param = copy.deepcopy(parameters[c])
                        try:
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                        except KeyError:pass    
                        
                        #
                        model = LinearSVC(**param)  
                elif (len(parameters)==1 and isinstance(parameters, list)):
                    params =  parameters[0]
                    try: 
                        
                        if parameters[0]['svm'] == 'SVC':
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVC(**param)
                        elif parameters[0]['svm'] == 'NuSVC': 
                            param = copy.deepcopy(parameters[0])
                            if 'nu2' in param:
                                param['nu'] = param['nu2']  # Wert von 'nu2' auf 'nu' setzen
                                del param['nu2'] 
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVC(**param)
                        else:
                            param = copy.deepcopy(parameters[0])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = LinearSVC(**param)  
                    except KeyError:
                        model = SVC()

                else: 
                    try:         
                        model = SVC(**parameters) 
                    except TypeError: model = SVC()    

                model.fit(train_set,label_train_cl)
                compare = pd.DataFrame()
                y_proba=model.predict(test_set)
                pre = y_proba


            
                
                
                comp = {}
                print(y_proba)
                
                mae = mean_absolute_error(label_test_cl, pre)
                f1 =f1_score(label_test_cl, pre)
                comp['acc'] = accuracy_score(label_test_cl, pre)
                comp['f1'] = f1_score(label_test_cl, pre)
                comp['mae'] = mean_absolute_error(label_test_cl, pre)
                comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
                acc = accuracy_score(label_test_cl, pre)
                print('ACC: ', acc)
                f1s = f1_score(label_test_cl, pre)
                print('F1S: ', f1s)
                mae = mean_absolute_error(label_test_cl, pre)
                print('MAE: ', mae)
                

                compare['mae'] = mae
                compare['f1s'] = f1s
                compare['acc'] = acc            
                name_fold = 'Fold_'+str(fold)
                pred[name_fold]=compare
                pred[name_fold+'pred_org'] = pre
                pred[name_fold+'_truth'] = label_test_cl
                pred[name_fold + "_truth_org"] = label_test
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
                    self.visualize_results(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=name)
                    self.visualize_results_color(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=name)        
            c = c+1
            if skip: c=c-1; skip =False
            predictions[str(name)]=pred    
            best_models[name] =best_model
        self.dump_files_for_shap([predictions],path=self.path_save_preds, suffix=self.trial_name+'svm_personalized')          
        return predictions, best_model


    def visualize_results(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int, participant: str) -> None:
        """
        Visualizing the results. This will plot a point plot of the predictions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.

        Parameters
        ----------
        ground_truth : np.array
            True values of the Category.
        pred : np.array
            Predicted values of the Category.
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
        plt.xlabel('True Category')
        plt.ylabel('Predicted Category')
        plt.title('Predicted software development category vs. actual category ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#e84a7f', label='True Category', marker='*')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#E98DAD', label='Predicted Category', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Category')
        plt.title('Predicted software development category vs. actual category ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_line_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)
        fig323cc = plt.figure(figsize=(40,7))
     
        sns.lineplot(data=temp['True label'], color='#E84a7f', label='True True Category',marker='o' , linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Category',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Category')
        plt.title('Predicted software development category vs. actual category '+str(participant)+ ' from fold '+str(fold), wrap=True)
     
        name_scatter = 'cat_true_pred_participant_with_line_'+str(participant)+'_Fold_'+str(fold)+nombre+'.pdf'
        plt.savefig(self.path+'/'+name_scatter)
        plt.close(fig323cc)
        plt.close('all')


    def visualize_results_color(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int, participant: str) -> None:
        """
        Visualizing the results. This will plot a point plot of the predictions and a bar plot of the error metrics: MAE, MAPE, RMSE, MSE.
        The colors used in the plots are suitable for individuals with red-green color blindness.

        Parameters
        ----------
        ground_truth : np.array
            True values of the Category.
        pred : np.array
            Predicted values of the Category.
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
        plt.xlabel('True Category')
        plt.ylabel('Predicted Category')
        plt.title('Predicted software development category vs. actual category ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participantf_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#377eb8', label='True Category', marker='o')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#4daf4a', label='Predicted Category', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Category')
        plt.title('Predicted software development category vs. actual category ' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)

        fig323cc = plt.figure(figsize=(10,7))
        sns.lineplot(data=temp['True label'], color='#377eb8', label='True Category', marker='o',linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Category',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Category')
        plt.title('Predicted software development category vs. actual category' + str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_with_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323cc)
        plt.close('all')




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
        label_train_cl = utils.map_labels(labels=self.label_train)
        label_test_cl = utils.map_labels(self.label_test)
        label_compl = pd.concat([pd.DataFrame(label_test_cl), pd.DataFrame(label_train_cl)], axis='index')
        u, oc = utils.occurence_count(label_compl)
        #2*min(m+,m-)/m
        upper_bound_nu = min(0.6,2*min(oc[0], oc[1])/len(label_compl))
        

        if svm == 'LinearSVC':
            # LinearSVC: kein epsilon, kein random_state
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
            # Validitäts-Guard: Wenn loss == 'hinge', setze dual zwingend True
            if params['loss'] == 'hinge':
                params['dual'] = True
            model = LinearSVC(**params)

        elif svm == 'NuSVC':
            # NuSVC: steuert Regularisierung über nu (statt C)
            params = {
                'nu': trial.suggest_float('nu', 1e-4, upper_bound_nu, log=True),  # (0,1], praktikabel: 0.01–0.5
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12),  # relevant nur bei 'poly', sonst ignoriert
                'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                'class_weight': 'balanced',
                'probability': trial.suggest_categorical('probability', [True, False]),
                'decision_function_shape': trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr']),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 200000),
                'random_state': 77,  # erlaubt bei SVC/NuSVC (für probability-Schätzung/Interna)
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


        
        model.fit(self.X_train,label_train_cl)
        pred = model.predict(self.X_test)
        

        mae = mean_absolute_error(label_test_cl, pred)
        f1 =f1_score(label_test_cl, pred)
        return f1




    def optimizing_with_optuna(self, n_iterations:int , sampler, 
                            warmup_steps:int, study_name:str='SVM Classifaction',
                            direction:str ='maximize', test_size:float=0.2, val_size: Optional[float]=None)-> dict: 
        """Finetune the autoencoder using optuna to find the best parameters

        Parameters
        ----------
        n_iterations : int
            number of trials (500 works the best on the HPC cluster)
        sampler : Optuna.Sampler
            Sampler which will choose the parameter combinations from the search space
        warmup_steps : int
            number of warmup steps
        study_name : str, optional
            name of the study, by default 'regression'
        direction : str, optional
            whether to maximize or minimize the return value of the trail, by default 'minimize'
        test_size : float, optional
            size of the testing dataset, by default 0.2
        val_size : Optional[float], optional
            size of the validation set, by default None

        Returns
        -------
        dict
            paramters
        """
        participants_tpe = {}
        best_params_list = []
        
        for participant in self.data:
            
            

            train_set, test_set = sklearn.model_selection.train_test_split(participant, shuffle=True, test_size=test_size)
            X_test, X_train, label_test, label_train = utils.personal_datasets_ml(train_data=train_set, test_data=test_set)
            id = utils.decode_string_columns(pd.DataFrame(participant['id'], columns=['id']), decode_dict=self.decode_dict)
            participants_name = id.iloc[0].item()
            label_train_cl = utils.map_labels(labels=label_train)
            label_test_cl = utils.map_labels(label_test)
            label_compl = pd.concat([pd.DataFrame(label_test_cl), pd.DataFrame(label_train_cl)], axis='index')
            u, oc = utils.occurence_count(label_compl)
            if len(u)==1:continue
            self.X_train = X_train
            self.label_train = label_train
            self.X_test = X_test
            self.label_test = label_test

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
            date = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")

            with open('best_parameters_kfold_svm' +participants_name +'_'+date+'.txt', 'w') as f:
                print(study.best_params, file=f)
            results_tpe['best_params'] = study.best_params
            results_tpe['best_value'] = study.best_value
            results_tpe['best_trial'] = study.best_trial
            results_tpe['study'] = study
            participants_tpe[participants_name] = results_tpe
            best_params_list.append(study.best_params)

        return participants_tpe, best_params_list

    def run_rfe(self):
            c = 0
            for participant in self.data:
                train_set, test_set = sklearn.model_selection.train_test_split(participant, shuffle=True)
                X_test, X_train, label_test, label_train = utils.personal_datasets_ml(train_data=train_set, test_data=test_set)
                
                parameters = self.parameters
                id = utils.decode_string_columns(pd.DataFrame(participant['id'], columns=['id']), decode_dict=self.decode_dict)
                participants_name = id.iloc[0].item()
                print('participant ', participants_name)
                
                if (len(parameters)>1 and isinstance(parameters, list)):
                    if parameters[c]['svm'] == 'SVC':
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = SVC(**param)
                    elif parameters[c]['svm'] == 'NuSVC': 
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        del param['random_state']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = NuSVC(**param)
                    else:
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = LinearSVC(**param)  
                elif (len(parameters)==1 and isinstance(parameters, list)):
                    params =  parameters[0]
                    try: 
                        c = 0
                        if parameters[c]['svm'] == 'SVC':
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVC(**param)
                        elif parameters[c]['svm'] == 'NuSVC': 
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVC(**param)
                        else:
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = LinearSVC(**param)  
                    except KeyError:
                        model = SVC()

                else: 
                    try:         
                        model = SVC(**parameters) 
                    except TypeError: model = SVC()    
                rfe = RFE(estimator=model)

                selector = rfe.fit(X_train, label_train)
                columns = X_train.columns.tolist()
                columns_bool = selector.support_
                from itertools import compress
                col_list = list(compress(columns, columns_bool))  
                #     col_list = data_kaggle_feature1.columns.to_list()[:-1]
                print(col_list)
                ranking_list = list(compress(columns, selector.ranking_))
                print(ranking_list)
                c= c+1
            return col_list, ranking_list

    
    def svm_trial(self)->dict:
        """Carries out the svm's training and testing

        Returns
        -------
        dict
            predicted values
        """

        if self.optuna:
            sampler = optuna.samplers.TPESampler()
            participants_tpe, best_params_list=self.optimizing_with_optuna(self.trials, sampler=sampler, warmup_steps=7, study_name=self.trial_name, direction='maximize')
            self.dump_files([participants_tpe, best_params_list], self.path_save_preds)
            self.parameters = best_params_list
        preds_svm, bestmodel =  self.train(self.data, self.num_folds,self.parameters)
        
        self.dump_files([preds_svm], self.path)
       
        
        return preds_svm
    




import argparse
import datetime
import torch
parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')   
args = parser.parse_args()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'/path/to/data'
# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print(decode_dict)
if args.percent < 100:
    data = utils.reduce_dataframes_size(data, percentage=args.percent)

if args.finetune:
# train the  xgb model
    n_iterations = args.trials or 1000   
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning_perso_cat' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, trials=n_iterations, decode_dict=decode_dict, optuna=True)
    preds = svms.svm_trial()
elif args.rfe:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning_perso_cat' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=True)
    col_list, ranking_list = svms.run_rfe()
elif args.initial:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_initial_perso_cat' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False, init=True)
    preds= svms.svm_trial()
elif args.final:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_final_run_endgültig_svm_cat_personalized' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False)
    preds= svms.svm_trial()
         
else: 
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_normal_perso_cat' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False)
    preds= svms.svm_trial()
    

