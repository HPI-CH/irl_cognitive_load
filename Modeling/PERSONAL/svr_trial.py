import platform
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
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
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
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), name:str='',name_label:str='label', num_folds:int=5,trial_name: str='regression_cognitve_load',trials:int=1000, decode_dict:dict=None, optuna:bool=False, init:bool=False) -> None:
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



        
        self.parameters ={ 'epsilon': 1.3042970425570691e-05,
                           'tol': 1.1812070386652136e-06,
                         'C': 3.0852475785465217, 
                         'kernel': 'linear',
                           'gamma': 'auto', 
                           'degree': 12}
        self.num_folds = num_folds
        if init:self.parameters = [{'max_iter':1000}]
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
                test_set,train_set, label_test, label_train = utils.personal_datasets_ml(train_data=train_set, test_data=test_set)
               
                print('------------------------------- ✧ ✶𐙚⋆°｡⋆♡ Fold No.: {} ♡⋆｡°⋆𐙚✶ ✧ -------------------------------'.format(str(fold)))
                if (len(parameters)>1 and isinstance(parameters, list)):
                    if parameters[c]['svm'] == 'SVR':
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = SVR(**param)
                    elif parameters[c]['svm'] == 'NuSVR': 
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        del param['random_state']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = NuSVR(**param)
                    else:
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = LinearSVR(**param)  
                elif (len(parameters)==1 and isinstance(parameters, list)):
                    params =  parameters[0]
                    try: 
                        c = 0
                        if parameters[c]['svm'] == 'SVR':
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVR(**param)
                        elif parameters[c]['svm'] == 'NuSVR': 
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVR(**param)
                        else:
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = LinearSVR(**param)  
                    except KeyError:
                        model = SVR()

                else: 
                    try:         
                        model = SVR(**parameters) 
                    except TypeError: model = SVR()    

                model.fit(train_set,label_train.to_numpy())
                compare = pd.DataFrame()
                pre=model.predict(test_set)
           

            
                mse = mean_squared_error(label_test, pre)
                mae = mean_absolute_error(label_test, pre)
                mape = mean_absolute_percentage_error(label_test, pre)
                r = r2_score(label_test, pre)
                print('MSE: ', mse)
                print('RMSE: ', math.sqrt(mse))
                print('MAE: ', mae)
                
                
                compare['rmse'] = math.sqrt(mse)
                compare['mape'] = mape
                compare['mae'] = mae
                compare['r2'] = r
                comp = {}
            
                comp['rmse'] = math.sqrt(mean_squared_error(label_test, pre))
                comp['mse'] = mean_squared_error(label_test, pre)
                comp['mape'] = mean_absolute_percentage_error(label_test, pre)
                comp['mae'] = mean_absolute_percentage_error(label_test, pre)
                comp['r2'] = r2_score(label_test, pre)
                comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
                name_fold = name+'_Fold_'+str(fold)
                pred[name_fold]=compare
                pred[name_fold+'_pred'] = pre
                pred[name_fold+'_truth'] = label_test
                # Speichern des Models, nutzlos ich weiß aber nun gut
                if fold ==0:
                    best_model['rmse']=math.sqrt(mse)
                    best_model['mape']= mape
                    best_model['mae'] = mae
                
                    best_model['model'] = model

                elif mae < best_model['mae']:
                    best_model['rmse']=math.sqrt(mse)
                    best_model['mape']= mape
                    best_model['mae'] = mae
                    best_model['model'] = model
                
                self.visualize_results(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=name)
                self.visualize_results_color(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=name)        
            c = c+1
            predictions[str(name)]=pred    
            best_models[name] =best_model
        self.dump_files_for_shap([predictions],path=self.path_save_preds, suffix=self.trial_name+'svm_personalized')          
        return predictions, best_model



    def visualize_results(self, ground_truth: np.array, pred:np.array, nombre:str, compare:pd.DataFrame, fold:int, participant:str)-> None:
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

        plt.close('all')

    def visualize_results_color(self, ground_truth: np.array, pred:np.array, nombre:str, compare:pd.DataFrame, fold:int, participant:str)-> None:
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

            
    

    def objective(self, trial)->float:
        """Generate a trial for finetuning. Sets up the search space and will determine the best parameters over a number of trials

        Parameters
        ----------
        trial : Optuna.Trial
            trial to be optimized

        Returns
        -------
        float
            value which marks the sucess of the trial; will be reported

        """
        svm = trial.suggest_categorical('svm', ['LinearSVR', 'SVR', 'NuSVR'])
        model = None
        if svm == 'LinearSVR':
            param ={
            'epsilon' : trial.suggest_float('epsilon', 0.001, 1.0, log=True),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'C': trial.suggest_float('C', 0.001, 10^2),
            'loss': trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'intercept_scaling': trial.suggest_float('intercept_scaling', 0.8, 2.0),
            'random_state': 77,
            'max_iter': trial.suggest_int('random_state', 700, 2000)
            }
            model = LinearSVR(**param)

        # NuSVR
        elif svm == 'NuSVR': 

            param = {
                'nu': trial.suggest_float('nu', 0.001, 1, log=True),
                'C': trial.suggest_float('C', 0.001, 10^2),
                'max_iter': trial.suggest_int('random_state', 700, 2000),
            #  'random_state': 77,
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['poly', 'rbf', 'linear', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12),
                


            }
            model = NuSVR(**param)
        else : 

        # SVR
            param = {
                'epsilon' : trial.suggest_float('epsilon', 0.00001, 1.0, log=True),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'C': trial.suggest_float('C', 0.001, 10^2),
                'kernel': trial.suggest_categorical('kernel', ['poly', 'rbf', 'linear', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12),
                'max_iter': trial.suggest_int('max_iter', 830, 2500)
                
            }
            model = SVR(**param)



        model.fit(self.X_train,self.label_train.values.ravel())
        pred = model.predict(self.X_test)
        

        mae = mean_absolute_error(self.label_test, pred)
        r2 = r2_score(self.label_test, pred)
        mse =mean_squared_error(self.label_test, pred)
        rmse = math.sqrt(mse)
        mape =mean_absolute_percentage_error(self.label_test, pred)
        return mae



    def optimizing_with_optuna(self, n_iterations:int , sampler, 
                            warmup_steps:int, study_name:str='SVM regression',
                            direction:str ='minimize', test_size:float=0.2, val_size: Optional[float]=None)-> dict: 
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
                    if parameters[c]['svm'] == 'SVR':
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = SVR(**param)
                    elif parameters[c]['svm'] == 'NuSVR': 
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        del param['random_state']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = NuSVR(**param)
                    else:
                        param = copy.deepcopy(parameters[c])
                        del param['svm']
                        gc.collect()
                        torch.cuda.empty_cache()
                        model = LinearSVR(**param)  
                elif (len(parameters)==1 and isinstance(parameters, list)):
                    params =  parameters[0]
                    try: 
                        c = 0
                        if parameters[c]['svm'] == 'SVR':
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVR(**param)
                        elif parameters[c]['svm'] == 'NuSVR': 
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVR(**param)
                        else:
                            param = copy.deepcopy(parameters[c])
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = LinearSVR(**param)  
                    except KeyError:
                        model = SVR()

                else: 
                    try:         
                        model = SVR(**parameters) 
                    except TypeError: model = SVR()    
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
            participants_tpe, best_params_list=self.optimizing_with_optuna(self.trials, sampler=sampler, warmup_steps=7, study_name=self.trial_name, direction='minimize')
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




path = r'/path/to/data' # 
# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print('Decode Dict: \n',decode_dict)
trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
if args.percent < 100:
    trial_name = trial_name + '_percent_'+str(args.percent)+'_'
    data = utils.reduce_dataframes_size(data,  percentage=args.percent, path=path, decode_dict=decode_dict)


if args.finetune:

    n_iterations = args.trials or 1000   
    trial_name= trial_name+'during_tuning_perso' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, trials=n_iterations, decode_dict=decode_dict, optuna=True)
    preds = svms.svm_trial()

elif args.rfe:
    trial_name= trial_name+'during_tuning_perso' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=True)
    col_list, ranking_list = svms.run_rfe()
elif args.initial:
    trial_name= trial_name+'_initial_perso' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False, init=True)
    preds= svms.svm_trial()
elif args.final:
    trial_name= trial_name+'_final_endgultig_run_personalized_svm_allerletzerrun' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False)
    preds= svms.svm_trial()
else: 
    trial_name= trial_name+'_normal_perso' # for the name
    svms = svm(data=data, path=path, trial_name=trial_name, num_folds=5, decode_dict=decode_dict, optuna=False)
    preds= svms.svm_trial()
    

