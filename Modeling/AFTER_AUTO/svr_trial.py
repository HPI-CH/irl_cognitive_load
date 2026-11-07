import copy
import platform

from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import pandas as pd 
import os
import matplotlib.pyplot as plt
from typing import Optional
import inspect
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR, LinearSVR, NuSVR
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
import gc

import torch


class svm:
    def __init__(self, path_data:str, path_label:str , path:str=os.path.abspath(os.getcwd()), name:str='',name_label:str='label', num_folds:int=4,trial_name: str='regression_cognitve_load', optuna:bool=False, init:bool=False, trials:int=1000, fold:int = 0) -> None:
        """Initialization of the SVR for the autoencoder feature

        Parameters
        ----------
        path_data : str
            where to load the data from
        path_label : str
            where the label from the autoencoder is saved
        path : str, optional
            where to save the data, by default os.path.abspath(os.getcwd())
        name : str, optional
            Name of the trial, by default ''
        name_label : str, optional
            name of the label, by default 'label'
        num_folds : int, optional
            number of folds, by default 4
        trial_name : str, optional
            name of the trial, by default 'regression_cognitve_load'
        """
        
        self.parameters = {'max_iter':1000}
       
       
        self.path_data = path_data
        self.path_label = path_label        
        if init: 
            # empty dict for no parameters -> intitial run with default params
            self.parameters = {'max_iter':1000}
        if 'tcn' in self.path_data: 
            self.tcn=True; trial_name=trial_name+'tcn'
            self.parameters = {'max_iter':1000}
        else: self.tcn=False
        self.fold = fold  
        path = path +'/svm_'+trial_name+'_LOOCV'
        self.N_ITER = trials or 1000
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.optuna = optuna
        self.path_save_preds =r'/path/to/folder'

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


  
    def train(self,data,parameters)->tuple[dict, dict]:
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
        num_folds = len(data)
        kfold = KFold(n_splits=num_folds)
        predictions = {}
        scores= []
        best_model = {}
        modeling = []
        for fold,  (train_idx, test_idx) in enumerate(kfold.split(data)):
            if not fold == self.fold: continue
            train_set = np.array(data, dtype=object)[train_idx]
            test_set = np.array(data, dtype=object)[test_idx]
            train_set = pd.concat(train_set.tolist(), axis=0)
            test_set = pd.concat(test_set.tolist(), axis=0)
            test_set = test_set.dropna(how='any')
            train_set= train_set.dropna(how='any')
            label_train = train_set['label']
            train_set.drop('label', axis=1, inplace=True)
            label_test = test_set['label']
            test_set.drop('label', axis=1, inplace=True)
            scaler = StandardScaler()
            train_set = scaler.fit_transform(train_set)
            test_set = scaler.transform(test_set)
            
            gc.collect()
            print('Fold {} ... ✧ ✶𐙚⋆°｡⋆♡'.format(fold))
           
            try:
                if parameters['svm'] == 'SVR':
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = SVR(**param)
                elif parameters['svm'] == 'NuSVR': 
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            model = NuSVR(**param)
                else:
                    param = copy.deepcopy(parameters)
                    del param['svm']
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = LinearSVR(**param)  
            except KeyError:     
                model = SVR()    
            model.fit(train_set,label_train.ravel())
            compare = pd.DataFrame()
            pre=model.predict(test_set)
            modeling.append(model)


        
            mse = mean_squared_error(label_test, pre)
            mae = mean_absolute_error(label_test, pre)
            mape = mean_absolute_percentage_error(label_test, pre)
            r = r2_score(label_test, pre)
            print('MSE: ', mse)
            print('RMSE: ', math.sqrt(mse))
            print('MAE: ', mae)
            
            comp = {}
            
            comp['rmse'] = math.sqrt(mean_squared_error(label_test, pre))
            comp['mse'] = mean_squared_error(label_test, pre)
            comp['mape'] = mean_absolute_percentage_error(label_test, pre)
            comp['mae'] = mean_absolute_percentage_error(label_test, pre)
            comp['r2'] = r2_score(label_test, pre)
            comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
            compare['rmse'] = math.sqrt(mse)
            compare['mape'] = mape
            compare['mae'] = mae
            compare['r2'] = r
            
            name = 'Fold_'+str(fold)
            predictions[name]=compare
            predictions[name+'_pred'] = pre
            predictions[name+'_truth'] = label_test
            # Speichern des Models, nutzlos ich weiß aber nun gut
            if fold ==self.fold:
                best_model['rmse']=math.sqrt(mse)
                best_model['mape']= mape
                best_model['mae'] = mae
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = model

            elif mae < best_model['mae']:
                best_model['rmse']=math.sqrt(mse)
                best_model['mape']= mape
                best_model['mae'] = mae
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = model
              
            self.visualize_results(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold)
            self.visualize_results_color(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold)
            self.dump_files_for_shap([predictions], self.path_save_preds, suffix=self.trial_name+'svm_auto_loso_Fold'+str(self.fold))    
        self.dump_files_for_shap([predictions], self.path_save_preds, suffix=self.trial_name+'svm_auto_loso')    
        return predictions, best_model

    def visualize_results(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int) -> None:
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
        plt.title('Error for Fold ' + str(fold) + ' of Participant ', wrap=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        name = 'Error_metrics_fold_' + nombre + str(fold)  + '.png'
        plt.savefig(self.path + '/' + name, bbox_inches='tight')
        plt.close(fig12)

        temp = pd.DataFrame({'True label': ground_truth, 'Predicted label': pred})
        fig22 = plt.figure(figsize=(40,7))
        import seaborn as sns
        sns.regplot(data=temp, x='True label', y='Predicted label', marker='o', scatter_kws={'color': '#E98DAD'}, line_kws={'color': '#c74777'})
        plt.xlabel('True Cognitive Load')
        plt.ylabel('Predicted Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_'  + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#e84a7f', label='True Cognitive Load', marker='*')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#E98DAD', label='Predicted Cognitive Load', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_line_' + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)
        fig323cc = plt.figure(figsize=(40,7))
     
        sns.lineplot(data=temp['True label'], color='#E84a7f', label='True True Cognitive Load',marker='o' , linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Cognitive load',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Cognitive Load')
        plt.title('Cognitve load estimated from psycho-physiological data vs. subjective perception of participant '+ ' from fold '+str(fold), wrap=True)
     
        name_scatter = 'cl_true_pred_participant_with_line_'+'_Fold_'+str(fold)+nombre+'.pdf'
        plt.savefig(self.path+'/'+name_scatter)
        plt.close(fig323cc)



    def visualize_results_color(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int) -> None:
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
        plt.title('Error for Fold ' + str(fold) + ' of Participant ', wrap=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        name = 'Error_metrics_foldf_' + nombre + str(fold)  + '.pdf'
        plt.savefig(self.path + '/' + name, bbox_inches='tight')
        plt.close(fig12)

        temp = pd.DataFrame({'True label': ground_truth, 'Predicted label': pred})
        fig22 = plt.figure(figsize=(40,7))
        import seaborn as sns
        sns.regplot(data=temp, x='True label', y='Predicted label', marker='o', scatter_kws={'color': '#377eb8'}, line_kws={'color': '#4daf4a'})
        plt.xlabel('True Cognitive Load')
        plt.ylabel('Predicted Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participantf_' + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#377eb8', label='True Cognitive Load', marker='o')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#4daf4a', label='Predicted Cognitive Load', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_linef_' + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)

        fig323cc = plt.figure(figsize=(40,7))
        sns.lineplot(data=temp['True label'], color='#377eb8', label='True Cognitive Load', marker='o',linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Cognitive load',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Cognitive Load')
        plt.title('Cognitive load estimated from psycho-physiological data vs. subjective perception of participant ' + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cl_true_pred_participant_with_linef_' + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323cc)



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

            
    def resize_array(self,arr: np.ndarray, new_shape: tuple) -> np.ndarray:
        """
        Resizes an array to the specified shape by repeating or truncating the values in the third dimension.

        Parameters
        ----------
        arr : np.ndarray
            The input array with three dimensions.
        new_shape : tuple
            The desired shape (must have three dimensions).

        Returns
        -------
        np.ndarray
            The resized array with the specified shape.

        Example
        -------
        >>> arr = np.full((1, 63, 384), 8)
        >>> resized_arr = resize_array(arr, (1, 63, 128))
        >>> print(resized_arr.shape)
        (1, 63, 128)
        """
        # Check the shape of the input array
        assert len(arr.shape) == 3, "Input array must have three dimensions"
        
        # Resize the array to the specified shape
        resized_arr = arr[:, :, :new_shape[2]]
        
        return resized_arr
        

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
            'max_iter': trial.suggest_int('random_state', 700, 200000),

            }
            model = LinearSVR(**param)

        # NuSVR
        elif svm == 'NuSVR': 

            param = {
                'nu': trial.suggest_float('nu', 0.001, 1, log=True),
                'C': trial.suggest_float('C', 0.001, 10^2),
                'max_iter': trial.suggest_int('random_state', 700, 200000),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 12)
                


            }
            model = NuSVR(**param)
        else : 

        # SVR
            param = {
                'epsilon' : trial.suggest_float('epsilon', 0.00001, 1.0, log=True),
                'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
                'C': trial.suggest_float('C', 0.001, 10^2),
                'kernel': trial.suggest_categorical('kernel', [ 'rbf', 'linear', 'sigmoid']),
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

    def make_train_test_sets(self, data:dict|pd.DataFrame, test_size:float=0.3, normalizing:bool=True, label:Optional[pd.DataFrame]=pd.DataFrame()):
        """Make the train and test sets for Optuna.

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
      

        data = pd.concat(data, axis='index')
        data = data.dropna(how = 'any')
        label = data['label']
        data=data.drop('label', axis='columns')
        X_train, X_test,label_train, label_test  = train_test_split(data, label, test_size=test_size, shuffle=True)



        return X_train,label_train, X_test, label_test


    def optimizing_with_optuna(self, n_iterations:int , sampler, 
                            warmup_steps:int, study_name:str='SVM regression',
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


    def finetune (self):

        sampler = TPESampler()
        warmup_steps = 7
        self.name= self.trial_name+'_optuna'
      
        self.X_train,self.label_train, self.X_test, self.label_test = self.make_train_test_sets(self.data, test_size=0.3, normalizing=False)
        optuna_results = self.optimizing_with_optuna(self.N_ITER,sampler, warmup_steps=warmup_steps,study_name=self.name)
        self.dump_files([optuna_results],self.path)
        return optuna_results
   
    
    def svm_trial(self)->dict:
        """Will execute the class and train, test the SVM.

        Returns
        -------
        dict
           predicitions
        """
        features = pickle.load(open(self.path_data, 'rb'))
        target = pickle.load(open(self.path_label, 'rb'))
        feature_frames_machine_learning=[]

        for participants in range(len(features)):
            print(np.shape(features[participants]))
            print(np.shape(target[participants]))
            if self.tcn:

                import torch
                import torch.nn.functional as F
                size_features = features[participants].size()
                seq_target = target[participants].size(1)
                upsampled_features = F.interpolate(features[participants], size=seq_target, mode='linear', align_corners=False)# (Batch, Channels, 7680)
                flattened = upsampled_features.permute(0, 2, 1).reshape(-1, size_features[1])

                
                print(np.shape(target[participants]))
                print(target[participants].shape)
                print(type(target[participants]))
                sizelabel= target[participants].size(1)*target[participants].size(0)
                print(sizelabel)
                reshaped_label = target[participants].numpy()
                reshaped_label = np.reshape(reshaped_label, shape=(sizelabel,))
                print(len(reshaped_label))
                
                temp = pd.DataFrame(flattened)
                print(len(temp))
                temp['label'] = reshaped_label
                print(len(temp))
                feature_frames_machine_learning.append(temp)
            else:
                reshaped_data = features[participants].reshape(-1, features[participants].size(2)).numpy()
                reshaped_label = target[participants].reshape(-1).numpy()
                temp = pd.DataFrame(reshaped_data)
                temp['label'] = reshaped_label
                feature_frames_machine_learning.append(temp)
            
            

        self.data = feature_frames_machine_learning    
        if 'percent' in self.trial_name:
            
            data = utils.reduce_dataframes_size(feature_frames_machine_learning, percentage=50)
            self.data = data
  
        if self.optuna:
            optuna_results = self.finetune()
            self.dump_files([optuna_results], self.path)
            self.parameters = optuna_results['best_params']
            preds_svm, bestmodel =  self.train(self.data,self.parameters)
            self.dump_files([preds_svm], self.path)


        else:    
            preds_svm, bestmodel =  self.train(self.data,self.parameters)
            self.dump_files([preds_svm], self.path)
       
        
        return preds_svm
    






import argparse
import datetime
import torch
parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')   
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--fold', default=0, type=int, help='Folds which will run')
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features=r'/path/to/features'
target= r'/path/to/target'
path = r'/path/to/data'


if args.percent < 100: # this is an option to finetune the svm if you have a lot of data, you can choose a percentange it will take this part from each of the persons
    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_percent_'+str(args.percent) # for the name
   
if args.finetune:

    n_iterations = args.trials or 1000    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'finetuning_' # for the name
    svm_model = svm(path_data=features, path_label=target, path=path,trial_name=trial_name,optuna=True, init=False)
    svm_model.svm_trial()
    


elif args.initial:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_init_' # for the name
    svm_model = svm(path_data=features, path_label=target, path=path,trial_name=trial_name,optuna=False, init=True)
    svm_model.svm_trial()

   
   
elif args.final:
    print('running final version of the model:')
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_final_run_svm_loocv_auto_tcn' # for the name
    svm_model = svm(path_data=features, path_label=target, path=path,trial_name=trial_name,optuna=False, init=False, fold=args.fold)
    svm_model.svm_trial()
       
         
else: 

    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_normal' # for the name
    svm_model = svm(path_data=features, path_label=target, path=path,trial_name=trial_name,optuna=False, init=False)
    svm_model.svm_trial()



