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
from sklearn.svm import SVR, NuSVR, LinearSVR, SVC, NuSVC
import optuna
from optuna.samplers import TPESampler
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


class svm:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()),optuna:bool=False, name:str='',trial_name: str='regression_cognitve_load', decode_dict:dict = None, trials:int =1000, tsne:bool=False, init=False) -> None:
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
        self.parameters = {'max_iter':1000}
        self.name = trial_name
        if 'percent' in trial_name and '50' in trial_name:
             self.parameters = {'max_iter':1000}
        elif  'percent' in trial_name and '75' in trial_name:
             self.parameters = {'max_iter':1000}
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
            features = ['EDA_iqr' ,'ACC_Vector_magnitude' ,
                        'x_max',
                          'PPG_hr_new',
                'ACC_norm_x acc1_max', 
                'PPG_median' ,
                'HRV_MadNN', 
                'PPG_iqr', 
                'HRV_MCVNN',
                'PPG_max',
                'PPG_slope' ,
                'ACC_signal_magnitude',
                'z_iqr', 
                'EDA_min',
                'y_min',
                'ACC_norm_x acc1_mean' ,
                'z_mean', 'TEMP_min' ,'y_mean', 'PPG_Pulse_Peaks',
                'HRV_VHF' ,'EDA_Phasic' ,'PPG_tpp_new' ,'EDA_median', 'z_max', 'HRV_Prc80NN',
                'HRV_LnHF' ,'y', 'TEMP_slope', 'HRV_TP' ,'ACC_pos_freq_y' ,'x' ,'TEMP_mean',
                'HRV_MaxNN' ,'PPG_min', 'y_iqr', 'ACC_pos_freq_x', 'ACC_norm_x acc1_min',
                'TEMP_iqr', 'y_max', 'x_mean' ,'HRV_SDNN', 'HRV_CVNN', 'EDA_Clean' ,'EDA_Tonic',
                'EDA_slope', 'z',
                'ACC_norm_amp_z',
                'EDA_max' ,
                'EDA_Acum',
                'PPG_Clean',
                'cat_1',
                'x_iqr', 
                'ACC_norm_x acc1_iqr', 
                'z_min', 
                'ACC_pos_freq_z']
            features = ['TEMP_min', 'y_min', 'PPG_hr_new', 'x', 'x_min', 'z', 'TEMP_iqr', 'ACC_norm_x acc1_min', 'ACC_norm_x acc1_iqr', 'y', 'y_mean', 'z_min', 'z_mean', 'ACC_Vector_magnitude', 'x_mean', 'HRV_CVSD', 'PPG_min', 'PPG_Pulse_Peaks', 'ACC_norm_x acc1_mean', 'EDA_Acum', 'y_max', 'PPG_median', 'PPG_iqr', 'z_max', 'TEMP_max', 'HRV_TP', 'HRV_MaxNN', 'ACC_pos_freq_x', 'EDA_max', 'TEMP_slope', 'PPG_max', 'HRV_LnHF', 'ACC_signal_magnitude', 'y_iqr', 'PPG_HR_mean', 'ACC_norm_amp_y', 'ACC_norm_amp_x', 'HRV_RMSSD', 'TEMP_median', 'TEMP_mean', 'PPG_slope', 'HRV_pNN20', 'ACC_pos_freq_y', 'z_iqr', 'ACC_norm_x acc1_max', 'HRV_MeanNN']
            if 'features' in self.trial_name:
                 train_set = train_set[features]
                 test_set = test_set[features]
            if self.tsne:
                utils.plot_tsne(X_train=train_set, y_train=label_train, person_labels_train=train_set_ids['id'],X_test=test_set, y_test=label_test, person_labels_test=id['id'], name='Set of '+str(fold),path= self.path)
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
            model.fit(train_set,label_train.to_numpy())
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
            if fold ==0:
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
            self.dump_files([testing_data, trainings_data, modeling], self.path)    
            self.visualize_results(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=participant)
            self.visualize_results_color(ground_truth=label_test, pred=pre, nombre='_svm', compare=comp, fold=fold, participant=participant)
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



    def visualize_results_color(self, ground_truth: np.array, pred: np.array, nombre: str, compare: pd.DataFrame, fold: int, participant: str) -> None:
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

            
    

   
    
    def svm_trial(self)->dict:
        """Will execute the class and train, test the SVM.

        Returns
        -------
        dict
           predicitions
        """
        device = 'cuda'

        preds_svm, bestmodel =  self.train(self.data, self.parameters)
        
        self.dump_files([preds_svm], self.path)
       
        
        return preds_svm
    

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
            #  'random_state': 77,
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
    def run_rfe(self):
            """
            Runs an iterative version of sklearn's RFE. Here it iterates through all features and selects for each of the number the best features
            In addition, the NuSVR has an customized implementation, as it natively does not have the feature_importance module.

            """
            X_train, X_test,label_train, label_test = utils.loso_finetuning_ml(self.data)
            parameters = self.parameters
            print(X_train.columns)
            print(X_train)
          
            try:
                if parameters['svm'] == 'SVR':
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            gc.collect()
                            torch.cuda.empty_cache()
                            svr = SVR(**param)
                            results, names, selected_features_names = [], [], []
                            print(len(X_train.columns))
                            for i in range(10,len(X_train.columns)+1):
                                xg = SVR(**param)
                                selection, xg, score_dict = self.rfe_svm_non_linear_pseudosamples_by_name(svm=xg, X=X_train, y = label_train, K=i, Q=21)
                               
                                xg.fit(X_train[selection], label_train)
                                pre=xg.predict(X_test[selection])
                                mae = mean_absolute_error(label_test, pre)
                                # selection = np.array(X_train.columns)[model.support_]
                                results.append(mae)
                                selected_features_names.append(selection)
                                names.append(str(i))





                elif parameters['svm'] == 'NuSVR': 
                            param = copy.deepcopy(parameters)
                            del param['svm']
                            del param['random_state']
                            gc.collect()
                            torch.cuda.empty_cache()
                            results, names, selected_features_names = [], [], []

                            for i in range(10,len(X_train.columns)+1):
                                xg = NuSVR(**param)
                               
                                try:
                                    selection, xg, score_dict = self.rfe_svm_non_linear_pseudosamples_by_name(svm=xg, X=X_train, y = label_train, K=i, Q=21)
                                   
                                    xg.fit(X_train[selection], label_train)
                                    pre=xg.predict(X_test[selection])
                                    mae = mean_absolute_error(label_test, pre)
                                    # selection = np.array(X_train.columns)[model.support_]
                                    results.append(mae)
                                    selected_features_names.append(selection)
                                    names.append(str(i))
                                    print('Number of features', i, ' MAE:', mae)
                                    print( 'Features: ', selection)
                                except Exception as e:
                                     print('Failed to collect {}'.format(i)) 
                                     continue   

                else:
                    param = copy.deepcopy(parameters)
                    del param['svm']
                    gc.collect()
                    torch.cuda.empty_cache()
                    results, names, selected_features_names = [], [], []

                    for i in range(1,len(X_train.columns)+1):
                        xg = LinearSVR(**param)
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

            except KeyError:     
                 
                params = copy.deepcopy(self.parameters)
                results, names, selected_features_names = [], [], []

                for i in range(1,len(X_train.columns)+1):
                    xg = SVR()
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


    
    def run_rfe2(self):
        """
        Determine the optimal SVM feature subset for each fold.

        The method applies pseudo-sample-based recursive feature elimination
        to obtain a complete feature ranking for each cross-validation fold.
        It then trains and evaluates an SVM using the first 1, 2, 3, up to
        all features from that ranking.

        The feature subset with the lowest mean absolute error on the test
        set is selected as the best subset for the respective fold.

        Results are stored in the following instance attributes:

        `self.svm_rfe_fold_results`
            DataFrame containing the best result and all evaluated results
            for each fold.

        `self.svm_rfe_best_features`
            Dictionary mapping each fold to its best feature subset.

        `self.svm_rfe_feature_rankings`
            Dictionary mapping each fold to its complete feature ranking.

        `self.svm_rfe_elimination_history`
            Dictionary containing the elimination history for each fold.

        Returns
        -------
        pd.DataFrame
            Fold-level results including the optimal number of features,
            selected feature names, and corresponding mean absolute error.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If an unsupported SVM type is specified or the input data do not
            contain valid features.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        The optimal feature count is selected according to the mean absolute
        error on the test data of the same fold. Therefore, the minimum test
        error should not be interpreted as an independent estimate of model
        generalization performance.
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

        if len(self.data) < 5:
            raise ValueError(
                "At least five data elements are required for 5-fold "
                "cross-validation."
            )

        cross_validator = KFold(
            n_splits=5,
            shuffle=False
        )

        parameters = copy.deepcopy(self.parameters)

        svm_type = parameters.pop(
            "svm",
            "SVR"
        )

        # SVR and NuSVR do not accept random_state.
        parameters.pop("random_state", None)
                     
                     
        try:
                        del parameters['random_state']
        except KeyError: pass  
        
        try:
            del parameters['svm']
        except KeyError: pass 
        columns_to_drop = [
            "id",
            "task",
            "cat_2",
            "participant",
            "PPG_fixed_peaks",
            "PPG_Hr"
        ]

        all_fold_results = []

        self.svm_rfe_fold_results = None
        self.svm_rfe_best_features = {}
        self.svm_rfe_feature_rankings = {}
        self.svm_rfe_elimination_history = {}

        for fold, (train_indices, test_indices) in enumerate(
            cross_validator.split(self.data)
        ):
            data_copy = copy.deepcopy(self.data)
            gc.collect()
            # Kein deepcopy notwendig, solange die einzelnen
            # Datensätze nicht innerhalb der Methode verändert werden.
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

            y_train = np.asarray(label_train).ravel()
            y_test = np.asarray(label_test).ravel()

            if X_train.empty:
                raise ValueError(
                    f"Fold {fold}: The training set contains no features."
                )

           
            X_test = X_test.reindex(
                columns=X_train.columns
            )

            if X_train.isna().any().any():
                raise ValueError(
                    f"Fold {fold}: The training data contain missing values."
                )

            if X_test.isna().any().any():
                raise ValueError(
                    f"Fold {fold}: The test data contain missing values "
                    "or do not provide all training features."
                )

            if svm_type == "SVR":
                base_svm = SVR(**parameters)

            elif svm_type == "NuSVR":
           
                base_svm = NuSVR(**parameters)

            else:
                raise ValueError(
                    "The pseudo-sample RFE implementation supports only "
                    f"'SVR' and 'NuSVR'. Received: '{svm_type}'."
                )

            print(f"SVM model: {svm_type}")
            print(f"Number of available features: {X_train.shape[1]}")
            print("Calculating the complete feature ranking...")

            (
                _,
                _,
                feature_scores,
                feature_ranking,
                elimination_history
            ) = self.rfe_svm_non_linear_pseudosamples_by_name(
                svm=base_svm,
                X=X_train,
                y=y_train,
                K=1,
                Q=21
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

            mae_values = []
            feature_counts = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for number_of_features in range(
                1,
                len(feature_ranking) + 1
            ):
                selected_features = feature_ranking[
                    :number_of_features
                ]

                if svm_type == "SVR":
                    model = SVR(**parameters)
                else:
                    model = NuSVR(**parameters)

                try:
                    model.fit(
                        X_train.loc[:, selected_features],
                        y_train
                    )

                    predictions = model.predict(
                        X_test.loc[:, selected_features]
                    )

                    mae = mean_absolute_error(
                        y_test,
                        predictions
                    )

                    feature_counts.append(number_of_features)
                    mae_values.append(float(mae))
                    selected_feature_sets.append(
                        selected_features.copy()
                    )

                    print(
                        f"Number of features: {number_of_features:3d} | "
                        f"MAE: {mae:.4f}"
                    )

                except Exception as error:
                    print(
                        f"Evaluation failed for {number_of_features} "
                        "features."
                    )
                    print(f"Error: {error}")

            if not mae_values:
                raise RuntimeError(
                    f"Fold {fold}: No feature subset could be evaluated."
                )

            best_result_index = int(
                np.argmin(mae_values)
            )

            best_mae = mae_values[
                best_result_index
            ]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ]

            self.svm_rfe_best_features[fold] = (
                best_features.copy()
            )

            fold_result = {
                "fold": fold,
                "participant": participant,
                "svm_type": svm_type,
                "best_number_of_features": best_number_of_features,
                "best_features": best_features,
                "best_mae": best_mae,
                "feature_ranking": feature_ranking,
                "feature_scores": feature_scores,
                "all_feature_counts": feature_counts,
                "all_mae_values": mae_values,
                "all_selected_feature_sets": selected_feature_sets
            }

            all_fold_results.append(fold_result)

            print("\n" + "-" * 70)
            print(f"Best result for fold {fold}")
            print("-" * 70)
            print(
                f"Best number of features: "
                f"{best_number_of_features}"
            )
            print(f"Best MAE: {best_mae:.4f}")
            print("Best features:")

            for feature_name in best_features:
                print(f"  - {feature_name}")

            ranked_results = sorted(
                zip(
                    feature_counts,
                    mae_values,
                    selected_feature_sets
                ),
                key=lambda result: result[1]
            )

            print(f"\nResults ranked by MAE for fold {fold}:")

            for (
                number_of_features,
                mae,
                selected_features
            ) in ranked_results:
                print(
                    f"Number of features: {number_of_features}, "
                    f"MAE: {mae:.4f}"
                )
                print(f"Features: {selected_features}")

            self.dump_files_for_shap(
                [
                    mae_values,
                    selected_feature_sets,
                    feature_counts,
                    fold_result
                ],
                path=self.path,
                suffix=f"Fold_{fold}_{svm_type}"
            )

            self.dump_files(
                [
                    mae_values,
                    selected_feature_sets,
                    feature_counts,
                    fold_result
                ],
                path=self.path
            )

        self.svm_rfe_fold_results = pd.DataFrame(
            all_fold_results
        )

        self.dump_files_for_shap(
            [all_fold_results],
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
                    "best_number_of_features",
                    "best_mae"
                ]
            ].to_string(index=False)
        )

        return self.svm_rfe_fold_results



    def quantile_grid(self, col_values: np.ndarray, q_min:float, q_max:float, Q:int) -> np.ndarray:
            """
            - calculates the quantiles for the rfe method for the svm from this paper doi:
            - it generates a grid of eqaully spaced quantiles in the range of the max and min quantile (used to remove extremes)
            - after generating the grid it calculates the quantiles and returns them

            Parameters
            ----------
            col_values : np.ndarray
                _description_
            q_min : float
                _description_
            q_max : float
                _description_
            Q : int
                _description_

            Returns
            -------
            np.ndarray
                _description_
            """
            qs = np.linspace(q_min, q_max, Q)
            return np.quantile(col_values, qs)



    def rfe_svm_non_linear_pseudosamples_by_name(
        self,
        svm,                   # SVC (classification) or SVR (regression)
        X: pd.DataFrame,       # standardized features (n_samples x n_features), column names used
        y: pd.Series,          # target (binary or continuous)
        K: int,                # number of features to keep
        Q: int = 21,           # number of pseudo-sample points per feature
        q_min: float = 0.02,   # lower quantile bound (drop extreme tails)
        q_max: float = 0.98,   # upper quantile bound
        baseline_value: float = 0.0,  # value for all non-varied features (0.0 for standardized data)
    ) -> list:
        """
        Paper-faithful SVM-RFE with pseudo-samples (Sanz et al., 2018), tracking column names throughout.

        Steps per iteration:
        - Fit SVM on current feature subset (by names).
        - For each feature name f in the subset:
            * Build Q pseudo-samples: vary only feature f over Q equally spaced quantiles of X[f];
                set all other features in the current subset to baseline_value, as the data is standarized it would be set to the mean 0 oder median 1. Here feafault is 0
            * Evaluate decision_function (SVC) or predict (SVR) on pseudo-samples.
            * Compute MAD = 1.4826 * median(|d - median(d)|).
        - Remove feature with smallest MAD.
        - Repeat until K features remain.

        Returns:
        List[str]: column names of the selected K features (sorted by their original order in X).
        """
        features_and_score = {}
    
        remaining_names = list(X.columns)

        # Helper: Q equally spaced quantiles from q_min..q_max for a 1D column (numpy array)
        

        # Decide evaluation function based on estimator type
        is_classifier = isinstance(svm, SVC) or isinstance(svm, NuSVC)
        is_regressor  = isinstance(svm, SVR) or isinstance(svm, NuSVR)
        if not (is_classifier or is_regressor):
            raise TypeError("svm must be sklearn.svm.SVC (classification) or sklearn.svm.SVR (regression) or svm must be sklearn.svm.NuSVC (classification) or sklearn.svm.NuSVR (regression).")

        
        while len(remaining_names) > K:
        
            X_sub = X.loc[:, remaining_names].values 
            svm.fit(X_sub, y)

            m = len(remaining_names)
            mads = np.zeros(m) 

            
            for j, fname in enumerate(remaining_names):
            
                z_vals = self.quantile_grid(X[fname].values, q_min=q_min, q_max=q_max, Q=Q)   
                X_pseudo = np.zeros((Q, m)) if baseline_value == 0 else np.full((Q,m), baseline_value)
                X_pseudo[:, j] = z_vals                    
            
                if is_classifier:
                    d = svm.decision_function(X_pseudo)
                    d = np.asarray(d)
                    
                else: 
                    d = svm.predict(X_pseudo).ravel()

            # constant was set by the paper 
                mads[j] = 1.4826 * np.median(np.abs(d - np.median(d)))
                features_and_score[fname] =mads[j]

            
            drop_pos = int(np.argmin(mads))
            remaining_names.pop(drop_pos)

        
        selected_set = list(set(remaining_names))
        
        return selected_set, svm, features_and_score

         


    def finetune (self):
        """Finetunes the SVR using optuna.

        Returns
        -------
        dict
            Results of the hyper-parameter-finetuing with optuna, contains best parameter, best run and so on
        """
        sampler = TPESampler()
        warmup_steps = 7
        self.name= self.name+'_optuna'
        X_train, X_test,self.label_train, self.label_test = utils.loso_finetuning_ml(self.data)
        optuna_results = self.optimizing_with_optuna(self.N_ITER,sampler, warmup_steps=warmup_steps,study_name=self.name)
        self.dump_files([optuna_results],self.path)
        return optuna_results



    
import argparse
import datetime

parser = argparse.ArgumentParser(description='SVM Training and finetuning')
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--tsne', default=False, type=bool,help='If you want to run tsne on a set of datapoints')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--rfe', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--rfe2', default=False, type=bool,help='If you want to run rfe on a set of datapoints')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')  
parser.add_argument('--final_selected_features', default=False, type=bool, help='Run the final models')   
args = parser.parse_args()




path = r'/path/to/data' 

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
    col_list, col = svm_model.run_rfe2()
elif args.rfe2:
    trial_name= trial_name+'_with_rfe' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    col_list, col = svm_model.run_rfe()    
elif args.final:
    trial_name= trial_name+'_endgultig_final_run_svm_loocv_with_tsne_allerletzterrun' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    preds= svm_model.svm_trial()  
elif args.final_selected_features:
    trial_name= trial_name+'_final_run_svm_loocv_with_tsne_and_rfe_features_allerletzterrun' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=True)
    preds= svm_model.svm_trial()          
else: 
    trial_name= trial_name+'_normal' # for the name
    svm_model = svm(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, tsne=False)
    preds= svm_model.svm_trial()
        
