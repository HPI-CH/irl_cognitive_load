# Imports 
import copy
import gc
import platform
from sklearn.metrics._plot import precision_recall_curve
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

from typing import Optional
import statistics
from statistics import mean
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_recall_curve
import math
import inspect
import math
import inspect
import pickle
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, f_classif
import numpy as np
import optuna
from optuna.samplers import TPESampler
import utils

from sklearn.metrics import precision_recall_curve



class xgboost:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), trial_name: str='regression_cognitve_load', optuna:bool=True, trials:int=1000, init:bool=False,
                 decode_dict:dict=None, kfold:bool=False, number_fold:int=5, 
                 select_features:bool=False,
                 features:list=[]) -> None:
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
    
        

        
        
        # finetuned parameters
        self.parameters = {
            }
            
        self.early_stopping = {
            'rounds': 30,
            'min_delta':7e-4,
            'save_best':False,
            'maximize':False,
            'data_name':'validation_1',
            'metric_name':'logloss'
        }
        self.kfold = kfold
        self.number_folds = number_fold if not np.nan or 0 else 5 # not sure if this works like this but i do not want nan or anything else
            
        self.optuna =optuna
        self.data = data
        self.trails = trials
        self.decode_dict = decode_dict
        path = path +'/xg_boost_'+trial_name+'_LOOCVCL'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.selected_features = select_features
        self.features = features or []
     

        
 
        self.path_save_preds =r''




   
    


   
    
   
    
   

    # Darstellen der wichtigen Features
    def importance_feature(self,score_pd,model, path: Optional[str] = '', name: Optional[str]=None, save: bool =False, top_k: Optional[int]=30) -> pd.DataFrame:  
        """Feature Importance. Plots the results of the feature importance tool of XGBoost.

        Parameters
        ----------
        score_pd : pd.DataFrame
            Contains the achieved MAE
        model : xgb.XGBModel
            
        path : Optional[str], optional
            where to save the plot, by default ''
        name : Optional[str], optional
            name of the plot, by default None
        save : bool, optional
            if the plot should be saved, by default False
        top_k : Optional[int], optional
            number of important features, be careful when changing it, if you have less features it will throw an error, by default 30

        Returns
        -------
        pd.DataFrame
            returns the updated score_pd frame and a frame with the feature importance scores
        """
        
        
        feature_important = model.get_booster().get_score(importance_type='weight')
        fi = pd.DataFrame(feature_important, index=[0])
        results = model.evals_result()['validation_1']['error']
        fi['error']=mean(results)
        
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        scores_table = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        temp = pd.DataFrame(data=values, index =keys, columns  =['score'])
        fi.sort_index(axis=1, inplace=True) 
        score_pd = pd.concat([score_pd, fi], ignore_index=True) 
        if not top_k: top_k = 30; 
        if not name: name='Feature Importance'; 
        if save:
            scores_table.to_csv(path+'/'+name+'_XG'+'.csv')
            temp.nlargest(n = top_k, columns="score").plot(title='Feature Importance',kind='barh', figsize = (20,10), xlabel='Feature Score', ylabel='Features', color ='#fe83cc') ## plot top 40 features
            plt.savefig(path+'/'+ name+'_XG'+'.png',bbox_inches = 'tight')
        return scores_table, score_pd
    
  


    def train(self,data,parameters, path,top_k=30):
        """Trains the XGBoost model, it will split the data into test and trainings data according the number of folds. 
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
        top_k : int, optional
            number of the most important features, by default 30

        Returns
        -------
        
        tuple[list, dict, xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]
            list of scores, predictions, (feature importance), best performing model, scores of the best model, scores
        """
        if self.kfold: kfold = KFold(n_splits=self.number_folds)
        else: kfold = KFold(n_splits=5)
        predictions = {}
        scores= []
        best_model = {}
        score_pd = pd.DataFrame()
        
        modeling = []
        for fold,  (train_idx, test_idx) in enumerate(kfold.split(data)):
            gc.collect()
            data2 = copy.deepcopy(data)
            train_data = np.array(data2, dtype=object)[train_idx].tolist()
            test_data = np.array(data2, dtype=object)[test_idx].tolist()
            
            train_set, test_set,label_train,label_test = utils.loso_datasets_ml(train_data=train_data, test_data=test_data)

            early_stopping = xgb.callback.EarlyStopping(**self.early_stopping)
            if self.kfold:
                id = fold
                participant = fold
                print('Fold {} ✧ ✶𐙚⋆°｡⋆♡'.format(fold))
            else:
                id = utils.decode_string_columns(pd.DataFrame(test_set['id'], columns=['id']), decode_dict=self.decode_dict)
                participant = id.iloc[0].item()
                print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡'.format(fold,participant))
            print(f"cat2:{np.unique(train_set['cat_2'])} ")    
            train_set = utils.clean_up_features(train_set, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])
            test_set = utils.clean_up_features(test_set, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])
            label_train_cl = utils.map_labels(labels=label_train)
            label_test_cl = utils.map_labels(label_test)
            
            self.features  = xgbc_rfe = [[
            "TEMP_min",
            "TEMP_mean",
            "TEMP_median",
            "TEMP_max",
            "cat_1"
        ],
        [
            "cat_1",
            "EDA_max",
            "EDA_Clean",
            "EDA_median",
            "EDA_Tonic",
            "z",
            "z_mean",
            "EDA_min",
            "ACC_Vector_magnitude",
            "ACC_norm_x acc1_mean",
            "ACC_signal_magnitude",
            "EDA_Acum",
            "y_iqr",
            "z_iqr",
            "TEMP_min"
        ],
        [
            "EDA_max",
            "EDA_Tonic",
            "EDA_Clean",
            "EDA_median",
            "EDA_min",
            "cat_1",
            "TEMP_min",
            "TEMP_mean",
            "TEMP_median",
            "TEMP_max",
            "EDA_Acum",
            "EDA_iqr",
            "ACC_norm_x acc1_mean",
            "ACC_Vector_magnitude",
            "y_iqr",
            "ACC_signal_magnitude",
            "TEMP_iqr",
            "z_iqr",
            "x_mean",
            "x",
            "ACC_norm_x acc1_iqr",
            "y",
            "y_mean",
            "z",
            "z_mean",
            "z_min",
            "y_max",
            "x_iqr",
            "y_min",
            "ACC_norm_x acc1_max",
            "ACC_norm_amp_y",
            "x_min",
            "z_max",
            "ACC_norm_x acc1",
            "ACC_norm_amp_z",
            "HRV_Prc80NN",
            "PPG_Pulse_Peaks",
            "ACC_norm_amp_x",
            "EDA_Phasic",
            "HRV_MedianNN",
            "HRV_MeanNN",
            "PPG_tpp_new",
            "PPG_hr_new",
            "HRV_MaxNN",
            "HRV_SDNN",
            "HRV_CVNN",
            "HRV_RMSSD",
            "HRV_SDSD",
            "PPG_median"
        ],
        [
            "cat_1",
            "z",
            "z_mean",
            "TEMP_min",
            "EDA_Acum",
            "TEMP_mean"
        ],
        [
            "TEMP_min",
            "TEMP_mean",
            "TEMP_median",
            "TEMP_max",
            "cat_1",
            "z",
            "z_mean",
            "y_iqr"
        ]]


            # isinstance(obj, list) and all(isinstance(x, list) for x in obj)
            if isinstance(self.features, list) and all(isinstance(x, list) for x in self.features): # for each of the folds get a differnt feature set.
                 train_set = train_set[self.features[fold]]
                 test_set = test_set[self.features[fold]]
            elif self.selected_features and len(self.features)>0:
                 train_set = train_set[self.features]
                 test_set = test_set[self.features]


            pos = (label_train_cl == 1).sum()
            neg = (label_train_cl == 0).sum()
            spw = neg / max(1, pos)
            parameters["scale_pos_weight"]=spw

            label_compl = pd.concat([pd.DataFrame(label_test_cl), pd.DataFrame(label_train_cl)], axis='index')
            u, oc = utils.occurence_count(label_compl)
            print(len(label_test_cl))
            print(len(label_test))
            print(len(label_train))
            print(len(label_train_cl))
            print(len(train_set.index))
            print(np.unique(label_test_cl))
            print(np.unique(label_train_cl))
            
            parameters.update({
                  "objective": 'binary:logistic',
                           'class_weight':'balanced',
                            "eval_metric": ['logloss', 'auc', 'error'],
                            'device':'cuda'})
            try:
                        md = copy.deepcopy(parameters['min_delta'])
                        r = copy.deepcopy(parameters['rounds'])
                        del parameters['min_delta'], parameters['rounds']
                        early_stopping = {
                            'rounds': r,
                            'min_delta':md,
                            'save_best':False,
                            'maximize':False,
                            'data_name':'validation_1',
                            'metric_name':'error'
                        }
                        early_stopping = xgb.callback.EarlyStopping(**early_stopping)
                        xg = xgb.XGBClassifier(**parameters,callbacks = [early_stopping])
            except KeyError:
                             xg = xgb.XGBClassifier(**parameters)

                
            
            xg.fit(train_set,label_train_cl,
                eval_set=[(train_set, label_train_cl),(test_set, label_test_cl)],
                    verbose=1)
            compare = pd.DataFrame()
            
            proba_test = xg.predict_proba(test_set)
            print(len(proba_test))
            print(len(test_set))
            print(len(label_test_cl))
            prob = proba_test
            proba_test = proba_test[:, 1]  # P(y=1)
            mae_ord = mean_absolute_error(label_test_cl, proba_test)
            print("Ordinal-MAE (auf P(y=1)):", mae_ord)
            comp = {}
            pre = np.argmax(prob, axis=1)
            comp['acc'] = accuracy_score(label_test_cl, pre)
            precision, recall, thresholds = precision_recall_curve(label_test_cl, proba_test)
            
            f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            best_idx = int(np.argmax(f1_scores))
            best_thresh = float(thresholds[best_idx])
            best_f1 = float(f1_scores[best_idx])
            comp['f1'] =best_f1
            comp['mae'] = mae_ord
            comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
            print('ACC: ', accuracy_score(label_test_cl, pre))
            print('F1S: ', best_f1)
            
           
            results = xg.evals_result()
           
            compare['auc_train'] = results['validation_0']['auc']
            compare['logloss_train'] = results['validation_0']['logloss']
            compare['mae_train'] = results['validation_0']['error']
            compare['auc_test'] = results['validation_1']['auc']
            compare['logloss_test'] = results['validation_1']['logloss']
            compare['mae_test'] = results['validation_1']['error']
         
           
          
            name = 'Fold_'+str(fold)
            predictions[name]=compare
            predictions[name+'_pred'] = prob
            predictions[name+"_raw_prob"] =proba_test
            predictions[name+'_truth'] = label_test_cl
            predictions[name + "_truth_org"] = label_test

           
            print('saving stuff')
            if fold ==0:
                best_model['min_mean_merror'] = mean(results['validation_1']['error'])
                best_model['min_mean_auc'] = mean(results['validation_1']['auc'])
                best_model['min_mean_mlogloss'] =  mean(results['validation_1']['logloss'])
               
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = xg

            elif statistics.mean(results['validation_1']['error']) < best_model['min_mean_merror']:
              
                best_model['min_mean_mae'] = mean(results['validation_1']['error'])
                best_model['min_mean_auc'] = mean(results['validation_1']['auc'])
                best_model['min_mean_mlogloss'] =  mean(results['validation_1']['logloss'])
                
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = xg
            
            
            scores_table, score_pd = self.importance_feature(score_pd,xg,path,name= 'Features Importance',top_k=top_k)
            scores.append(scores_table)
        print(best_model.keys())
        best_scores, _ = self.importance_feature(pd.DataFrame(),best_model['model'], path, 'best_model', save=True)
        score_pd.fillna(0, inplace=True)
        self.dump_files_for_shap([predictions], self.path_save_preds, suffix=self.trial_name+'xgb_loocv')
        return scores, predictions, best_model, best_scores, score_pd

    
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
                with open(rf"{path}/{name[0]}{suffix}.pickle", 'wb') as handle:
                    pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


    def objective(self,trial):
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
            "objective": 'binary:logistic',
            'class_weight':'balanced',
            "eval_metric": 'logloss',
            "seed": 77,
            "verbosity": 0,
            "device": 'cuda',
            "n_estimators" : trial.suggest_int('n_estimators', 20, 1000),
            'min_child_weight':trial.suggest_float('min_child_weight', 0.01, 1.0, log=True),
           "booster": trial.suggest_categorical("booster", ["gbtree", "dart", 'gblinear']),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
           'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10,100)
        }
        
        if param["booster"] == "gbtree":
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["max_depth"] = trial.suggest_int("max_depth", 1, 300)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            param['sampling_method'] = trial.suggest_categorical('sampling_method', ['uniform'])
            
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["max_depth"] = trial.suggest_int("max_depth", 1, 300)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        
        if param["booster"] =="gblinear":
            param['updater'] = trial.suggest_categorical('updater', ['coord_descent' ])  
           
            param['feature_selector'] = trial.suggest_categorical('feature_selector',['shuffle', 'cyclic'])
           
            param['top_k'] = trial.suggest_int('top_k',0 ,100)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
          
        early_stopping_dict = {
            'rounds': trial.suggest_int('rounds', 10,50),
            'min_delta':trial.suggest_float('min_delta', 1e-5,2, log=True ),
            'save_best':False,
            'maximize':False,
            'data_name':'validation_0',
            'metric_name':'logloss'
        }
        early_stopping = xgb.callback.EarlyStopping(**early_stopping_dict)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-logloss")
        label_train_cl = utils.map_labels(labels=self.label_train)
        label_test_cl = utils.map_labels(self.label_test)
        print(type(self.train_set))
        print(type(label_train_cl))
        pos = (label_train_cl == 1).sum()
        neg = (label_train_cl == 0).sum()
        spw = neg / max(1, pos)
        param["scale_pos_weight"]=spw

        model = xgb.XGBClassifier(**param,  callbacks = [pruning_callback, early_stopping])
        
        
        model.fit(self.train_set,label_train_cl,
                eval_set=[(self.test_set, label_test_cl)],
                    verbose=0)
        
        
        proba1 = model.predict_proba(self.test_set)[:, 1]

       
        mae_ord = mean_absolute_error(label_test_cl, proba1)

        
        precision, recall, thresholds = precision_recall_curve(label_test_cl, proba1)
        
        f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        return best_f1



    def optimizing_with_optuna(self,n_iterations:int , sampler, 
                           warmup_steps:int, study_name:str='biclassification',
                           direction:str ='maximize', test_size:float=0.2, val_size: Optional[float]=None)-> dict: 
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
        X_train = utils.clean_up_features(X_train, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])
        X_test = utils.clean_up_features(X_test, extra_columns_to_drop=['id', 'task', 'cat_2', 'participant'])

        self.train_set = X_train
        self.label_train = label_train
        self.test_set = X_test
        self.label_test = label_test
        results_tpe={}
        # Create the study
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(
            min_resource=30, reduction_factor=6
            ),
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

        results_tpe['best_params'] = study.best_params
        results_tpe['best_value'] = study.best_value
        results_tpe['best_trial'] = study.best_trial
        results_tpe['study'] = study
        return results_tpe    

    
    def run_rfe_number(self):
        """
        Determine the optimal number of features for an XGBoost classifier.

        The method performs five-fold cross-validation on `self.data`. Within
        each fold, recursive feature elimination is executed once to obtain a
        complete feature ranking. Separate XGBoost classifiers are then fitted
        using the first 1, 2, 3, up to all ranked features.

        For every feature subset, the classification threshold that maximizes
        the F1 score is determined from the precision-recall curve. The feature
        subset with the highest F1 score is selected as the best subset for the
        respective fold.

        Results are stored in the following instance attributes:

        `self.xgb_rfe_fold_results`
            DataFrame containing the best feature count, selected features,
            F1 score, probability MAE, and classification threshold for every
            fold.

        `self.xgb_rfe_best_features`
            Dictionary mapping each fold to its best feature subset.

        `self.xgb_rfe_feature_rankings`
            Dictionary mapping each fold to its complete feature ranking.

        `self.xgb_rfe_all_results`
            Dictionary containing the results for every tested feature count
            in each fold.

        Returns
        -------
        pd.DataFrame
            Fold-level results containing the optimal number of features,
            selected feature names, best F1 score, probability MAE, and
            classification threshold.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If the input data contain fewer than five elements, no features,
            missing values, or a non-binary classification target.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        The optimal feature count and classification threshold are selected
        using the test data of the respective fold. Consequently, the maximum
        F1 score is not an independent estimate of generalization performance.

        Running RFE once with one retained feature and `step=1` produces a
        complete elimination ranking. This ranking is reused to evaluate all
        possible feature counts, avoiding repeated RFE computations.
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
                "At least five data elements are required for "
                "five-fold cross-validation."
            )

        cross_validator = KFold(
            n_splits=5,
            shuffle=False
        )

        columns_to_drop = [
            "id",
            "task",
            "cat_1",
            "participant",
            "PPG_fixed_peaks",
            "PPG_Hr"
        ]

        # Create the parameters once instead of copying them in every
        # feature-count iteration.
        model_parameters = copy.deepcopy(self.parameters)

        # RFE does not provide an evaluation set to XGBoost.
        model_parameters.pop("early_stopping_rounds", None)

        self.xgb_rfe_fold_results = None
        self.xgb_rfe_best_features = {}
        self.xgb_rfe_feature_rankings = {}
        self.xgb_rfe_all_results = {}

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

            # Ensure identical feature names and feature order.
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

            unique_train_labels = np.unique(y_train)
            unique_test_labels = np.unique(y_test)

            if not set(unique_train_labels).issubset({0, 1}):
                raise ValueError(
                    f"Fold {fold}: This implementation requires binary "
                    f"training labels encoded as 0 and 1. Received: "
                    f"{unique_train_labels.tolist()}."
                )

            if not set(unique_test_labels).issubset({0, 1}):
                raise ValueError(
                    f"Fold {fold}: This implementation requires binary "
                    f"test labels encoded as 0 and 1. Received: "
                    f"{unique_test_labels.tolist()}."
                )

            number_of_available_features = X_train.shape[1]

            print(
                f"Number of available features: "
                f"{number_of_available_features}"
            )

           

            print("Calculating the complete RFE feature ranking...")

            ranking_estimator = xgb.XGBClassifier(
                **model_parameters
            )

            ranking_selector = RFE(
                estimator=ranking_estimator,
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

            self.xgb_rfe_feature_rankings[fold] = (
                feature_ranking.copy()
            )

            print("\nFeature ranking:")

            for rank, feature_name in enumerate(
                feature_ranking,
                start=1
            ):
                print(f"{rank:3d}. {feature_name}")

           
            feature_counts = []
            f1_values = []
            mae_values = []
            best_thresholds = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for number_of_features in range(
                1,
                number_of_available_features + 1
            ):
                selected_features = feature_ranking[
                    :number_of_features
                ]

                classifier = xgb.XGBClassifier(
                    **model_parameters
                )

                try:
                    classifier.fit(
                        X_train.loc[:, selected_features],
                        y_train
                    )

                    positive_class_probabilities = (
                        classifier.predict_proba(
                            X_test.loc[:, selected_features]
                        )[:, 1]
                    )

                    probability_mae = mean_absolute_error(
                        y_test,
                        positive_class_probabilities
                    )

                    (
                        precision,
                        recall,
                        thresholds
                    ) = precision_recall_curve(
                        y_test,
                        positive_class_probabilities
                    )

               
                    if thresholds.size > 0:
                        f1_scores = (
                            2.0
                            * precision[:-1]
                            * recall[:-1]
                            / np.maximum(
                                precision[:-1] + recall[:-1],
                                1e-12
                            )
                        )

                        best_threshold_index = int(
                            np.argmax(f1_scores)
                        )

                        best_threshold = float(
                            thresholds[best_threshold_index]
                        )

                        best_f1 = float(
                            f1_scores[best_threshold_index]
                        )

                    else:
                        
                        best_threshold = 0.5

                        binary_predictions = (
                            positive_class_probabilities
                            >= best_threshold
                        ).astype(int)

                        best_f1 = float(
                            f1_score(
                                y_test,
                                binary_predictions,
                                zero_division=0
                            )
                        )

                    feature_counts.append(
                        number_of_features
                    )

                    f1_values.append(
                        best_f1
                    )

                    mae_values.append(
                        float(probability_mae)
                    )

                    best_thresholds.append(
                        best_threshold
                    )

                    selected_feature_sets.append(
                        selected_features.copy()
                    )

                    print(
                        f"Number of features: "
                        f"{number_of_features:3d} | "
                        f"F1: {best_f1:.4f} | "
                        f"Probability MAE: {probability_mae:.4f} | "
                        f"Threshold: {best_threshold:.4f}"
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
                key=lambda index: (
                    -f1_values[index],
                    mae_values[index],
                    feature_counts[index]
                )
            )

            best_result_index = result_order[0]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ]

            best_f1 = f1_values[
                best_result_index
            ]

            best_probability_mae = mae_values[
                best_result_index
            ]

            best_threshold = best_thresholds[
                best_result_index
            ]

            self.xgb_rfe_best_features[fold] = (
                best_features.copy()
            )

            fold_evaluation_results = pd.DataFrame({
                "number_of_features": feature_counts,
                "f1_score": f1_values,
                "probability_mae": mae_values,
                "best_threshold": best_thresholds,
                "selected_features": selected_feature_sets
            })

            fold_evaluation_results = (
                fold_evaluation_results
                .sort_values(
                    by=[
                        "f1_score",
                        "probability_mae",
                        "number_of_features"
                    ],
                    ascending=[
                        False,
                        True,
                        True
                    ]
                )
                .reset_index(drop=True)
            )

            self.xgb_rfe_all_results[fold] = (
                fold_evaluation_results
            )

            fold_result = {
                "fold": fold,
                "participant": participant,
                "total_number_of_features": (
                    number_of_available_features
                ),
                "best_number_of_features": (
                    best_number_of_features
                ),
                "best_features": best_features,
                "best_f1_score": best_f1,
                "best_probability_mae": best_probability_mae,
                "best_threshold": best_threshold,
                "feature_ranking": feature_ranking
            }

            all_fold_results.append(
                fold_result
            )

            # -----------------------------------------------------
            # Print the best result
            # -----------------------------------------------------

            print("\n" + "-" * 70)
            print(f"Best result for fold {fold}")
            print("-" * 70)

            print(
                f"Best number of features: "
                f"{best_number_of_features}"
            )

            print(
                f"Best F1 score: {best_f1:.4f}"
            )

            print(
                f"Probability MAE: "
                f"{best_probability_mae:.4f}"
            )

            print(
                f"Best classification threshold: "
                f"{best_threshold:.4f}"
            )

            print("Best features:")

            for feature_name in best_features:
                print(f"  - {feature_name}")

           
        self.xgb_rfe_fold_results = pd.DataFrame(
            all_fold_results
        )

  

        print("\n" + "=" * 70)
        print("Best feature count and feature subset per fold")
        print("=" * 70)

        print(
            self.xgb_rfe_fold_results[
                [
                    "fold",
                    "participant",
                    "total_number_of_features",
                    "best_number_of_features",
                    "best_f1_score",
                    "best_probability_mae",
                    "best_threshold"
                ]
            ].to_string(index=False)
        )

        return self.xgb_rfe_fold_results

   

    def xgboost_trial(self):
        """Will execute the class and train, test and if specified fine-tune the XGBoost

        Returns
        -------
        tuple[dict, dict,pd.DataFrame]
           Errormetrics, predicitions, errors for the best model
        """
        
        sampler = TPESampler()
        warmup_steps = 7
        N_ITER = self.trails or 10000 # Max number 10000
        name = 'regression'
        direction = 'maximize'
        

        if self.optuna: 
            # Hyper-parameter-fine-tuning
            optuna_results = self.optimizing_with_optuna(N_ITER,sampler, warmup_steps=warmup_steps,study_name=name, direction=direction)
            self.dump_files([optuna_results],self.path)
            optuna_results['best_params']["eval_metric"] = ['logloss', 'auc', 'error']
            optuna_results['best_params']['device'] = 'cuda'
            self.parameters = optuna_results['best_params']

            # Training
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data, self.parameters, self.path, top_k=30)
            
            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)
        else: 
           
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data,self.parameters, self.path, top_k=30)
        
            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)    
        return scores_table,preds,score_pd
      
    def run_anova_feature_selection(
        self,
        model_type: str = "SVC",
        n_splits: int = 5
    ):
        """
        Determine the optimal number of features using ANOVA feature selection.

        The method performs K-fold cross-validation on `self.data`. Within
        each fold, an ANOVA F-score is calculated independently for every
        feature using only the training data. Features are ranked from the
        highest to the lowest F-score.

        Separate classification models are subsequently fitted using the
        first 1, 2, 3, up to all ranked features. The feature subset with
        the highest F1 score on the test data is selected as the best subset
        for the respective fold.

        If multiple feature subsets produce the same F1 score, the subset
        containing fewer features is preferred.

        Results are stored in the following instance attributes:

        `self.anova_fold_results`
            DataFrame containing the best feature subset and performance
            metrics for every fold.

        `self.anova_best_features`
            Dictionary mapping each fold to its best feature subset.

        `self.anova_feature_rankings`
            Dictionary mapping each fold to its complete ANOVA feature
            ranking.

        `self.anova_score_tables`
            Dictionary containing the ANOVA F-score and p-value of every
            feature in each fold.

        `self.anova_all_results`
            Dictionary containing the evaluation result for every tested
            feature count in each fold.

        Parameters
        ----------
        model_type : {"SVC", "NuSVC", "LinearSVC", "XGBClassifier"}, default="SVC"
            Classification model used to evaluate the ANOVA-ranked feature
            subsets.
        n_splits : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        pd.DataFrame
            Fold-level results containing the optimal number of features,
            selected feature names, best F1 score, and complete feature
            ranking.

        Raises
        ------
        AttributeError
            If a required instance attribute is missing.
        ValueError
            If the number of folds, model type, labels, or feature matrix is
            invalid.
        RuntimeError
            If no feature subset can be evaluated successfully within a fold.

        Notes
        -----
        ANOVA feature selection is a univariate filter method. Each feature
        is evaluated independently, and interactions between features are
        not considered.

        The optimal feature count is selected using the test data of the
        respective fold. Therefore, the maximum F1 score should not be
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

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        if n_splits > len(self.data):
            raise ValueError(
                "n_splits cannot exceed the number of elements "
                "in self.data."
            )

        supported_models = {
            "SVC",
            "NuSVC",
            "LinearSVC",
            "XGBClassifier"
        }

        if model_type not in supported_models:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported values are: {sorted(supported_models)}."
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

        # Remove custom model identifiers that are not valid estimator
        # parameters.
        model_parameters.pop("svm", None)
        model_parameters.pop("model", None)

       
        model_parameters.pop("early_stopping_rounds", None)

      
        if model_type == "NuSVC":
            model_parameters.pop("random_state", None)

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

            print(
                f"Number of available features: {X_train.shape[1]}"
            )

            
            print("Calculating ANOVA F-scores...")

            f_scores, p_values = f_classif(
                X_train,
                y_train
            )

            anova_score_table = pd.DataFrame({
                "feature": X_train.columns,
                "f_score": f_scores,
                "p_value": p_values,
                "original_position": np.arange(X_train.shape[1])
            })

            
            anova_score_table["f_score"] = (
                anova_score_table["f_score"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-np.inf)
            )

            anova_score_table["p_value"] = (
                anova_score_table["p_value"]
                .replace([np.inf, -np.inf], np.nan)
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

            feature_ranking = anova_score_table[
                "feature"
            ].tolist()

            self.anova_feature_rankings[fold] = (
                feature_ranking.copy()
            )

            self.anova_score_tables[fold] = (
                anova_score_table.copy()
            )

            print("\nANOVA feature ranking:")

            for ranking_position, score_row in (
                anova_score_table.iterrows()
            ):
                print(
                    f"{ranking_position + 1:3d}. "
                    f"{score_row['feature']} | "
                    f"F-score: {score_row['f_score']:.4f} | "
                    f"p-value: {score_row['p_value']:.6f}"
                )


            feature_counts = []
            f1_values = []
            selected_feature_sets = []

            print("\nStarting feature-count evaluation...")

            for number_of_features in range(
                1,
                len(feature_ranking) + 1
            ):
                selected_features = feature_ranking[
                    :number_of_features
                ]

                
                classifier = xgb.XGBClassifier(
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

                    current_f1 = f1_score(
                        y_test,
                        predictions,
                        zero_division=0
                    )

                    feature_counts.append(
                        number_of_features
                    )

                    f1_values.append(
                        float(current_f1)
                    )

                    selected_feature_sets.append(
                        selected_features.copy()
                    )

                    print(
                        f"Number of features: "
                        f"{number_of_features:3d} | "
                        f"F1: {current_f1:.4f}"
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

           
            ordered_result_indices = sorted(
                range(len(f1_values)),
                key=lambda index: (
                    -f1_values[index],
                    feature_counts[index]
                )
            )

            best_result_index = ordered_result_indices[0]

            best_number_of_features = feature_counts[
                best_result_index
            ]

            best_f1_score = f1_values[
                best_result_index
            ]

            best_features = selected_feature_sets[
                best_result_index
            ]

            self.anova_best_features[fold] = (
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

            self.anova_all_results[fold] = (
                fold_evaluation_results
            )

            fold_result = {
                "fold": fold,
                "participant": participant,
                "model_type": model_type,
                "total_number_of_features": X_train.shape[1],
                "best_number_of_features": best_number_of_features,
                "best_features": best_features,
                "best_f1_score": best_f1_score,
                "feature_ranking": feature_ranking
            }

            all_fold_results.append(fold_result)

            print("\n" + "-" * 70)
            print(f"Best result for fold {fold}")
            print("-" * 70)

            print(
                f"Best number of features: "
                f"{best_number_of_features}"
            )

            print(f"Best F1 score: {best_f1_score:.4f}")

            print("Best features:")

            for feature_name in best_features:
                print(f"  - {feature_name}")

            self.dump_files_for_shap(
                [
                    fold_result,
                    fold_evaluation_results,
                    anova_score_table
                ],
                path=self.path,
                suffix=f"Fold_{fold}_{model_type}_ANOVA"
            )

        
        self.anova_fold_results = pd.DataFrame(
            all_fold_results
        )

        self.dump_files_for_shap(
            [
                self.anova_fold_results,
                self.anova_best_features,
                self.anova_feature_rankings,
                self.anova_score_tables,
                self.anova_all_results
            ],
            path=self.path,
            suffix=f"all_folds_{model_type}_ANOVA"
        )

        print("\n" + "=" * 70)
        print("Best ANOVA feature count and feature subset per fold")
        print("=" * 70)

        print(
            self.anova_fold_results[
                [
                    "fold",
                    "participant",
                    "model_type",
                    "total_number_of_features",
                    "best_number_of_features",
                    "best_f1_score"
                ]
            ].to_string(index=False)
        )

        return self.anova_fold_results



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
parser.add_argument('--final_selected_features', default=False, type=bool, help='Run the final models')   
parser.add_argument('--kfold', default=False, type=bool, help='Runs the model with kfold using by default 5 folds unless specified with --folds')
parser.add_argument('--folds', default=5, type=int, help='Number of folds which will be used for kfold.')
parser.add_argument('--annova', default=False, type=bool)
args = parser.parse_args()
folds = args.folds or 5
features = []



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = r'' # 

trial_name = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S") # for the name
if args.kfold: trial_name=trial_name+'_kfold'
# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print(decode_dict) # just that you know how the data was encoded

if args.finetune:
    n_iterations = args.trials or 1000    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning' # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, trials= n_iterations, decode_dict=decode_dict, kfold=False)
    scores_table,preds,score_pd = xg_model.xgboost_trial()
if args.percent < 100:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_percent_'+str(args.percent) # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=False,decode_dict=decode_dict)
    scores_table,preds,score_pd = xg_model.xgboost_trial()

elif args.rfe:
    trial_name= trial_name+'_rfeCL' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, decode_dict=decode_dict, kfold=False)
    features, features3 = xgboost_model.run_rfe_number()
elif args.initial:
    trial_name= trial_name+'_init' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=True, kfold=args.kfold)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
elif args.final and  not args.final_selected_features :
    print('running final version of the model:')
    trial_name= trial_name+'_final' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=False, kfold=args.kfold, number_fold=args.folds)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()
elif args.final_selected_features:
    trial_name= trial_name+'_features' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, kfold=args.kfold,number_fold=args.folds, features=features)
    scores_table,preds,score_pd = xgboost_model.xgboost_trial()

elif args.annova:
   
    trial_name= trial_name+'_with_annova' # for the name
    xgboost_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, decode_dict=decode_dict)
    features = xgboost_model.run_anova_feature_selection(
        model_type='XGBClassifier'
    )         
else: 
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_initial' # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xg_model.xgboost_trial()
    
