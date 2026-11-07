# Imports 
import copy
import datetime
import gc
import platform
import xgboost
import optuna
import numpy as np
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
from sklearn.metrics import mean_absolute_error, precision_recall_curve, zero_one_loss, accuracy_score, f1_score, precision_score, recall_score
import math
import inspect
import math
import inspect
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np
import optuna
from optuna.samplers import TPESampler
import utils


class xgboost:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), name:str='',name_label:str='label', num_folds:int=5, trial_name: str='regression_cognitve_load', optuna:bool=True, decode_dict:dict=None, trials:int=1000, init:bool=False) -> None:
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
        
        
        
        # General parameters
        
        self.parameters = [{
                   "objective": 'binary:logistic',
                    'class_weight':'balanced',
                    "eval_metric": ['logloss', 'auc', 'error'],
                    'device':'cuda'
             } ]
       
        if init: self.parameters = [{
                   "objective": 'binary:logistic',
                    'class_weight':'balanced',
                    "eval_metric": ['logloss', 'auc', 'error'],
                    'device':'cuda'}]
        self.early_stopping = {
            'rounds': 30,
            'min_delta':7e-4,
            'save_best':False,
            'maximize':False,
            'data_name':'validation_1',
            'metric_name':'logloss'
        }
            
        self.optuna =optuna
        self.num_folds = num_folds
        self.name = name
        self.name_label = name_label
        self.decode_dict = decode_dict
        self.trials = trials or 1000
        self.data = data
        path = path +'/xg_boost_'+trial_name + '_cat_personalized'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
        self.path_save_preds =r'/path/to/folder'

   
    
   

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
        fi.sort_index(axis=1, inplace=True) # Sorting the columns based on their number
        score_pd = pd.concat([score_pd, fi], ignore_index=True)  # concat the frames for the runs
        # Only saved if the save is set to true
        if not top_k: top_k = 30; 
        if not name: name='Feature Importance'; 
        if save:
            scores_table.to_csv(path+'/'+name+'_XG'+'.csv')
            temp.nlargest(n = top_k, columns="score").plot(title='Feature Importance',kind='barh', figsize = (20,10), xlabel='Feature Score', ylabel='Features', color ='#fe83cc') ## plot top 40 features
            plt.savefig(path+'/'+ name+'_XG'+'.png',bbox_inches = 'tight')
        return scores_table, score_pd
  


    def train(self,num_folds:int,parameters:list[dict]|dict, path:str,top_k:int=30):
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

        predictions = {}
        scores_participant= {}
        score_pd_part = {}
        best_models ={}
        

        
        c = 0
        skip=False
        for participant in self.data:
            score_pd = pd.DataFrame()
            splitter = KFold(n_splits=num_folds)
            best_model = {}
            pred = {}
            scores = []
            id = utils.decode_string_columns(pd.DataFrame(participant['id'], columns=['id']), decode_dict=self.decode_dict)
            name = id.iloc[0].item()
            #participant = participant.sample(frac=1)
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
                    skip=True; 
                    continue
                print('------------------------------- ✧ ✶𐙚⋆°｡⋆♡ Fold No.: {} ♡⋆｡°⋆𐙚✶ ✧ ------------------------------'.format(str(fold)))
                pos = (label_train_cl == 1).sum()
                neg = (label_train_cl == 0).sum()
                spw = neg / max(1, pos)
                early_stopping = xgb.callback.EarlyStopping(**self.early_stopping)
                if (len(parameters)>1 and isinstance(parameters, list)):
                    param = copy.deepcopy(parameters[c])
                    
                    param.update({
                  "objective": 'binary:logistic',
                           'class_weight':'balanced',
                            "eval_metric": ['logloss', 'auc', 'error'],
                            "scale_pos_weight":spw,
                            "device":'cuda'})
                    try:
                        md = copy.deepcopy(param['min_delta'])
                        r = copy.deepcopy(param['rounds'])
                        del param['min_delta'], param['rounds']
                        early_stopping = {
                            'rounds': r,
                            'min_delta':md,
                            'save_best':False,
                            'maximize':False,
                            'data_name':'validation_1',
                            'metric_name':'error'
                        }
                        early_stopping = xgb.callback.EarlyStopping(**early_stopping)
                        xg = xgb.XGBClassifier(**param,callbacks = [early_stopping])
                    except KeyError:
                             xg = xgb.XGBClassifier(**param)

                elif (len(parameters)==1 and isinstance(parameters, list)):
                    param = parameters[0]
                     
                    xg = xgb.XGBClassifier(**param)
            
           
                else:         
                    xg = xgb.XGBClassifier(**parameters)
           
                print(np.unique(label_train))
                print(np.unique(label_test))
                xg.fit(train_set,label_train_cl,
                eval_set=[(train_set, label_train_cl),(test_set, label_test_cl)],
                    verbose=1)
                compare = pd.DataFrame()
                proba = xg.predict_proba(test_set)
                proba1 = proba[:, 1]  # P(y=1)
                mae_ord = mean_absolute_error(label_test_cl, proba1)
                print("Ordinal-MAE (auf P(y=1)):", mae_ord)

                mae_ord = mean_absolute_error(label_test_cl, proba1)
                comp = {}
               
                precision, recall, thresholds = precision_recall_curve(label_test_cl, proba1)
                
                f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
                best_idx = int(np.argmax(f1_scores))
                best_thresh = float(thresholds[best_idx])
                best_f1 = float(f1_scores[best_idx])
                comp['f1'] =best_f1
                comp['mae'] = mae_ord
                comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
                
                print('F1S: ', best_f1)
                results = xg.evals_result()
                
                compare['auc_train'] = results['validation_0']['auc']
                compare['logloss_train'] = results['validation_0']['logloss']
                compare['mae_train'] = results['validation_0']['error']
                compare['auc_test'] = results['validation_1']['auc']
                compare['logloss_test'] = results['validation_1']['logloss']
                compare['error_test'] = results['validation_1']['error']
            
            
            
                name_fold = 'Fold_'+str(fold)
                pred[name_fold]=compare
                pred[name_fold+'_pred'] = proba1
                pred[name_fold+'pred_org'] = proba
                pred[name_fold+'_truth'] = label_test_cl
                pred[name_fold + "_truth_org"] = label_test
                # Speichern des Models, nutzlos ich weiß aber nun gut
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
                
                try:
                    scores_table, score_pd = self.importance_feature(score_pd,xg,path,name= 'Features Importance '+name,top_k=top_k)
                    scores.append(scores_table)
                except ValueError: pass    
            predictions[str(name)]=pred    
            best_models[name] =best_model
            
            try:
                self.final_plot(score_pd, self.path, top_k=30, name='_'+name)    
                best_scores, _ = self.importance_feature(pd.DataFrame(),best_model['model'], path, 'best_model'+name, save=True)
            except ValueError:pass
            score_pd.fillna(0, inplace=True)
            score_pd_part[name] = score_pd
            scores_participant[name] = scores
            c = c+1
            if skip: c=c-1
        self.dump_files_for_shap([predictions], self.path_save_preds, suffix=self.trial_name+'xgb_cl_personalized')

        
        return scores_participant, predictions, best_model, best_scores, score_pd_part

    
           
        


    def visualize_results(self, ground_truth: np.array, pred:np.array, nombre:str,compare:pd.DataFrame, fold:int, participant:str)-> None:
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
        plt.xlabel('True Category')
        plt.ylabel('Predicted Category')
        plt.title('Predicted software development category vs. actual category  '+ str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#e84a7f', label='True Category', marker='*')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#E98DAD', label='Predicted Category', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Category')
        plt.title('Predicted software development category vs. actual category  '+ str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_line_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)
        fig323cc = plt.figure(figsize=(40,7))
     
        sns.lineplot(data=temp['True label'], color='#E84a7f', label='True True Category',marker='o' , linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Category',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Category')
        plt.title('Cognitve load estimated from psycho-physiological data vs. subjective perception of participant '+str(participant)+ ' from fold '+str(fold), wrap=True)
     
        name_scatter = 'cat_true_pred_participant_with_line_'+str(participant)+'_Fold_'+str(fold)+nombre+'.pdf'
        plt.savefig(self.path+'/'+name_scatter)
        plt.close(fig323cc)


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
        plt.xlabel('True Category')
        plt.ylabel('Predicted Category')
        plt.title('Predicted software development category vs. actual category  '+ str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participantf_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig22)

        fig323c = plt.figure(figsize=(40,7))
        sns.scatterplot(data=temp, x=temp.index, y='True label', color='#377eb8', label='True Category', marker='o')
        sns.scatterplot(data=temp, x=temp.index, y='Predicted label', color='#4daf4a', label='Predicted Category', marker='o')
        plt.xlabel('Time windows')
        plt.ylabel('Category')
        plt.title('Predicted software development category vs. actual category  '+ str(participant) + ' from fold ' + str(fold), wrap=True)
        name_scatter = 'cat_true_pred_participant_linef_' + str(participant) + '_Fold_' + str(fold) + nombre + '.pdf'
        plt.savefig(self.path + '/' + name_scatter)
        plt.close(fig323c)

        fig323cc = plt.figure(figsize=(10,7))
        sns.lineplot(data=temp['True label'], color='#377eb8', label='True Category', marker='o',linestyle = 'dotted')
        sns.lineplot(data=temp['Predicted label'], color='#E98DAD', label='Predicted Category',marker='o',  linestyle = '')
         
        plt.xlabel('Time windows')
        plt.ylabel('True Category')
        plt.title('Predicted software development category vs. actual category  '+ str(participant) + ' from fold ' + str(fold), wrap=True)
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

            

    # Plot the final top 40 features 
    def final_plot(self,score_pd, path, top_k :Optional[int]=None, name:Optional[str]=''):
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
        name = 'Best_Mean_Top_40_Features'+name
        score_pd.fillna(0, inplace=True)
        if not top_k: top_k = 30;
        
        for i in score_pd.columns:
            average_feature[i] = [score_pd[i].mean()]   
        average_feature = average_feature.transpose()
        average_feature.columns = ['score']
        average_feature.nlargest(n = top_k, columns='score').plot(title='Feature Importance',kind='barh', figsize = (20,10), ylabel='Features', xlabel='Features Importance', color = '#fe83cc') ## plot top 40 features
        plt.savefig(path+'/'+ name+'.png',bbox_inches = 'tight')
        plt.close()
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

        # finetuned parameters
        label_train_cl = utils.map_labels(labels=self.label_train)
        label_test_cl = utils.map_labels(self.label_test)


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
                           warmup_steps:int, study_name:str='regression',
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

        participants_results = {}
        results_as_list = []
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
            
            ### DROP THE COLUMNS WHICH ARE NOT NESSACARY 1!!1!!!!


            self.train_set = X_train
            self.label_train = label_train
            self.test_set = X_test
            self.label_test = label_test
            results_tpe={}
            # Create the study
            study = optuna.create_study(
                #pruner = optuna.pruners.MedianPruner(n_warmup_steps = warmup_steps),
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
            date = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")

            with open('best_parameters_kfold_xgb' +participants_name+'_'+date+'.txt', 'w') as f:
                print(study.best_params, file=f)
            participants_results[participants_name] = results_tpe
            results_as_list.append(study.best_params)
        return participants_results, results_as_list
        
   
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
    
    def run_rfe(self):
            counter = 0
            for participant in self.data:
                train_set, test_set = sklearn.model_selection.train_test_split(participant, shuffle=True)
                X_test, X_train, label_test, label_train = utils.personal_datasets_ml(train_data=train_set, test_data=test_set)
                
                params = copy.deepcopy(self.parameters[counter])
                id = utils.decode_string_columns(pd.DataFrame(participant['id'], columns=['id']), decode_dict=self.decode_dict)
                participants_name = id.iloc[0].item()
                print('participant ', participants_name)
                try: 
                    # remove the early stopping parameter
                    del params[counter]['early_stopping']
                except KeyError: pass    
                xg = xgb.XGBClassifier(**params)
                model = RFE(estimator=xg)


                selector = model.fit(X_train, label_train)
                columns = X_train.columns.tolist()
                columns_bool = selector.support_
                from itertools import compress
                col_list = list(compress(columns, columns_bool))  
                #     col_list = data_kaggle_feature1.columns.to_list()[:-1]
                print(col_list)
                ranking_list = list(compress(columns, selector.ranking_))
                print(ranking_list)
                counter = counter+1
            return col_list, ranking_list

    def xgboost_trial(self):

        """Will execute the class and train, test and if specified fine-tune the XGBoost

        Returns
        -------
        tuple[dict, dict,pd.DataFrame]
           Errormetrics, predicitions, errors for the best model
        """
        # hyperparameter fine-tuning
        sampler = TPESampler()
        warmup_steps = 7
        name = 'cl_classification'
        direction = 'maximize'
        device = 'cuda'

        if self.optuna: 
            optuna_results, optuna_list = self.optimizing_with_optuna(self.trials,sampler, warmup_steps=warmup_steps,study_name=name, direction=direction)
            self.dump_files([optuna_results, optuna_list],self.path)
            scores_participant, predictions, best_model, best_scores, score_pd_part= self.train(self.num_folds,optuna_list, self.path,top_k=30)
            return optuna_results

        else: 
           

            scores_participant, predictions, best_model, best_scores, score_pd_part= self.train(self.num_folds,self.parameters, self.path,top_k=30)
            self.dump_files([scores_participant, predictions, best_model, best_scores, score_pd_part], self.path)    
            return scores_participant, predictions, score_pd_part
    






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



print('Decode Dict: \n',decode_dict)


if args.finetune:
# train the  xgb model
    n_iterations = args.trials or 1000    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning_cat_perso' # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=True, trials= n_iterations, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost.xgboost_trial()
if args.percent < 100:
    data = utils.reduce_dataframes_size(data, percentage=args.percent)
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_percent__cat_perso'+str(args.percent) # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=False,decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost.xgboost_trial()
elif args.rfe:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning_cat_perso' # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=False, trials= n_iterations, decode_dict=decode_dict)
    features, features_list = xgboost.run_rfe()
elif args.initial:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_init_cat_perso' # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict, init=True)
    scores_table,preds,score_pd = xgboost.xgboost_trial()
elif args.final:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_final_run_endgültig_cat_xgb_cl_personalized' # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost.xgboost_trial()
             
else: 
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_normal_cat_perso' # for the name
    xgboost = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xgboost.xgboost_trial()
    