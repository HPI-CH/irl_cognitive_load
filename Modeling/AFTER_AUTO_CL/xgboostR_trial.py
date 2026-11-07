# Imports 
import copy
import gc
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
import platform

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
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from optuna.samplers import TPESampler
import utils


class xgboost:
    def __init__(self,label_path:str ,path_data:str, path:str=os.path.abspath(os.getcwd()), name:str='',name_label:str='label', num_folds:int=4, trial_name: str='regression_cognitve_load', optuna:bool=True, init:bool=False, trials:int=1000) -> None:
        """Initialization of the xgboost class. This class contains the finetuning, training, and testing of/with XGBoost. The model will be validated with the LOOCV (currently a 4-fold CV). \
            The class also contains multiple plot functions to visualize the predictions and the importance of features with the built feature importance module.


        Parameters
        ----------
        path_data : str
            where to load the data from
        path_label : str
            where the label from the autoencoder is saved
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
        
        
        
        if init: 
            self.parameters = {
                "objective": 'reg:squarederror',
                'device':'cuda', 
                "eval_metric":['mape','rmse', 'mae']
            }

      
        self.early_stopping = {
            'rounds': 30,
            'min_delta':7e-4,
            'save_best':False,
            'maximize':False,
            'data_name':'validation_1',
            'metric_name':'logloss'
        }
        self.parameters = {"objective": 'reg:squarederror',
                            "eval_metric": ['mape','rmse', 'mae'],
                          
                            'device' :'cuda'}
        
        
        if 'tcn' in label_path: 
            self.tcn=True; trial_name = trial_name+'_tcn'
          
        else: self.tcn=False; trial_name = trial_name+'_lstm'
            
        self.optuna =optuna
        self.num_folds = num_folds
        self.name = name
        self.name_label = name_label
        self.path_label = label_path
        self.path_data = path_data
        path = path +'/xg_boost_'+trial_name+'_LOOCV_auto'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.trial_name = trial_name
        self.path = path
     

        self.path_save_preds =r'/path/to/folder'
   
    

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
    
  
    
  
    def train(self,data,num_folds, path,top_k=30):
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
        num_folds = len(data)
        kfold = KFold(n_splits=num_folds)
        predictions = {}
        scores= []
        best_model = {}
        score_pd = pd.DataFrame()
        labeling_data = []
        testing_data = []
        modeling = []
        

        for fold,  (train_idx, test_idx) in enumerate(kfold.split(data)):
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
            label_train_cl = utils.map_labels(labels=label_train)
            label_test_cl = utils.map_labels(label_test)
            pos = (label_train_cl == 1).sum()
            neg = (label_train_cl == 0).sum()
            spw = neg / max(1, pos)
            self.parameters["scale_pos_weight"]=spw
            gc.collect()
            print('Fold {} ... ✧ ✶𐙚⋆°｡⋆♡'.format(fold))
            early_stopping = xgb.callback.EarlyStopping(**self.early_stopping)
            labeling_data.append(label_train_cl)
            testing_data.append(label_test_cl)
            parameters= copy.deepcopy(self.parameters)
            pos = (label_train_cl == 1).sum()
            neg = (label_train_cl == 0).sum()
            spw = neg / max(1, pos)
            parameters["scale_pos_weight"]=spw
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
            # thresholds ist um 1 kürzer als precision/recall
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
            # Anpassen !!!!! mit validation und 0 und 1
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
                #best_model['logloss'] = results['validation_1']['logloss']
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
        # y_proba = model.predict(self.test_set)
        
        # f1 =f1_score(self.label_test, y_proba, average='binary')
        
        proba1 = model.predict_proba(self.test_set)[:, 1]

        # 1) Ordinaler MAE (nur zum Vergleich, nicht Return)
        mae_ord = mean_absolute_error(label_test_cl, proba1)

        # 2) Beste Schwelle für F1 via PR-Kurve (feiner als grobes Raster)
        precision, recall, thresholds = precision_recall_curve(label_test_cl, proba1)
        # thresholds ist um 1 kürzer als precision/recall
        f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_idx = int(np.argmax(f1_scores))
        best_thresh = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

        return best_f1

    def make_train_test_sets(self, data:dict|list, test_size:float=0.3, normalizing:bool=True, label:Optional[pd.DataFrame]=pd.DataFrame()):
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
        

        data = pd.concat(data, axis='index')
        data = data.dropna(how = 'any')
        label = data['label']
        data=data.drop('label', axis='columns')
        X_train, X_test,label_train, label_test  = train_test_split(data, label, test_size=test_size, shuffle=False)


        return X_train,label_train, X_test, label_test  






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
        X_train,label_train, X_test, label_test = self.make_train_test_sets(self.data, test_size=0.4)


        self.train_set = X_train
        self.label_train = label_train
        self.test_set = X_test
        self.label_test = label_test
        results_tpe={}
        # Create the study
        study = optuna.create_study(
            pruner=optuna.pruners.HyperbandPruner(
            min_resource=20, reduction_factor=6
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
    
    
    def reshape_for_svm(self, autoencoder_output: np.ndarray) -> np.ndarray:
        """
        Reshapes the output from the Autoencoder to the correct form for SVM, where 128 is the sequence length and 56* are the features.

        Parameters
        ----------
        autoencoder_output : np.ndarray
            The output from the Autoencoder with shape (63, 56, 128).

        Returns
        -------
        np.ndarray
            The reshaped array with shape (63*128, 56).
        
        Example
        -------
        >>> autoencoder_output = np.random.rand(63, 56, 128)
        >>> reshaped_output = reshape_for_svm(autoencoder_output)
        >>> print(reshaped_output.shape)
        (8064, 56)
        """
        # Check the shape of the input array
        assert len(autoencoder_output.shape) == 3, "Input array must have three dimensions"
        
        # Reshape the array to the specified shape
        reshaped_output = autoencoder_output.transpose(0, 2, 1).reshape(-1, autoencoder_output.shape[1])
        return reshaped_output

    
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
           

    def xgboost_trial(self):
        """Will execute the class and train, test and if specified fine-tune the XGBoost

        Returns
        -------
        tuple[dict, dict,pd.DataFrame]
           Errormetrics, predicitions, errors for the best model
        """
        sampler = TPESampler()
        warmup_steps = 7
        N_ITER = 3000 # Max number 10000
        test_size = 0.2
        name = 'classification'
        direction = 'maximize'
        device = 'cuda'

        features = pickle.load(open(self.path_data, 'rb'))
        target = pickle.load(open(self.path_label, 'rb'))
        print(features)
        feature_frames_machine_learning=[]

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
        print(self.data)
   



        if self.optuna: 
            optuna_results = self.optimizing_with_optuna(N_ITER,sampler, warmup_steps=warmup_steps,study_name=name, direction=direction)
            self.dump_files([optuna_results],self.path)
            optuna_results['best_params']["eval_metric"] = ['logloss','error', 'auc']

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
                            'metric_name':'logloss'
                        }
            except KeyError: pass 
            self.parameters = param
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data, self.num_folds,self.parameters, self.path,self.name_label, top_k=30)
            
        
            mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)
        else: 
            

            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data,5,self.path_save_preds, top_k=30)
            
        
            # mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)    
        return scores_table,preds,score_pd
    


import argparse
import datetime
import torch
parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--final', default=False, type=bool, help='Run the final models')   
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
parser.add_argument('--initial', default=False, type=bool, help='Will let the model run with its default parameter')
parser.add_argument('--tcn', default=False, type=bool, help='runs it with tcn data ')
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features=r'/path/to/features'
target= r'/path/to/target'
path = r'/path/to/data'



if args.finetune:

    n_iterations = args.trials or 1000    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'finetuninglstm' # for the name
    xgb_model = xgboost(label_path=target, path_data=features,path=path, name='', trial_name=trial_name,optuna=True, init=False, trials=n_iterations)
    scores_table,preds,score_pd = xgb_model.xgboost_trial()
    
if args.percent < 100:
    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_percent_'+str(args.percent) # for the name
   

elif args.initial:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_init' # for the name
    xgb_model = xgboost(label_path=target, path_data=features,path=path, name='',trial_name=trial_name,optuna=False, init=True)
    scores_table,preds,score_pd = xgb_model.xgboost_trial()

   
   
elif args.final:
    print('running final version of the model:')
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_final_run_xgb_loocv_auto_bisherigeParamstcn' # for the name
    xgb_model = xgboost(label_path=target, path_data=features,path=path, name='', trial_name=trial_name,optuna=False, init=False)
    scores_table,preds,score_pd = xgb_model.xgboost_trial()
       
         
else: 

    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_normal' # for the name
    xgb_model = xgboost(label_path=target, path_data=features,path=path, name='', trial_name=trial_name,optuna=False)
    scores_table,preds,score_pd = xgb_model.xgboost_trial()



