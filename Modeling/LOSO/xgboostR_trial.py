# Imports 
import copy
import gc
import platform
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
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error, r2_score
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



class xgboost:
    def __init__(self, data: dict, path:str=os.path.abspath(os.getcwd()), trial_name: str='regression_cognitve_load', optuna:bool=True, trials:int=1000, decode_dict=None) -> None:
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
    
        
        self.parameters = {"objective": 'reg:squarederror',
                            "eval_metric": ['mape','rmse', 'mae'],
                            'device' :'cuda'}
        
        
       
            
        self.early_stopping = {
            'rounds': 30,
            'min_delta':7e-4,
            'save_best':False,
            'maximize':False,
            'data_name':'validation_1',
            'metric_name':'mae'
        }
            
        self.optuna =optuna
        self.data = data
        self.trails = trials
        self.decode_dict = decode_dict
        path = path +'/xg_boost_'+trial_name+'_LOOCV'
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
        results = model.evals_result()['validation_1']['mae']
        fi['MAE']=mean(results)
        
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
        kfold = KFold(n_splits=len(data))
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
            id = utils.decode_string_columns(pd.DataFrame(test_set['id'], columns=['id']), decode_dict=self.decode_dict)
            participant = id.iloc[0].item()
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡'.format(fold,participant))
            
           

            xg = xgb.XGBRegressor(**parameters, callbacks = [early_stopping])
           
            
            xg.fit(train_set,label_train,
                eval_set=[(train_set, label_train),(test_set, label_test)],
                    verbose=1)
            compare = pd.DataFrame()
            pre=xg.predict(test_set)
            modeling.append(xg)
            comp = {}
            
            comp['rmse'] = math.sqrt(mean_squared_error(label_test, pre))
            comp['mse'] = mean_squared_error(label_test, pre)
            comp['mape'] = mean_absolute_percentage_error(label_test, pre)
            comp['mae'] = mean_absolute_percentage_error(label_test, pre)
            comp['r2'] = r2_score(label_test, pre)
            comp = pd.DataFrame(comp, index=['Fold '+str(fold)])
            mse = mean_squared_error(label_test, pre)
            mae = mean_absolute_error(label_test, pre)
            print('MSE: ', mse)
            print('RMSE: ', math.sqrt(mse))
            print('MAE: ', mae)
            results = xg.evals_result()
            # Anpassen !!!!! mit validation und 0 und 1
            compare['rmse_train'] = results['validation_0']['rmse']
            compare['mape_train'] = results['validation_0']['mape']
            compare['mae_train'] = results['validation_0']['mae']
            compare['rmse_test'] = results['validation_1']['rmse']
            compare['mape_test'] = results['validation_1']['mape']
            compare['mae_test'] = results['validation_1']['mae']
         
            compare['r2'] = r2_score(label_test, pre)
          
            name = 'Fold_'+str(fold)
            predictions[name]=compare
            predictions[name+'_pred'] = pre
            predictions[name+'_truth'] = label_test
            # Speichern des Models, nutzlos ich weiß aber nun gut
            print('saving stuff')
            if fold ==0:
                best_model['min_mean_mae'] = mean(results['validation_1']['mae'])
                best_model['min_mean_rmse'] = mean(results['validation_1']['rmse'])
                best_model['min_mean_mape'] = statistics.mean(results['validation_1']['mape'])
                best_model['rmse']=results['validation_1']['rmse']
                best_model['mape']= results['validation_1']['mape']
                best_model['mae'] = results['validation_1']['mae']
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = xg

            elif statistics.mean(results['validation_1']['mae']) < best_model['min_mean_mae']:
                best_model['min_mean_rmse'] = statistics.mean(results['validation_1']['rmse'])
                best_model['min_mean_mae'] = mean(results['validation_1']['mae'])
                best_model['min_mean_mape'] = statistics.mean(results['validation_1']['mape'])
                best_model['rmse']=results['validation_1']['rmse']
                best_model['mape']= results['validation_1']['mape']
                best_model['mae'] = results['validation_1']['mae']
                best_model['train_idx'] = train_idx
                best_model['test_idx'] = test_idx
                best_model['model'] = xg
            
            self.visualize_results(ground_truth=label_test, pred=pre, nombre='_xgb', compare=comp, fold=fold, participant=participant)
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
                with open(f"{path}/{name[0]}{suffix}.pickle", 'wb') as handle:
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
            "objective": 'reg:absoluteerror',
            "eval_metric": 'mae',
            "seed": 77,
            "verbosity": 0,
        "device": 'cuda'
        ,
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
          
        
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-mae")
        model = xgb.XGBRegressor(**param,  callbacks = [pruning_callback])
        model.fit(self.train_set,self.label_train,
                eval_set=[(self.test_set, self.label_test)],
                    verbose=0)
        pred = model.predict(self.test_set)
        mae = mean_absolute_error(self.label_test, pred)
        r2 = r2_score(self.label_test, pred)
        mse =mean_squared_error(self.label_test, pred)
        rmse = math.sqrt(mse)
        mape =mean_absolute_percentage_error(self.label_test, pred)
        return r2


    def make_train_test_sets(self, data:dict, test_size:float=0.3, normalizing:bool=True, label:Optional[pd.DataFrame]=pd.DataFrame())-> tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]:
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
       
      
        if normalizing: 
            train_samples = []
            test_samples = []
            train2 = np.array([*data.values()]).tolist()
            for participant_data in data.values():
                for task_data in participant_data.values():
                    task_data = task_data.sample(frac=1, random_state=42)  # Shuffle the data but keep the index
                    split_index = int(len(task_data) * (1 - test_size))
                    train_samples.append(task_data.iloc[:split_index])
                    test_samples.append(task_data.iloc[split_index:])

            train_data = pd.concat(train_samples)
            test_data = pd.concat(test_samples)

            train_list = utils.generate_train_test_dict(train2=train2, x_split=train_data)
            test_list = utils.generate_train_test_dict(train2=train2, x_split=test_data)
            test_dicts_scaled, test_scaled, train_scaled, train_dicts_scaled = utils.normalize(train_list, test_list)

           

            test_set = pd.concat([pd.concat([*x.values()], axis='index') for x in test_dicts_scaled], axis='index').sample(frac=1, random_state=42)
          
            train_set = pd.concat([pd.concat([*x.values()], axis='index') for x in train_dicts_scaled], axis='index').sample(frac=1, random_state=42)  # added shuffling# added shuffling

            train_set.dropna(how='any', axis='index', inplace=True)
            test_set.dropna(how='any', axis='index', inplace=True)
            label_test = test_set['label']
            test_set = test_set.drop(['label'], axis='columns')
            label_train = train_set['label']
            train_set = train_set.drop(['label'], axis='columns')

            
        else: 
            all_data = pd.concat([x for xs in [[*x.values()] for x in list(data.values())] for x in xs])
            train_data, test_data = train_test_split(all_data, test_size=test_size)
            train_set.dropna(how='any', axis='index', inplace=True)
            test_set.dropna(how='any', axis='index', inplace=True)
            label_test = test_set['label']
            test_set = test_set.drop(['label'], axis='columns')
            label_train = train_set['label']
            train_set = train_set.drop(['label'], axis='columns')
            
        return train_set, label_train, test_set, label_test      


    def normalize(self,train_data:pd.DataFrame, test_data:pd.DataFrame, scale_method:Optional[str]='standard')-> tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]:
        """Calls scikit normalizer to scale the features

        Parameters
        ----------
        train_data : pd.DataFrame
            contains the training data
        test_data : pd.DataFrame
            contains the testing data
        scale_method : Optional[str], optional
            which Scikti scaler should be used: standard, minmax, by default 'standard'

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.DataFrame,pd.Series]
            scaled train data, train  label, scaled test data, test label
        """
        if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
        elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            scaler = sklearn.preprocessing.StandardScaler()
        
        label_train = train_data['label'].astype(int)
        label_test = test_data['label'].astype(int)
        train_data.drop('label', inplace=True, axis='columns')    
        test_data.drop('label', inplace=True, axis='columns')
        columns = train_data.columns    
        X_train = pd.DataFrame(scaler.fit_transform(train_data), columns=columns)
        X_test = pd.DataFrame(scaler.transform(test_data), columns=columns)
        X_train.interpolate('cubicspline', order=5, limit_direction='both', inplace = True)
        X_test.interpolate('cubicspline', order=5, limit_direction='both', inplace = True)
        return X_train, label_train, X_test, label_test







    def optimizing_with_optuna(self,n_iterations:int , sampler, 
                           warmup_steps:int, study_name:str='regression',
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
        test_size = 0.2
        name = 'regression'
        direction = 'maximize'
        device = 'cuda'

        if self.optuna: 
            optuna_results = self.optimizing_with_optuna(N_ITER,sampler, warmup_steps=warmup_steps,study_name=name, direction=direction)
            self.dump_files([optuna_results],self.path)
            optuna_results['best_params']["eval_metric"] = ['mape','rmse', 'mae']
            self.parameters = optuna_results['best_params']


            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data, self.parameters, self.path, top_k=30)
            
        
            mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)
        else: 
           
            scores_table, preds, best_model, best_scores, score_pd= self.train(self.data,self.parameters, self.path, top_k=30)
            
        
            mean_score, sum_score = self.calculate_mean_sum_scores(scores=score_pd)
            

            self.final_plot(score_pd, self.path, top_k=30)
            self.dump_files([scores_table, preds, best_model, best_scores, score_pd], self.path)    
        return scores_table,preds,score_pd
    



import argparse
import datetime
import torch
parser = argparse.ArgumentParser(description='XGBOOST Training and finetuning')
parser.add_argument('--percent', default=100, type=float, help='Amount data used for training and test, not implemented yet')    
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=5000, type=int, help='Number of trials the models should be finetuned')
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'/path/to/data'



# transform the data for the machine learning models
data, decode_dict = utils.preprare_data(path, 'ml_features')
print(decode_dict) # just that you know how the data was encoded

if args.finetune:
    n_iterations = args.trials or 1000    
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'during_tuning' # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=True, trials= n_iterations, decode_dict=decode_dict)
    scores_table,preds,score_pd = xg_model.xgboost_trial()
if args.percent < 100:
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_percent_'+str(args.percent) # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=False,decode_dict=decode_dict)
    scores_table,preds,score_pd = xg_model.xgboost_trial()

else: 
    trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'_initial' # for the name
    xg_model = xgboost(data,path=path, trial_name=trial_name,optuna=False, decode_dict=decode_dict)
    scores_table,preds,score_pd = xg_model.xgboost_trial()
    