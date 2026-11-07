# Imports
import copy
import datetime
import inspect
import pickle
import platform
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

import numpy as np
import utils

import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
import optuna
from typing import Optional
 
import lstmnet


import os
import pandas as pd


class MeanMetric:
    def __init__(self,device='cpu'):
          self.device = device
          self.reset()



    def update(self, value):
        self.sum += value.sum().item()
        self.count += value.numel()

    def compute(self):
        return self.sum / self.count if self.count != 0 else 0.0
    def reset(self):
        self.sum = 0.0
        self.count = 0





class lstm_model:



    def __init__(self, data:dict,parameters:dict=None, WINDOW_SIZE:Optional[float]=30*32,OVERLAP:Optional[float]=None,n_iterations:Optional[float]=500, study_name:Optional[str]="Autoencoder Optimization", direction:Optional[str]='minimize',decode_dict:dict=None, fold:int=0) -> None:
        """Initialize the automatic feature generation class, necessary for finetuning and running the autoencoder class, it will train and test the autoencoder and will generate the features

        Parameters
        ----------
        data : dict
            contains the data which is specific for the deep learning models e.g. only "raw" data
        name_label : str, optional
            name of the label column, by default 'label'
        WINDOW_SIZE : Optional[float], optional
            size of the window, by default 384
        OVERLAP : Optional[float], optional
            lenght of the overlap in the window, by default None
        n_iterations : Optional[float], optional
            Number of trials, Optuna should carry out, by default 1000
        study_name : Optional[str], optional
            Name of the fine-tuning run, by default "Autoencoder Optimization"
        direction : Optional[str], optional
            in which direction optubna should go, either elevating the metric or reducing the metric, by default 'minimize'
        INPUT_SIZE : Optional[int], optional
            Number of features, by default 92
        """
        
        self.decode_dict = decode_dict
        self.data = data
        self.WINDOW_SIZE = WINDOW_SIZE
        self.fold = fold
        if not OVERLAP:
            OVERLAP = self.WINDOW_SIZE//2
        self.OVERLAP = OVERLAP
        
        self.n_interations = n_iterations
        self.sampler = optuna.samplers.TPESampler()
        self.warm_upsteps = 0
        self.study_name = study_name
        self.direction = direction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parameters = parameters or {
        "attention": True,
        "sparse": True,
        "num_heads": 4,
        "attention_dropout": 0.2,
        "dropout": 0.1,
        "num_layer": 2,
        "bidirectional": True,
        "output": 1,
        "output_size": None

        }




        saving_path = r'/path/to/folder'

        trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'before_tuning'
        self.saving_path = saving_path+'/lstm_compl_'+trial_name
        if not os.path.isdir(self.saving_path):
            os.mkdir(self.saving_path)










    def train(self, model: lstmnet.LSTMNET, train_loder: torch.utils.data.DataLoader, optimizer: torch.optim,scheduler:torch.optim, epoch:str, criterion:torch.nn.MSELoss)->tuple[lstmnet.LSTMNET, list]:
        """Training of the autoencoder: Contains the training loop, and uses the MSE as loss function combined with a scheduler and Adam as the optimizer. 

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : lstmnet.LSTMNET
            TKAN-KAN-Attention-Autoencoder
        device : str
            wether to use cuda or cpu
        train_loder : torch.utils.data.DataLoader
            for loading the trainingsdata
        optimizer : torch.optim
            the used optimizer
        scheduler : torch.optim
            the used scheduler
        epoch : str
            current epoch
        criterion : torch.nn.MSELoss
            loss function: defaut is MSE
        num_classes : float
            number of classes. However, this autoencoder was made for regression, not used

        Returns
        -------
        LSTMAttentionAutoencoder.LAAE, list
            the trained model and a list containing the mean loss
            
        """
 
        model.train()

        #ce_mean = metric.MeanMetric()
        ce_mean = MeanMetric()
        ce_mean.reset()

        train_loss = []
        


        for batch_idx, (data, target) in enumerate(train_loder):
            data = data.to(self.device)
            target = target.to(self.device)
            print(target)
            
            y_pred = model(data)
            print(y_pred[0])
           
            del data
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(y_pred, target)
            
        
            
            del target
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         
            ce_mean(loss.cpu())
           
            if batch_idx % 100 == 0:
              
               
                tqdm.write('Epoch {} (train) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
           
        ce_mean(loss.cpu())
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        train_loss.append(ce_mean.compute())
        scheduler.step()
        return model, train_loss
        


        




    def test(self, model:  lstmnet.LSTMNET, test_loder: torch.utils.data.DataLoader,  epoch:str, criterion:torch.nn.CrossEntropyLoss)->tuple[lstmnet.LSTMNET, list]:
        """Testing of the autoencoder: Contains the testing loop, and uses the MSE as loss function.

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : lstmnet.LSTMNET
           Model
        device : str
            wether to use cuda or cpu
        test_loder : torch.utils.data.DataLoader
            loads the test data
        epoch : str
            current epoch
        criterion : torch.nn.CrossEntropyLoss
            loss function, only for showing the loss not for gradient
        num_classes : float
            unused

        Returns
        -------
        LSTMNET, list
            the model and a list containing the mean loss
        """
        model.eval()
       
      
        ce_mean = MeanMetric()
        ce_mean.reset()
      

        test_f1 = []
        test_acc = []
        test_loss = []
        


        for batch_idx, (data, target) in enumerate(test_loder):
            data = data.to(self.device)
            target = target.to(self.device)

            
            y_pred = model(data)

            del data
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(y_pred,target)
            
     
            
            del target
            gc.collect()
            torch.cuda.empty_cache()
           
            ce_mean(loss.cpu())
            

            if batch_idx % 100 == 0:
                
                tqdm.write('Epoch {} (test) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
          
            del loss
            gc.collect()
            torch.cuda.empty_cache()
            

        test_loss.append(ce_mean.compute())
        
        return model, test_loss
        

    
    def get_model_size(self, model):
        param_size = sum(param.numel() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def train_test_call(self, data:dict,random_seeds:float=77)->tuple[dict[list], list]:
        """Will train and test the autoencoder. It calls the functions `train`, `test` and will also generate the features.

        Parameters
        ----------
        data : dict
            the data from the participants sorted into a dict. The dict contains the participants as dicts, each of the participant has a dict with the tasks. For each task there is one dataframe
        parameters : dict
            parameters choosen for the autoencoder
        num_folds : float
            number of splits for the KFOLD
        name_label : str, optional
            name of the label in the dataframes, by default 'label'
        random_seeds : float, optional
            random seed, by default 77
        use_weighting : bool, optional
            if the loss should be weighted or not, this statement is not used anymore, by default True
        window_size : float, optional
            size of the time window in number of points, by default 100
        overlap : float, optional
            length of the overlap between the windows, by default 100//2
        learning_rate : float, optional
            learning rate, by default 0.001
        device : str, optional
            where to train the autoencoder can be 'cuda' or 'cpu, by default 'cpu'
        mum_classes : float, optional
            number of classes, not used anymore, by default 3

        Returns
        -------
        dict[list], list[torch.tensor]
            metrics to evaluate the autoencoders performance, list of predicted features
        """
        




        
        BATCH_SIZE = 32
        EPOCHS = 25
        lr =  0.029368467331057336
        best_model = None
        counter = 0
        kfold = KFold(n_splits=len(data))
        torch.manual_seed(random_seeds)
        metrics_dict = {}
        features_pred = []
        target_pred = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(self.data)):
            if not(int(fold) == 4): 
                 print('Skipped FOld: '+ str(fold))
                 continue
            torch.cuda.empty_cache()
            data2 = copy.deepcopy(data)
            
            train_data = np.array(data2, dtype=object)[train_idx].tolist()
            test_data = np.array(data2, dtype=object)[test_idx].tolist()
            
            test_set, train_set, label_test, label_train, auto_test, auto_train = utils.loso_datasets_dl(train_data=train_data, test_data=test_data)
            id = utils.decode_string_columns(pd.DataFrame(test_set['id'].iloc[0][0], columns=['participant'], index=[0]), decode_dict=self.decode_dict)
            
            
            participant = id.iloc[0].item()
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(fold,participant))
            
            training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(label_train.values))
            testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(label_test.values))
          
          
            train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE*2, shuffle=True)
            
            x_size =np.shape(train_set)
            self.parameters['input_size'] = x_size[1]
            self.parameters['hidden'] = int(x_size[1]*0.75)
            self.parameters['output_size'] = int(x_size[1]*0.5)
            self.parameters['output'] = 1
            
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()


         

            model =  lstmnet.LSTMNET(**self.parameters)
            model = model.double()
            model = model.to(self.device)
            memory_after = torch.cuda.memory_allocated()
            total_params = sum(p.numel() for p in model.parameters())
            print('total params ', total_params)
            print(f'Model size: {memory_after - memory_before:.7f}')
            criterion = torch.nn.MSELoss()
            optimzier = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimzier, T_0=int(EPOCHS*0.2))
            opt = {'epochs':EPOCHS,
            'max_lr': 0.08182919525169613,
            'pct_start': 0.07130462374112945,
            'anneal_strategy': 'linear',
            'cycle_momentum': True,
            'base_momentum': 0.5817843343801046,
            'max_momentum': 9.72154707691482e-05,
            'div_factor': 27,
            'final_div_factor': 1.1952405888515022e-05,
            'three_phase': True,
            'steps_per_epoch': 9}

           
            scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimzier, **opt)
          
            scheduler = scheduler2

            for epoch in range(EPOCHS):
                counter = counter+1
               
                model, loss=self.train(model=model, train_loder=train_loader, optimizer=optimzier, scheduler=scheduler, epoch=epoch,criterion=criterion)
                model, loss= self.test(model=model,test_loder=test_loader, epoch=epoch, criterion=criterion)
                




                if epoch ==0:
                    best_model = model
                    best_loss = loss
                elif loss < best_loss:
                    best_model = model
            
           
            del test_loader, train_loader
            torch.cuda.empty_cache()
            gc.collect()
            feature_loader = torch.utils.data.DataLoader(testing_data, batch_size=len(testing_data))
            features, target = self.get_features(best_model, feature_loader)
            del feature_loader
            torch.cuda.empty_cache()
            gc.collect()
            features_pred.append(features)
            target_pred.append(target)
        
            torch.save(best_model, 'bestmodel_tcn.pt')
        
        self.dump_files([features_pred, target_pred],  self.saving_path)

        return metrics_dict, features_pred


    def optuna_train(self,n_iterations:int , sampler,warmup_steps:int, study_name:str='regression',direction:str ='minimize')->dict:
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

        Returns
        -------
        dict
            paramters
        """
        results_tpe={}
        study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, reduction_factor=6
                ),
                sampler=sampler,
                direction=direction, study_name=study_name)
        study.optimize(self.objective, n_trials=n_iterations, catch=(NameError,AttributeError, ModuleNotFoundError, TypeError, ValueError, AssertionError), gc_after_trial=True)


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
    
    def func_list_(self,module:object)->list:
        """This function will list all possible functions corresponing to a module of a package

        Parameters
        ----------
        module : object
            module of package or class, e.g., torch.nn.init

        Returns
        -------
        list
            retruns the names of the functions in a list

        Example
        -------
            `func_list_(torch.nn.init)` \n
            `l = ['normal_', 'xavier_norm_', ...]` \n
            `getattr(torch.nn.init, l[0])(...)`    \n
            
        """
        func_list = []
        for a  in inspect.getmembers(module):
            if inspect.isfunction(a[1]) and a[0].endswith('_') and (not a[0].startswith('_')):
                func_list.append(a[0])
                
        return func_list 



    def get_possible_hidden_sizes(self, input_size, num_heads):
        """get the size of the encoder layer, for the bottleneck
        Parameters
        ----------
        input_size : float
            size of the input
        num_heads : float
            number of heads

        Returns
        -------
        list
            of possible sizes of the hidden layer
        """
        return [i for i in range(input_size // 2 - 2, input_size - 2) if i % num_heads == 0]

    def objective(self, trial):
        """Generate a trial for finetuning. Sets up the search space and will determine the best parameters over a number of trials

        Parameters
        ----------
        trial : Optuna.Trial
            trial to be optimized

        Returns
        -------
        float
            value which marks the sucess of the trial; will be reported

        Raises
        ------
        optuna.exceptions.TrialPruned
            unsuccessful trials will be pruned
        optuna.exceptions.TrialPruned
            unsuccessful trials will be pruned
        """
       
        remove_list = ['Threshold', 'Hardshrink', 'LogSigmoid', 'Softplus', 'Softshrink', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax', 'MultiheadAttention']
        activations = [None] + [func for func in torch.nn.modules.activation.__all__ if func not in remove_list]
        activations.remove(None)
        # ['relu', 'tanh', 'leaky_relu', 'sigmoid', 'elu', 'gelu', 'selu', 'softmax', 'log_softmax']
        print('activations',activations)
        parameter = {
           "input_size": None,
        "dropout": trial.suggest_float('dropout', 0.001, 0.5, log=True),
        "attention": trial.suggest_categorical('attention', ['True', 'False']),
        "sparse": trial.suggest_categorical('sparse', ['True', 'False']),
        "attention_dropout": trial.suggest_float('attention_dropout', 0.001, 0.5, log=True),
        "num_layer": trial.suggest_int('num_layer', 2, 8),
        "bidirectional":trial.suggest_categorical('bidirectional', [True, False]),
        "output": 1,
        'act':trial.suggest_categorical('act', activations),

        }
        

        
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        
        epochs = trial.suggest_int('epochs', 20, 50)
        batch_size = trial.suggest_int('batch_size', 32, 64)
       
        criterion = torch.nn.MSELoss()

        # Cosine Annealing Warmrestart
        t_0 = trial.suggest_int('t_0', 5, epochs)
        t_mult = trial.suggest_int('t_mult',1, 5)
        eta_min = trial.suggest_float('eta_min', 1e-8,lr, log=True)

        # ReduceLROnPlateau
        threshold = trial.suggest_float('threshold',1e-5, 1e-2, log=True)
        threshold_mode = trial.suggest_categorical('threshold_mode', ['rel', 'abs'])
        cooldown = trial.suggest_float('cooldown', 0, 10)
        eps = trial.suggest_float('eps',1e-9, 1e-5, log=True)
        patience = trial.suggest_int('patience', 3, 20)
        factor = trial.suggest_float('factor', 1e-5, 0.9, log=True)

        #OneCycleLR
        max_lr =  trial.suggest_float('max_lr',1e-5, 1e-1, log=True)

        pct_start= trial.suggest_float('pct_start',4e-3, 9e-1, log=True)
        anneal_strategy=trial.suggest_categorical('anneal_startegy', ['cos', 'linear'])
        cycle_momentum=trial.suggest_categorical('cycle_momentum', [True, False]),
        base_momentum= trial.suggest_float('base_momentum',1e-3, 9e-1, log=True)
        max_momentum= trial.suggest_float('max_momentum',1e-5, 9e-1, log=True)
        div_factor= trial.suggest_int('div_factor',1, 50)
        final_div_factor= trial.suggest_float('final_div_factor',1e-5, 1e-3, log=True)
        three_phase=trial.suggest_categorical('three_phase', [False, True])




        test_set, train_set, label_test, label_train, auto_test, auto_train =  utils.loso_finetuning_dl(self.data, windowsize=self.WINDOW_SIZE, overlap=self.OVERLAP)
        
        x_size =np.shape(train_set) 
            
        parameter['input_size'] = x_size[1]
        parameter['output'] = 1 #XXX
        INPUT_SIZE = x_size[1]
        parameter['num_heads'] = trial.suggest_int('num_heads', 1, 1) 
        
       
        
        parameter["hidden"] =  trial.suggest_int('hidden', int(INPUT_SIZE*1.25), int(INPUT_SIZE*3)) 
        temp = parameter['hidden']
        parameter["output_size"] = trial.suggest_int('output_size', int(INPUT_SIZE*1.25), temp)
        model = lstmnet.LSTMNET(**parameter)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
       
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)


        training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_train.values.tolist(), dtype=np.float64)))
        testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_test.values.tolist(), dtype=np.float64)))
          

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True, drop_last=True)
        steps_per_epoch=trial.suggest_int('steps_per_epoch', 1, len(train_loader))
        
        #scheduler
        scheduler_sugg = trial.suggest_categorical('scheduler', ["CosineAnnealingWarmRestarts",  "OneCycleLR"])#, "CosineDecayTrackerRate"])
        

        if scheduler_sugg == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=t_0, T_mult=t_mult, eta_min=eta_min)
        elif scheduler_sugg ==  "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=factor, threshold=threshold, threshold_mode=threshold_mode, patience=patience, cooldown=cooldown, min_lr=eta_min, eps=eps)
        elif scheduler_sugg ==   "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start, anneal_strategy=anneal_strategy, cycle_momentum=cycle_momentum, base_momentum=base_momentum,max_momentum=max_momentum, div_factor=div_factor, final_div_factor=final_div_factor, three_phase=three_phase)



        model = model.double()
        model = model.to(self.device)
        print('Model is training on ', self.device)
        print(f'Model size: {self.get_model_size(model):.2f} MB')
        for epoch in range(epochs):
               
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    print(data.size())
                    data = data.to(self.device)
                    target = target.to(self.device)

                    
                    y_pred = model(data)

                    del data
                    gc.collect()
                    torch.cuda.empty_cache()

                    loss = criterion(y_pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                    
                    

                
                    
                    del loss
                    gc.collect()
                    torch.cuda.empty_cache()


                loss_values = []
                model.eval()
                for batch_idx, (data, target) in enumerate(test_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)

                    
                    y_pred = model(data)

                    del data
                    gc.collect()
                    torch.cuda.empty_cache()

                    loss = criterion(y_pred, target)


                    
                    
                    del target
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    loss_value = loss.item()
                    print(loss_value)
                    loss_values.append(loss_value)
                    del loss, loss_value
                    gc.collect()
                    torch.cuda.empty_cache()   
                    mean_loss = np.nanmean(np.array(loss_values))
                    #print(mean_loss)
                    trial.report(mean_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned() 

                scheduler.step()
                mean_loss = np.nanmean(np.array(loss_values))
                print(mean_loss)
                trial.report(mean_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned() 




                
      
        return np.array(loss_values).mean()


    def get_features(self, model:lstmnet.LSTMNET, dataloader:torch.utils.data.dataloader)-> torch.tensor:
        """Will generate the feature from the test data using only the trained autoencoder

        Parameters
        ----------
        model : lstmnet.LSTMNET
            trained model
        dataloader : torch.utils.data.dataloader
            dataloader containing the data for the feature generation

        Returns
        -------
        torch.tensor
            features
        """
        feature_generator = model
        feature_generator.eval()
        predicitions = []
        targets = []
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            pred = feature_generator(data)
            del data
            torch.cuda.empty_cache()
            gc.collect()
            predicitions.append(pred)
            targets.append(target)
            del target, pred
            torch.cuda.empty_cache()
            gc.collect()
        preds = torch.cat(predicitions, dim=0) 
        target = torch.cat(targets, dim=0)
        target = target.detach().cpu()
        preds = preds.detach().cpu()  

        
        return preds, target

    def fine_tune(self)->dict:
        """Fine tunes the AE

        Returns
        -------
        dict
            parameter
        """
        parameters = self.optuna_train(self.n_interations, self.sampler, self.warm_upsteps, self.study_name, self.direction)
        date = datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")
        
        with open('best_parameters_' +date+'.txt', 'w') as f:
            print(parameters, file=f)
        return parameters

    def dump_files(self, files_list: list, path: str) -> None:
            """ This function will take the variable name and use it as a file name to dump it in pickle files

            :param files_list: list of variable names which should be dumped, just copy and paste the names
            :param path: path which you want to dump the files
            :return: None
            """""

            for file in files_list:
                callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                name = [var_name for var_name, var_val in callers_local_vars if var_val is file]
                if name[0].startswith('_') and name[0].endswith('_') and len(name) > 1:
                    name[0] = name[1]
                with open(path + '/' + name[0] + '.pickle', 'wb') as handle:
                    pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)





    def run(self) ->dict:
        """Runs the training and the testing

        Returns
        -------
        dict
            metrics and features
        """
        
        metrics_dict, features= self.train_test_call(self.data,self.parameter, len(self.data), name_label=self.name_label, random_seeds=77, use_weighting=True, window_size=self.WINDOW_SIZE, overlap=self.OVERLAP, learning_rate=3.4e-3, device=self.device, num_classes=3)   
        
        return metrics_dict, features


# add finetuning lol
import argparse
parser = argparse.ArgumentParser(description='Single AE training, testing and finetuning')
parser.add_argument('--windowsize', default=30*32, type=int, help='Size of the moving timewindow, equals the sequence length of the input data')
parser.add_argument('--overlap', default=15*32, type=int, help='Steps/Overlap the moving timewindow should slide. Recommended size: windowsize//2')
parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
parser.add_argument('--trials', default=1000, type=int, help='Number of trials Optuna should run to determine the best parameters')
parser.add_argument('--fold', default=0, type=int, help='From which Fold you should start (<fold)')
args = parser.parse_args()
parameters = { "attention": True,
        "sparse": True,
        "num_heads": 4,
        "attention_dropout": 0.2,
        "dropout": 0.1,
        "num_layer": 2,
        "bidirectional": True,
        "output": 1,
        "output_size": None}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = r'/path/to/data' # 
f = args.fold or 0
windowsize =args.windowsize or 30*32
overlap = args.overlap or 15*32


data, decode_dict = utils.preprare_data(path=path, use_case='dl_feature')

if args.finetune:
    n_iterations = args.trials or 1000
    trial_name = '_finetuning'
    tcn = lstm_model(data=data, parameters=parameters, WINDOW_SIZE=windowsize, OVERLAP=overlap, n_iterations=n_iterations, study_name=trial_name)
    f = tcn.fine_tune()
else:
    trial_name = '_intial'
    tcn = lstm_model(data=data,parameters=parameters, WINDOW_SIZE=windowsize, OVERLAP=overlap, n_iterations=1000, study_name=trial_name, decode_dict=decode_dict)
    metrics_dict, final_feature_data = tcn.train_test_call(data=data)

    
