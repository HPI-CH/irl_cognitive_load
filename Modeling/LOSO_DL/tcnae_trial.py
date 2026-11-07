import copy
import datetime
from functools import partial
import inspect
import pickle
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import pandas as pd
import numpy as np
import sklearn
import utils
from AE.tcn import TCNAttentionAutoencoder
import gc
import tqdm
from statistics import mean
from sklearn.model_selection import KFold
import optuna
import math
from typing import Optional, Any
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


class MeanMetric:
    """ Class Mean Metric, calculates the mean. It collects the values and then if you want it calculates the mean. It does the same as 
        torchmetrics.mean_metric.
    """
    def __init__(self,device='cpu')->None:
          self.device = device
          self.reset()



    def update(self, value:list[float])->None:
        """Updates the mean metric, you can give the function your loss and it will add the sum of your values and counts them

        Parameters
        ----------
        value : list[float]
            list of floats, e.g., loss values
        """
        self.sum += value.sum().item()
        self.count += value.numel()

    def compute(self)->float:
        """Will compute the mean across all values which were given into the metric

        Returns
        -------
        float
            mean
        """
        return self.sum / self.count if self.count != 0 else 0.0
    
    def reset(self)->None:
        """Will reset the metric, so that you can restart with mean calculations
        """
        self.sum = 0.0
        self.count = 0




def ddp_setup(rank:int, world_size:int, port:int=12357)-> None:
    """Initializes the process group with the backend nccl and the number of GPUs/Processes.

    Parameters
    ----------
    rank : int
        Unique identifier of each process
    world_size : int
        Total number of processes
    """
    port = str(port)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port or "12357"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run(rank: int, world_size: int, parameters:dict, data:dict, decode_dict:dict=None, window_size:int=384, overlap:int = None, f:int=0, port:int=12345)->None:
        """Will start the training and the testing as a "subprocess" with torch.distrubuted parallel. Notice that this only runs on cuda-compatible gpus on a single node.
            This function will generate equal sized data chunks for each gpu. On each gpu runs one process. This function will also split the model, so it can be trained simoustansly on two gpus.
            Please don#t pass the rank as a parameter, since it will be automatically set by the multiprocessing spawn function.


        Parameters
        ----------
        rank : int
            logical id of the process, corresponds to the number of the GPU
        world_size : int
            total number of process = total number of GPUs
        parameters : dict
            parameters of the model
        data : dict
            training and testing data
        name_label : str, optional
            col name where to label is saved, by default 'label'
        window_size : int, optional
            size of the sliding window, by default 384
        overlap : int, optional
            overlap between two windows, if none is passsed then the overlap = window_size //2, by default None
        """


        saving_path = r'/path/to/folder'

        trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'before_tuning'
        saving_path = saving_path+'/TCNAE_'+trial_name
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)
        if not overlap: overlap = window_size//2
        
        ddp_setup(rank, world_size,port=port)
        BATCH_SIZE = 16
        EPOCHS = 16 
        lr =  0.001
        best_model = None
        counter = 0
        kfold = KFold(n_splits=len(data))
        random_seeds = 77
        torch.manual_seed(random_seeds)
        metrics_dict = {}
        features_pred = []
        target_pred = []
        local_rank = rank
       
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
            
            if int(fold)<f:
                 print('Skipped FOld: '+ str(fold))
                 continue
            torch.cuda.empty_cache()
            data2 = copy.deepcopy(data)
            
            train_data = np.array(data2, dtype=object)[train_idx].tolist()
            test_data = np.array(data2, dtype=object)[test_idx].tolist()
            
            test_set, train_set, label_test, label_train, auto_test, auto_train = utils.loso_datasets_dl(train_data=train_data, test_data=test_data)
            id = utils.decode_string_columns(pd.DataFrame(test_set['id'].iloc[0][0], columns=['participant'], index=[0]), decode_dict=decode_dict)
            
            
            participant = id.iloc[0].item()
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(fold,participant))
            
            training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(auto_train))
            testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(auto_test))
          
            
            sampler_train = DistributedSampler(training_data)
            sampler_test = DistributedSampler(testing_data)
            BATCH_SIZE = BATCH_SIZE*world_size
            train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler_train, num_workers=1)
            test_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler_test,num_workers=1)
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
            gc.collect()
            x_size =np.shape(train_set) # getting the size and setting the models parameter right.
            # HACK getting the parameters right of the input
            parameters['input_size'] = x_size[1]
            parameters['output_size'] = x_size[1] #XXX
            model = TCNAttentionAutoencoder.TCNAAE(**parameters)
            model = model.double()
            model = model.to('cuda:' + str(local_rank))
            model = DDP(model, device_ids=[local_rank])
                
            
            

       
            
            criterion = torch.nn.MSELoss()
            optimzier = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimzier, T_0=int(EPOCHS*0.2))
            opt = {'epochs':EPOCHS,
            }
            scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimzier, **opt)
          
            scheduler = scheduler2
        
            for epoch in range(EPOCHS):
                counter = counter+1
                train_loader.sampler.set_epoch(epoch)
                model = train(model=model, train_loder=train_loader, optimizer=optimzier, scheduler=scheduler, epoch=epoch,criterion=criterion, local_rank=local_rank)
                test_loader.sampler.set_epoch(epoch)
                model, loss= test(model=model,test_loder=test_loader, epoch=epoch, criterion=criterion, local_rank=local_rank)
                

                model.modules()


                if epoch ==0:
                    best_model = model
                    best_loss = loss
                elif loss < best_loss:
                    best_model = model
                    best_loss = loss
        
            del train_loader
            del test_loader
            torch.cuda.empty_cache()
            gc.collect()
            feature_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE)
            if local_rank ==0:
                get_features(best_model, feature_loader, local_rank=local_rank, path = saving_path, fold = fold)
                
            del feature_loader 
            torch.cuda.empty_cache()
            gc.collect()
         


        destroy_process_group()

def dump_files(files_list: list, path: str) -> None:
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








def train(model: TCNAttentionAutoencoder.TCNAAE,train_loder: torch.utils.data.DataLoader, optimizer: torch.optim,scheduler:torch.optim, epoch:str, criterion:torch.nn.MSELoss, local_rank)->tuple[TCNAttentionAutoencoder.TCNAAE, list]:
        """Training of the autoencoder: Contains the training loop, and uses the MSE as loss function combined with a scheduler and Adam as the optimizer. 

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : LSTMAttentionAutoencoder.LAAE
            LSTM-Attention-Autoencoder
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

       
        ce_mean = MeanMetric(device=local_rank)

        ce_mean.reset()

       
        


        for batch_idx, (data, target) in enumerate(train_loder):
            
            data = data.to(local_rank)
            target = target.to(local_rank)
           
            
            y_pred = model(data)
           
            del target
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(y_pred, data)
            
        
            
            del data
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         
            ce_mean.update(loss)
           
            if batch_idx % 100 == 0:
              
               
                tqdm.write('Epoch {} (train) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
           
        
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
        return model

def test(model: TCNAttentionAutoencoder.TCNAAE,test_loder: torch.utils.data.DataLoader,  epoch:str, criterion:torch.nn.CrossEntropyLoss, local_rank)->tuple[ TCNAttentionAutoencoder.TCNAAE, list]:
        """Testing of the autoencoder: Contains the testing loop, and uses the MSE as loss function.

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : LSTMAttentionAutoencoder.LAAE
            LSTM-Attention-Autoencoder
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
        LSTMAttentionAutoencoder.LAAE, list
            the model and a list containing the mean loss
        """
        model.eval()
       
      
        ce_mean = MeanMetric(device=local_rank)
        ce_mean.reset()
      
        test_loss = []


        for batch_idx, (data, target) in enumerate(test_loder):
            
            data = data.to(local_rank)
            target = target.to(local_rank)
           


            
            y_pred = model(data)

            del target
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(y_pred, data)
            
     
            
            del data
            gc.collect()
            torch.cuda.empty_cache()
           
            ce_mean.update(loss)
            

            if batch_idx % 100 == 0:
                
                tqdm.write('Epoch {} (test) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
          
            del loss
            gc.collect()
            torch.cuda.empty_cache()
        test_loss.append(ce_mean.compute())

   
        
        return model, test_loss
        

def get_features(model:TCNAttentionAutoencoder.TCNAAE, dataloader:torch.utils.data.dataloader, local_rank,path, fold=None)-> torch.tensor:
        """Will generate the feature from the test data using only the trained autoencoder

        Parameters
        ----------
        model : TCNAttentionAutoencoder.TCNAAE
            traineed autoencoder
        dataloader : torch.utils.data.dataloader
            dataloader containing the data for the feature generation

        Returns
        -------
        torch.tensor
            features
        """
        feature_generator = model.module.encoder
        feature_generator.eval()
        predicitions = []
        targets = []
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(local_rank)
            pred = feature_generator(data)
            del data
            torch.cuda.empty_cache()
            predicitions.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
            del pred, target
            torch.cuda.empty_cache()
            gc.collect()
        preds = torch.cat(predicitions, dim=0) 
        target = torch.cat(targets, dim=0)
        target = target.detach().cpu()
        preds = preds.detach().cpu() 
        trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")# for the name
        if not fold:
             name = 'tcn_features_'+trial_name
        else: 
            name = 'tcn_features_'+trial_name+'_fold_'+str(fold)      

        dump_files_for_shap([target, preds], suffix=name, path=path)
        del target, predicitions
        torch.cuda.empty_cache()
        gc.collect()
def dump_files_for_shap(files_list: list, path: str, suffix: str = '') -> None:
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

def optuna_train(local_rank,world_size, n_iterations:int , sampler,return_dict:dict,data, study_name:str='regression',direction:str ='minimize', windowsize=30*32, overlap=30*32)->dict:
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
        
        ddp_setup(local_rank, world_size)
      
        
        if local_rank == 0:
            study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, reduction_factor=6
                ),
                sampler=sampler,
                direction=direction, study_name=study_name)
            study.optimize(
                    partial(objective, rank=local_rank, data=data, windowsize=windowsize,overlap=overlap),
                    n_trials=n_iterations,
                    timeout=300,
                    catch=(NameError,AttributeError, ModuleNotFoundError, TypeError, ValueError, AssertionError), gc_after_trial=True
                )
            return_dict["study"] = study
        else:
                for _ in range(n_iterations):
                    try:
                        objective(None, rank=local_rank, data=data, windowsize=windowsize, overlap=overlap)
                    except optuna.TrialPruned:
                        pass

        destroy_process_group()

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

def get_possible_hidden_sizes(input_size, num_heads):
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

def func_list_(module:object)->list:
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



def objective(single_trial, rank, data, windowsize, overlap):
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
        local_rank = rank
        INPUT_SIZE = 72
        WINDOW_SIZE = windowsize or 30*32
        OVERLAP = overlap or 30*32
        
        trial = optuna.integration.TorchDistributedTrial(single_trial)


        parameter = {
        "input_size": INPUT_SIZE,
        "output_size": INPUT_SIZE,
        "dropout_encoder": trial.suggest_int('dropout_encoder', 0.001, 0.5),
        "dropout_decoder":trial.suggest_int('dropout_encoder', 0.001, 0.5),
        "attention": trial.suggest_categorical('attention', ['True', 'False']),
        "sparse": trial.suggest_categorical('sparse', ['True', 'False']),
        "attention_dropout": trial.suggest_int('attention_dropout', 0.001, 0.5),
        "stride": trial.suggest_int('stride', 1, 5),
        "groups":1,#trial.suggest_int('groups', 1, 5),
        "groups_de":1,#trial.suggest_int('groups_de', 1, 5),
        "padding_mode": trial.suggest_categorical('padding_mode',  ['zeros','reflect','replicate','circular']),
        "padding_mode_de": trial.suggest_categorical('padding_mode_de',  ['zeros','reflect','replicate','circular']),
        "dilation":trial.suggest_int('dilation', 1, 5),
        "dilation_de":1,

        }
        parameter['kernel_size']=2*parameter['stride']
        parameter['stride_de'] = parameter['stride']
        parameter['kernel_size_de'] = parameter['kernel_size']
        parameter["kernel_size_de"] = 2*parameter['stride_de']


        
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        
        
        epochs = trial.suggest_int('epochs', 20, 200)
        batch_size = trial.suggest_int('batch_size', 3, 6)
       
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
        

        world_size = torch.cuda.device_count()


        
        test_set, train_set, label_test, label_train, auto_test, auto_train = utils.loso_finetuning_dl(data, windowsize=WINDOW_SIZE, overlap=OVERLAP)
        
        torch.cuda.empty_cache()
        gc.collect()
        x_size =np.shape(train_set) # getting the size and setting the models parameter right.
        # HACK getting the parameters right of the input
        parameter['input_size'] = x_size[1]
        parameter['output_size'] = x_size[1] #XXX
        INPUT_SIZE = x_size[1]
        parameter['num_heads'] = trial.suggest_int('num_heads', 1, 4) # Generiere mögliche Werte für hidden_size_encoder, die durch num_heads teilbar sind 
        possible_hidden_sizes = [i for i in range(INPUT_SIZE // 2 - 2, INPUT_SIZE - 2) if i % parameter['num_heads'] == 0] 
        possible_hidden_sizes = get_possible_hidden_sizes(INPUT_SIZE, parameter['num_heads']) 
        if len(possible_hidden_sizes) ==0: 
            raise optuna.exceptions.TrialPruned() 
        parameter['num_heads'] =1
        hidden = trial.suggest_int('hidden', 4,11)
        parameter["hidden_size_encoder"] =  hidden
        parameter["hidden_size_decoder"] = trial.suggest_int('hidden_size_encoder',hidden, int(INPUT_SIZE*0.9))
        
        model = TCNAttentionAutoencoder.TCNAAE(**parameter).to('cuda:'+str(local_rank))

        
        model = DDP(model, device_ids=[local_rank])
        
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)


        training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_train.values.tolist(), dtype=np.float64)))
        testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_test.values.tolist(), dtype=np.float64)))
        
       
        sampler_train = DistributedSampler(training_data)
        sampler_test = DistributedSampler(testing_data)
        batch_size = batch_size*world_size
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, sampler=sampler_train)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, sampler=sampler_test)
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
        for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                 
                    data = data.to(local_rank)
                    target = target.to(local_rank)
                   
                    
                    y_pred = model(data)

                    del target
                    gc.collect()
                    torch.cuda.empty_cache()

                    loss = criterion(y_pred, data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                    del data 
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                    
                    

                
                    
                    del loss
                    gc.collect()
                    torch.cuda.empty_cache()


                loss_values = []
                model.eval()
                for batch_idx, (data, target) in enumerate(test_loader):
                    
                    data = data.to(local_rank)
                    target = target.to(local_rank)

                    
                    y_pred = model(data)

                    del target
                    gc.collect()
                    torch.cuda.empty_cache()

                    loss = criterion(y_pred, data)


                    
                    
                    del data
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    loss_value = loss.item()
                    print(loss_value)
                    loss_values.append(loss_value)
                    del loss, loss_value
                    gc.collect()
                    torch.cuda.empty_cache()   
                scheduler.step()
                mean_loss = np.nanmean(np.array(loss_values))
                print(mean_loss)
                trial.report(mean_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned() 




                
      
        return np.array(loss_values).mean()


# MAIN CALL

if __name__ == "__main__":
    import argparse
    import platform
    parser = argparse.ArgumentParser(description='Distrubuted AE training, testing and finetuning')
    parser.add_argument('--windowsize', default=30*32, type=int, help='Size of the moving timewindow, equals the sequence length of the input data')
    parser.add_argument('--fold', default=0, type=int, help='From which Fold you should start (<fold)')
    parser.add_argument('--port', default=12334, type=int, help='choose the port the multi gpus should listen') 
    parser.add_argument('--overlap', default=15*32, type=int, help='Steps/Overlap the moving timewindow should slide. Recommended size: windowsize//2')
    parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
    parser.add_argument('--trials', default=1000, type=int, help='Number of trials Optuna should run to determine the best parameters')
    args = parser.parse_args()
    parameters = {"input_size": None,
        "hidden_size_encoder": None,
        "hidden_size_decoder": None,
        "output_size": None,
        "kernel_size": 4,
        "dropout_encoder": 0.1,
        "dropout_decoder": 0.1,
        "attention": True,
        "sparse": True,
        "num_heads": 1,
        "attention_dropout": 0.2,
        "stride": 1,
        "stride_de": 2,
        "groups": 1,
        "groups_de": 1,
        "padding_mode": 'zeros',
        "padding_mode_de": 'zeros',
        "dilation": 1,
        "dilation_de": 1}
    parameters['kernel_size'] = 2*parameters['stride']
    parameters['stride_de'] = parameters['stride']
    parameters['kernel_size_de'] = parameters['kernel_size']
    parameters["kernel_size_de"] = 2*parameters['stride_de']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    path = r'/path/to/data' # 




    # Setting window size and stepsize of the sliding window
    windowsize =args.windowsize or 30*32
    overlap = args.overlap or 15*32

   
    # get the data
    data, decode_dict = utils.preprare_data(path=path, use_case='dl_feature')

    sampler_optuna = optuna.samplers.TPESampler()  
    world_size = torch.cuda.device_count()
    f = args.fold or 0 # this scripts currently uses k-fold. Unfortunaly due to time limits of the HPC, it only get a specific amount of folds done (1.5 fold), therefore, set the fold number you want to start with.

    if args.finetune:
        n_iterations = args.trials or 1000
        trial_name = '_finetuning'
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(optuna_train, args=(world_size,n_iterations, sampler_optuna, return_dict, data,trial_name,'minimize', windowsize, overlap), nprocs=world_size)
    else:
        trial_name = '_intial'
        mp.spawn(run, args=(world_size,parameters,data,decode_dict,windowsize,overlap,f,args.port), nprocs=world_size)






#rial 0 finished with value: 1.041544878816953 and parameters: {'hidden_size_decoder': 64, 'dropout_encoder': 0.34187491708867296, 'bilstm': False, 'attention': True, 'attention_dropout': 0.4407543969314387, 'num_layers': 2, 'num_layers_decoder': 3, 'num_heads': 4, 'hidden': 8, 'optimizer': 'AdamW', 'lr': 0.003535800124519772, 'epochs': 67, 'batch_size': 16, 't_0': 67, 't_mult': 4, 'eta_min': 6.449494301013649e-05, 'threshold': 8.733656996964008e-05, 'threshold_mode': 'rel', 'cooldown': 7.450852070295138, 'eps': 5.946430760499725e-07, 'patience': 13, 'factor': 0.2904982805535051, 'max_lr': 0.0002016999928270348, 'pct_start': 0.022217605641107148, 'anneal_startegy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.002259811492596662, 'max_momentum': 0.0017135165888352106, 'div_factor': 30, 'final_div_factor': 1.262937625742742e-05, 'three_phase': False, 'steps_per_epoch': 5, 'scheduler': 'OneCycleLR'}. Best is trial 0 with value: 1.041544878816953.
