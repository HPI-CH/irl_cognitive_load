import copy
import datetime
from functools import partial
import inspect
import pickle
import platform
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

import numpy as np
import pandas as pd
from AE.lstm import LSTMAttentionAutoencoder
import gc
import tqdm
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import utils

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




def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # paste in the console
    # export TORCH_DISTRIBUTED_DEBUG=INFO
    # export NCCL_DEBUG=INFO
    # export NCCL_P2P_DISABLE=0
    # export NCCL_P2P_LEVEL=NVL


    print('setting up env')
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12598"
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    gc.collect()
    import datetime

    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def run(rank: int, world_size: int, parameters:dict, data:dict,decode_dict:dict=None, window_size:int=384, overlap:int = None):
        print('started spawn')


        saving_path = r'/path/to/data'

        trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'before_tuning'
        saving_path = saving_path+'/LSTMAEKFOLD_'+trial_name
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)

        if not overlap: overlap = window_size//2
        
        ddp_setup(rank, world_size)
        BATCH_SIZE = 21
        EPOCHS = 10
        lr = 0.001
        best_model = None
        counter = 0
        kfold = KFold(n_splits=5)
        random_seeds = 77
        torch.manual_seed(random_seeds)
        metrics_dict = {}
        features_pred = []
        target_pred = []
        local_rank = rank
        
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
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


            x_size =np.shape(train_set)
            parameters['input_size'] = x_size[1]
            parameters['output_size'] = x_size[1]
            ## self.HIDDEN_SIZE = self.INPUT_SIZE//2+1
            parameters['hidden_size_encoder'] = x_size[1]//2+1
            model = LSTMAttentionAutoencoder.LAAE(**parameters)
            model = model.double()
            model = model.to('cuda:' + str(local_rank))
            # try it :)
            model = torch.compile(model, mode="reduce-overhead")
            model = DDP(model, device_ids=[local_rank])
                
            
            

       
            
            criterion = torch.nn.MSELoss()
            optimzier = torch.optim.AdamW(model.parameters(), lr=lr)
            #scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimzier, T_0=int(EPOCHS*0.2))
            opt = {'epochs':EPOCHS,
            }



            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimzier, T_0=7, T_mult=4, eta_min=1.0677276954162395e-07, last_epoch=-1)
            



           
        
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
            
            del test_loader
            del train_loader
            torch.cuda.empty_cache()
            gc.collect()
            feature_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE*2)
            if local_rank ==0:
                get_features(best_model, feature_loader, local_rank=local_rank, path= saving_path, fold=fold)
               
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








def train(model: LSTMAttentionAutoencoder.LAAE,train_loder: torch.utils.data.DataLoader, optimizer: torch.optim,scheduler:torch.optim, epoch:str, criterion:torch.nn.MSELoss, local_rank)->tuple[LSTMAttentionAutoencoder.LAAE, list]:
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

        # ce_mean =metric.MeanMetric().to(local_rank)
        ce_mean = MeanMetric()

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

def test(model: LSTMAttentionAutoencoder.LAAE,test_loder: torch.utils.data.DataLoader,  epoch:str, criterion:torch.nn.CrossEntropyLoss, local_rank)->tuple[LSTMAttentionAutoencoder.LAAE, list]:
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
        
def get_features(model:LSTMAttentionAutoencoder.LAAE, dataloader:torch.utils.data.dataloader, local_rank,path, fold=None)-> torch.tensor:
        """Will generate the feature from the test data using only the trained autoencoder

        Parameters
        ----------
        model : LSTMAttentionAutoencoder.LAAE
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
             name = 'lstm_features_'+trial_name
        else: 
            name = 'lstm_features_'+trial_name+'_fold_'+str(fold)      

        dump_files_for_shap([target, preds], suffix=name, path=path)
        del target, predicitions
        torch.cuda.empty_cache()
        gc.collect()
        #return preds, target

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

def optuna_train(local_rank,world_size, n_iterations:int , sampler,return_dict:dict,data,windowsize:int=30*32, overlap:int=15*32, study_name:str='regression',direction:str ='minimize')->dict:
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
        
        print('ok')
        if local_rank == 0:
            study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, reduction_factor=6
                ),
                sampler=sampler,
                direction=direction, study_name=study_name)
            study.optimize(
                    partial(objective, rank=local_rank, data=data, windowsize=windowsize, overlap=overlap),
                    n_trials=n_iterations,
                    timeout=300,
                    catch=(NameError,AttributeError, ModuleNotFoundError, TypeError, ValueError, AssertionError), gc_after_trial=True
                )
            return_dict["study"] = study
        else:
                for _ in range(n_iterations):
                    try:
                        objective(None, rank=local_rank, data=data,windowsize=windowsize, overlap=overlap)
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
        WINDOW_SIZE = windowsize
        OVERLAP = overlap
        test_set, train_set, label_test, label_train, auto_test, auto_train = utils.loso_finetuning_dl(data, windowsize=WINDOW_SIZE, overlap=OVERLAP)
        
        torch.cuda.empty_cache()
        gc.collect()
        INPUT_SIZE = np.shape(train_set)[1]
        trial = optuna.integration.TorchDistributedTrial(single_trial)
        parameter  = {
            'input_size':INPUT_SIZE,
           
            
            'output_size':INPUT_SIZE,
            'dropout_encoder': trial.suggest_float('dropout_encoder', 0.01, 0.8),
            'dropout_decoder':trial.suggest_float('dropout_encoder', 0.01, 0.8),
            'bilstm':trial.suggest_categorical('bilstm', [True, False]),
            'attention': trial.suggest_categorical('attention', [True, False]),
            "sparse":False,
            'attention_dropout':trial.suggest_float('attention_dropout', 0.001, 0.89),
            'num_layers':trial.suggest_int('num_layers', 1,8),
            'num_layers_decoder':trial.suggest_int('num_layers_decoder', 1,8)
        }

        parameter['num_heads'] = trial.suggest_int('num_heads', 1, 6) # Generiere mögliche Werte für hidden_size_encoder, die durch num_heads teilbar sind 
        
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        
        epochs = trial.suggest_int('epochs', 10, 80)
        batch_size = trial.suggest_int('batch_size', 2, 6)
       
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


        
    
        INPUT_SIZE = np.shape(train_set)[1]
        parameter['input_size']=INPUT_SIZE
        parameter['output_size'] = INPUT_SIZE
        parameter['hidden_size_decoder']=trial.suggest_int('hidden_size_decoder', INPUT_SIZE/2-2, INPUT_SIZE-2)
        possible_hidden_sizes = [i for i in range(INPUT_SIZE // 2 - 2, INPUT_SIZE - 2) if i % parameter['num_heads'] == 0] 
        possible_hidden_sizes = get_possible_hidden_sizes(INPUT_SIZE, parameter['num_heads']) 
        if len(possible_hidden_sizes) ==0: 
            raise optuna.exceptions.TrialPruned() 
     
        hidden = trial.suggest_int('hidden', 8,15)
        parameter['hidden_size_encoder'] = hidden*parameter['num_heads']
        model =LSTMAttentionAutoencoder.LAAE(**parameter).to('cuda:'+str(local_rank))
        model = DDP(model, device_ids=[local_rank])
        
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
        
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

        

        training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_train.values.tolist(), dtype=np.float64)))
        testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_test.values.tolist(), dtype=np.float64)))
            #np.array(test.values.tolist(), dtype=np.float64)
       
        sampler_train = DistributedSampler(training_data)
        sampler_test = DistributedSampler(testing_data)
        batch_size = batch_size*world_size
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8, sampler=sampler_train)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=4, shuffle=False, drop_last=True, num_workers=8, sampler=sampler_test)
      
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
        #model = model.to(device)
        for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                 
                    data = data.to(local_rank)
                    torch.cuda.empty_cache()
                    gc.collect()
                    #target = target.to(local_rank)
                   
                    
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
                    #target = target.to(local_rank)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
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
                    mean_loss = np.nanmean(np.array(loss_values))
                    print(mean_loss)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distrubuted AE training, testing and finetuning')
   
    parser.add_argument('--trials', default=500, type=int, help='Number of trials the models should be finetuned')
    parser.add_argument('--finetune', default=False, type=bool, help='If the AE should be finetuned or not')
    args = parser.parse_args()
    parameters ={
        # TODO add your parameters
       
    }
       
    path = r'/path/to/data' 
    windowsize =30*32
    overlap = 15*32

    data, decode_dict = utils.preprare_data(path=path, use_case='dl_feature')

    trial_name = 'lstm_multigepulosos'
    sampler_optuna = optuna.samplers.TPESampler()  
    world_size = torch.cuda.device_count()
    
    print(torch.cuda.is_available())

    print('start')

    if args.finetune:
       
        n_iterations = args.trials or 1000
        manager = mp.Manager()
        return_dict = manager.dict()
        print('start finetuning')
        print(world_size)
        mp.spawn(optuna_train, args=(world_size,n_iterations, sampler_optuna, return_dict, data, windowsize, overlap), nprocs=world_size)
    else:
        print('start spawn')
        print(world_size)
        mp.spawn(run, args=(world_size,parameters,data,decode_dict, windowsize, overlap), nprocs=world_size)
