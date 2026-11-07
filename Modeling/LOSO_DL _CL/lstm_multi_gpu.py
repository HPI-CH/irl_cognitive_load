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
#import torchmetrics as metric
import numpy as np
import pandas as pd
import lstmnet
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
from torch.cuda.amp import autocast, GradScaler

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
    os.environ["MASTER_PORT"] = "12382"
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    gc.collect()
    import datetime

    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600)) #setting a higher timeout

def run(rank: int, world_size: int, parameters:dict, data:dict,decode_dict:dict=None, window_size:int=384, overlap:int = None):
        print('started spawn')


        saving_path = r'/path/to/folder'

        trial_name= datetime.datetime.strftime(datetime.datetime.today(),"%Y-%m-%d %H-%M-%S")+'before_tuning'
        saving_path = saving_path+'/lstm_compl_'+trial_name
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)

        if not overlap: overlap = window_size//2
        
        ddp_setup(rank, world_size)
        BATCH_SIZE = 16
        EPOCHS = 26
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
        
        predicitions_per_fold = {}
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
            torch.cuda.empty_cache()
            data2 = copy.deepcopy(data)
            train_data = np.array(data2, dtype=object)[train_idx].tolist()
            test_data = np.array(data2, dtype=object)[test_idx].tolist()
            
            test_set, train_set, label_test, label_train, auto_test, auto_train = utils.loso_datasets_dl(train_data=train_data, test_data=test_data)
            id = utils.decode_string_columns(pd.DataFrame(test_set['id'].iloc[0][0], columns=['participant'], index=[0]), decode_dict=decode_dict)
            drop = ['label', 'participant', 'cat_2', 'task', 'id', 'PPG_Hr', 'PPG_Pulse_Peaks', 'EDA_Raw']
            test_set = utils.clean_up_features(test_set, extra_columns_to_drop=drop)
            train_set = utils.clean_up_features(train_set, extra_columns_to_drop=drop)
            label_train_cl = utils.map_labels(labels=label_train)
            label_test_cl = utils.map_labels(label_test)
            participant = id.iloc[0].item()
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(fold,participant))
            
            training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float32)), torch.from_numpy(label_train_cl))
            testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float32)), torch.from_numpy(label_test_cl))
          
          
            train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE*2, shuffle=True)
            
            if len(np.unique(label_train_cl))==1:continue
            participant = id.iloc[0].item()
            print('Fold {} - Testset: Participant {} ... ✧ ✶𐙚⋆°｡⋆♡ '.format(fold,participant))
            
            training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float32)), torch.from_numpy(np.array(label_train_cl.tolist(), dtype=np.float32)))
            testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float32)), torch.from_numpy(np.array(label_test_cl.tolist(), dtype=np.float32)))
          
            
            sampler_train = DistributedSampler(training_data,  num_replicas=world_size, rank=rank)
            sampler_test = DistributedSampler(testing_data, num_replicas=world_size, rank=rank)
            
            train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler_train, num_workers=0,pin_memory=True)
            test_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler_test,num_workers=0,pin_memory=True)
            torch.cuda.set_device(local_rank)
            torch.cuda.empty_cache()
            gc.collect()


            x_size =np.shape(train_set)
            parameters['input_size'] = x_size[1]
            parameters['output'] = 1
            model =  lstmnet.LSTMNET(**parameters)
            model = model.float()
            model = model.to('cuda:' + str(local_rank))
            model = DDP(model, device_ids=[local_rank])
                
            
            

       
            
            criterion = torch.nn.BCELoss()
            optimzier = torch.optim.SGD(
                  params=model.parameters(),
                  lr=lr
            )
            # optimzier = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimzier, T_0=int(EPOCHS*0.2))
            opt = {'epochs':EPOCHS,
             }
            scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimzier, **opt)
          
            scheduler = scheduler2
            scaler = torch.amp.GradScaler(rank)
            for epoch in range(EPOCHS):
                counter = counter+1
                train_loader.sampler.set_epoch(epoch)
                model = train(model=model, train_loder=train_loader, optimizer=optimzier, scheduler=scheduler, epoch=epoch,criterion=criterion, local_rank=local_rank, scaler=scaler)
                test_loader.sampler.set_epoch(epoch)
                model, loss, predicitions = test(model=model,test_loder=test_loader, epoch=epoch, criterion=criterion, local_rank=local_rank)
                

                


                if epoch ==0:
                    best_model = model
                    best_loss = loss
                    preds = predicitions
                elif loss < best_loss:
                    best_model = model
                    best_loss = loss
                    preds = predicitions
            metrics_dict['Fold model'+str(fold)] = best_model
            predicitions_per_fold['Preds_'+str(fold)]=preds #save the best predictions :)
            predicitions_per_fold['Truths_'+str(fold)]=label_test
           
            if local_rank ==0:
                torch.save(best_model.module.state_dict(), 'bestmodel.pt')
            
        dump_files([features_pred, target_pred],  saving_path)



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








def train(model: lstmnet.LSTMNET,train_loder: torch.utils.data.DataLoader, optimizer: torch.optim,scheduler:torch.optim, epoch:str, criterion:torch.nn.BCELoss, local_rank, scaler:torch.cuda.amp.GradScaler)->tuple[lstmnet.LSTMNET, list]:
        """Training of the autoencoder: Contains the training loop, and uses the MSE as loss function combined with a scheduler and Adam as the optimizer. 

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : lstmnet.LSTMNET
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
        lstmnet.LSTMNET, list
            the trained model and a list containing the mean loss
            
        """
 
        model.train()

     
        ce_mean = MeanMetric()
        ce_mean.reset()

       
        


        for batch_idx, (data, target) in enumerate(train_loder):
            
            data = data.to(local_rank, non_blocking=True)
            target = target.to(local_rank).unsqueeze(1)
            torch.cuda.empty_cache()
            gc.collect()
           
            optimizer.zero_grad()
            
            
            y_pred = model(data)
            del data
            gc.collect()
            torch.cuda.empty_cache()
            loss = criterion(y_pred, target)
            del y_pred
            gc.collect()
            torch.cuda.empty_cache()
            
            del target
            gc.collect()
            torch.cuda.empty_cache()
           
            
            # Scaled backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

         
            ce_mean.update(loss.cpu())
           
            
            if batch_idx % 100 == 0 and local_rank == 0:
                tqdm.write('Epoch {} (train) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
                           

        del loss
        gc.collect()
        torch.cuda.empty_cache()
        scheduler.step()
        return model

def test(model: lstmnet.LSTMNET,test_loder: torch.utils.data.DataLoader,  epoch:str, criterion:torch.nn.CrossEntropyLoss, local_rank:str)->tuple[lstmnet.LSTMNET, list]:
        """Testing of the autoencoder: Contains the testing loop, and uses the MSE as loss function.

        Parameters
        ----------
        fold : str
            Fold of the KFOLD/LOOCV validation currently unused
        model : lstmnet.LSTMNET
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
        lstmnet.LSTMNET, list
            the model and a list containing the mean loss
        """
        model.eval()
       
        #ce_mean = metric.MeanMetric().to(local_rank)
        ce_mean = MeanMetric(device=local_rank)
        
        ce_mean.reset()
      
        test_loss = []
        y_pred_list = []


        for batch_idx, (data, target) in enumerate(test_loder):
            
            data = data.to(local_rank)
            target = target.to(local_rank).unsqueeze(1)
            torch.cuda.empty_cache()
            gc.collect()
           



            y_pred = model(data)
            del data
            loss = criterion(y_pred, target)

            
            
            gc.collect()
            torch.cuda.empty_cache()

            loss = criterion(y_pred, target)
                
            gc.collect()
            torch.cuda.empty_cache()
            if local_rank ==0:
                    y_pred_list.append(y_pred.detach().cpu().numpy())
            else:
                 del y_pred
                 torch.cuda.empty_cache()
                 gc.collect()        
            
            
            
            del target
            gc.collect()
            torch.cuda.empty_cache()
           
            ce_mean.update(loss)
            

            if batch_idx % 100 == 0:
                
                tqdm.write('Epoch {} (test) -- loss: {:.4f}'.format(epoch+1, ce_mean.compute()))
          
            del loss
            gc.collect()
            torch.cuda.empty_cache()
        test_loss.append(ce_mean.compute())

   
        
        return model, test_loss, y_pred_list
        
def get_features(model:lstmnet.LSTMNET, dataloader:torch.utils.data.dataloader, local_rank)-> torch.tensor:
        """Will generate the feature from the test data using only the trained autoencoder

        Parameters
        ----------
        model : lstmnet.LSTMNET
            traineed autoencoder
        dataloader : torch.utils.data.dataloader
            dataloader containing the data for the feature generation

        Returns
        -------
        torch.tensor
            features
        """
        feature_generator = model.module
        feature_generator.eval()
        predicitions = []
        targets = []
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(local_rank)
            pred = feature_generator(data)
            predicitions.append(pred)
            targets.append(target)
        preds = torch.stack(predicitions, dim=0) 
        target = torch.stack(targets, dim=0)
        target = target.detach().cpu()
        preds = preds.detach().cpu()  
        
        return preds, target


def optuna_train(local_rank,world_size, n_iterations:int , sampler,return_dict:dict,data:str,windowsize:int=30*32, overlap:int=15*32, study_name:str='regression',direction:str ='maximize')->dict:
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
        dataset, _ = utils.preprare_data(path=data, use_case='dl_feature')
     
        print('ok')
        if local_rank == 0:
            study = optuna.create_study(pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, reduction_factor=6
                ),
                sampler=sampler,
                direction=direction, study_name=study_name)
            study.optimize(
                    partial(objective, rank=local_rank, data=dataset, windowsize=windowsize, overlap=overlap),
                    n_trials=n_iterations,
                    timeout=3000,
                    catch=(NameError,AttributeError, ModuleNotFoundError, TypeError, ValueError, AssertionError), gc_after_trial=True
                )
            return_dict["study"] = study
        else:
                for _ in range(n_iterations):
                    try:
                        objective(None, rank=local_rank, data=dataset,windowsize=windowsize, overlap=overlap)
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
        #results_tpe['study'] = study
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
        
        
        
        torch.cuda.empty_cache()
        gc.collect()
        world_size = torch.cuda.device_count()
        #INPUT_SIZE = np.shape(train_set)[1]
        trial = optuna.integration.TorchDistributedTrial(single_trial)
        
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
        batch_size = 158
       
        criterion = torch.nn.BCELoss()

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



        
        test_set, train_set, label_test, label_train, auto_test, auto_train =  utils.loso_finetuning_dl(data, windowsize=windowsize, overlap=overlap)
        drop = ['label', 'participant', 'cat_2', 'task', 'id', 'PPG_Hr', 'PPG_Pulse_Peaks', 'EDA_Raw']
        test_set = utils.clean_up_features(test_set, extra_columns_to_drop=drop)
        train_set = utils.clean_up_features(train_set, extra_columns_to_drop=drop)
        if len(np.unique(label_train))<=2:
                     label_train_cl = label_train
                     label_test_cl = label_test
        else:     
                    label_train_cl = utils.map_labels(labels=label_train)
                    label_test_cl = utils.map_labels(label_test)
        x_size =np.shape(train_set) # getting the size and setting the models parameter right.
            # HACK getting the parameters right of the input
        parameter['input_size'] = x_size[1]
        parameter['output'] = 1 #XXX
        INPUT_SIZE = x_size[1]
        parameter['num_heads'] = trial.suggest_int('num_heads', 1, 1) # Generiere mögliche Werte für hidden_size_encoder, die durch num_heads teilbar sind 
        
       
        
         
        parameter["hidden"] =  trial.suggest_int('hidden', int(INPUT_SIZE*1.25), int(INPUT_SIZE*3)) 
        temp = parameter['hidden']
        parameter["output_size"] = trial.suggest_int('output_size', int(INPUT_SIZE*1.25), temp)
        

        model =lstmnet.LSTMNET(**parameter).to('cuda:'+str(local_rank))
        model = DDP(model, device_ids=[local_rank])
        
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"])
       
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

        

        training_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_train_cl.tolist(), dtype=np.float64)))
        testing_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_set.values.tolist(), dtype=np.float64)), torch.from_numpy(np.array(label_test_cl.tolist(), dtype=np.float64)))
          
       
        sampler_train = DistributedSampler(training_data)
        sampler_test = DistributedSampler(testing_data)
        batch_size = batch_size*world_size
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, sampler=sampler_train)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, sampler=sampler_test)
      
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
                    target = target.to(local_rank).unsqueeze(1)
                    torch.cuda.empty_cache()
                    gc.collect()
                    #target = target.to(local_rank)
                   
                    
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
                    
                    data = data.to(local_rank)
                    target = target.to(local_rank).unsqueeze(1)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
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
    parameters = { "attention": True,
        "sparse": True,
        "num_heads": 4,
        "attention_dropout": 0.2,
        "dropout": 0.1,
        "num_layer": 2,
        "bidirectional": True,
        "output": 1,
        "output_size": None}
    path = r'/path/to/data'
    



   # TODO Paths
    windowsize =30*32
    overlap = 15*32

    data, decode_dict = utils.preprare_data(path=path, use_case='dl_feature')

    trial_name = 'lstm_multigepulosos'
    sampler_optuna = optuna.samplers.TPESampler()  
    world_size = torch.cuda.device_count()
    
    print(torch.cuda.is_available())

    print('start')

    if args.finetune:
        #torch.set_num_threads(1)
        n_iterations = args.trials or 1000
        manager = mp.Manager()
        return_dict = manager.dict()
        print('start finetuning')
        print(world_size)
        mp.spawn(optuna_train, args=(world_size,n_iterations, sampler_optuna, return_dict, path, windowsize, overlap), nprocs=world_size)
    else:
       
        
        print('start spawn')
        print(world_size)
        mp.spawn(run, args=(world_size,parameters,data,decode_dict, windowsize, overlap), nprocs=world_size)
