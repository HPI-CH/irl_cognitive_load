#!/usr/bin/env python
# coding: utf-8



# # PIPELINES
# ## Preprocessing Pipeline
# - will process the sensor data of the participants in parallel and add the labels to them, the output of this pipe will be the DL features or the clean data with labels
# 
# ## Feature Pipeline
# - will process the clean sensor data with label into features of a given time window, for example 30 sec
# - for running this pipeline you need to run the preprocessing pipeline first.
# 
# ## Ouput
# - will be saved in a) pickle files b) csv-files c) it will create plots for the preprocessing

# In[279]:


import empatica_prep
import empatica_prep.ppg.ppg_functions as bvp
import empatica_prep.acc.acc_functions as acc
import empatica_prep.eda.eda_functions as gsr
import empatica_prep.temp.temp_functions as temp
import empatica_prep.utils.reading_functions as read
import pandas as pd
import numpy as np
import os 
import argparse
import inspect, pickle



parser = argparse.ArgumentParser(description='Prpeprocessing ich werde ')
parser.add_argument('--part', default=1, type=int, help='welcher Teil der Daten verabritet werden soll')
args = parser.parse_args()

if os.getcwd().startswith('/path'): # TODO change paths
    if args.part == 1:
        print('starting preprocessing part ', args.part)
        PATH = r'/path/to/part1'
        PATH_PLOT = r'/path/to/save'
    elif args.part == 2:
        print('starting preprocessing part ', args.part)
        PATH = r'/path/to/part2'
        PATH_PLOT = r'/path/to/save'
    else:
        print('starting preprocessing part ', args.part)
        PATH = r'/path/to/part3'
        PATH_PLOT = r'/path/to/save'
else: 
    PATH = r'/path/to/part3'
    PATH_PLOT = r'/path/to/save'
SENSOR_NAME = 'PIPELINE_NK'
SAVEPATH = PATH_PLOT+'/'+SENSOR_NAME
if not os.path.isdir(SAVEPATH):
    os.mkdir(SAVEPATH)


# Time and shared frequency
shared_fs = 32
timewindow = 30
overlap = 15







participants = read.read_and_sort_data(PATH=PATH)




sampling_rate_dict = {
    'EDA':4,
    'GSR':4,
    'TEMP':4,
    'PPG':64,
    'BVP':64,
    'ACC':32
}



def pipe_1(participants_data:dict, sampling_rate_dict:dict, shared_fs:int=32, participants_id:str=None, path:str=None)-> tuple[pd.DataFrame, pd.DataFrame]:
    if not path: raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
    try: 
        os.path.isdir(path)
    except Exception:raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
    if not participants_id:
        raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
    try: os.path.isdir(path+'/'+participants_id)
    except Exception:
        raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
    
    acc_clean = acc.clean_acc(participants_data['ACC'], sampling_rate=sampling_rate_dict['ACC'])
    bvp_clean, _ = bvp.ppg(participants_data['BVP'],acc=acc_clean, sampling_freq=sampling_rate_dict['PPG'], path=SAVEPATH, name='PPG')
    bvp_clean = bvp_clean.resample(str(1/64)+'s').ffill(limit = 1).interpolate('pchip', limit_direction = 'both').sort_index()
    eda_clean = gsr.eda(participants_data['EDA'], acc_clean=acc_clean, sampling_freq=sampling_rate_dict['EDA'], show=False)
    temp_clean = temp.temp(participants_data['TEMP'], acc=acc_clean, sampling_freq=sampling_rate_dict['TEMP'])
    
    complete_path = path+'/'+participants_id

    temp_clean.name = 'TEMP'
   
    acc_inter = acc_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
    bvp_inter = bvp_clean.resample(str(1/shared_fs)+'s').ffill(limit=2)
    eda_inter =eda_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
    temp_inter = temp_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
    
    

    acc_inter.columns = 'ACC_'+acc_clean.columns
    all_sensor_data = pd.concat([acc_inter, bvp_inter, eda_inter, temp_inter], axis='columns')
    complete_def , all_tsk_list = read.get_label_and_task(signal=all_sensor_data, path=complete_path)
    return complete_def, all_tsk_list, eda_clean, bvp_clean, temp_clean, acc_clean
    
    




def remove_duplicate_index(df_or_series):
    """
    Remove duplicate indices from a DataFrame or Series.

    Parameters
    ----------
    df_or_series : pandas.DataFrame or pandas.Series
        The DataFrame or Series from which to remove duplicate indices.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        A DataFrame or Series with duplicate indices removed.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    >>> index = ['a', 'b', 'a']
    >>> df = pd.DataFrame(data, index=index)
    >>> df_cleaned = remove_duplicate_index(df)
    >>> print(df_cleaned)
       A  B
    a  1  4
    b  2  5

    >>> series = pd.Series([1, 2, 3], index=['a', 'b', 'a'])
    >>> series_cleaned = remove_duplicate_index(series)
    >>> print(series_cleaned)
    a    1
    b    2
    dtype: int64
    """
    df_or_series = df_or_series[~df_or_series.index.duplicated(keep='first')]
    return df_or_series



def pipe_2(bvp_clean, eda_clean, temp_clean, acc_clean,sampling_rate_dict:dict, timewindow:int=30, overlap:int=15,participants_id:str=None, path:str=None, acc_option:str='A')-> pd.DataFrame:
    
    # catching errors
    if not path: raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
    try: 
        os.path.isdir(path)
    except Exception:raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
    if not participants_id:
        raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
    try: os.path.isdir(path+'/'+participants_id)
    except Exception:
        raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
    

    # read in the labels for ml labels
    complete_path = path+'/'+participants_id
    acc_complete_df, acc_clean_lst = read.get_label_and_task(signal=acc_clean, path=complete_path)
    bvp_complete_df, bvp_clean_lst = read.get_label_and_task(signal=bvp_clean, path=complete_path)
    eda_complete_df, eda_clean_lst = read.get_label_and_task(signal=eda_clean, path=complete_path)
    temp_complete_df, temp_clean_lst = read.get_label_and_task(signal=temp_clean, path=complete_path)

    # Features
    bvp_features = bvp.features(ppg=bvp_clean_lst, sampling_freq=sampling_rate_dict['PPG'], timewindows=timewindow, overlap=overlap)
    eda_features = gsr.feature(tasks_lst_eda=eda_clean_lst, eda_complete=eda_complete_df, sampling_freq=sampling_rate_dict['EDA'], timewindow=timewindow, overlapp=overlap)
    temp_features = temp.generate_statistical_features(temp_lst=temp_clean_lst, sampling_rate=sampling_rate_dict['TEMP'], timewindow=timewindow, overlap=overlap)
    acc_features = acc.features(acc_list=acc_clean_lst, acc_complete=acc_complete_df, sampling_rate=sampling_rate_dict['ACC'], timewindow=timewindow, overlap=overlap, option=acc_option)

    # Merge
    acc_features = remove_duplicate_index(acc_features)
    eda_features = remove_duplicate_index(eda_features)
    temp_features = remove_duplicate_index(temp_features)
    bvp_features = remove_duplicate_index(bvp_features)
    all_ml_features = pd.concat([bvp_features, eda_features, temp_features, acc_features], axis='columns')
    all_ml_features = all_ml_features.loc[:,~all_ml_features.columns.duplicated()].copy()
    all_ml_features = all_ml_features.dropna(thresh=80, axis='index').dropna(axis='columns', how = 'all')

    #
    return all_ml_features, [bvp_features, eda_features, temp_features,acc_features], [acc_complete_df, bvp_complete_df, eda_complete_df, temp_complete_df] 



def run(participants_data:dict, sampling_rate_dict:dict, shared_fs:int=32, participants_id:str=None, path:str=None, timewindow:int=30, overlap:int = 15):
    """
    Run the. Signal cleaning and artifact correction for all sensors (ACC, BVP, EDA, TEMP)Run the complete Empatica E4 preprocessing and feature-extraction pipeline
        2. Resampling to a shared frequency for sensor fusion
        3. Task and label alignment based on participant-specific annotation files
        4. Sliding-window feature extraction for each modality
        5. Merging and post-processing of all extracted features

    Each invocation of this function is independent and side-effect free,
    making it safe to use in parallelized batch processing workflows.

    Parameters
    ----------
    participants_data : dict
        Dictionary containing raw Empatica sensor signals for one participant.
        Expected keys are:
            - 'ACC'  : Accelerometer signal
            - 'BVP'  : Blood Volume Pulse signal
            - 'EDA'  : Electrodermal Activity signal
            - 'TEMP' : Skin temperature signal
        Values must be pandas Series or DataFrames indexed by timestamps.

    sampling_rate_dict : dict
        Dictionary mapping each sensor to its original sampling rate (in Hz).
        Expected keys:
            - 'ACC'
            - 'PPG' (for BVP)
            - 'EDA'
            - 'TEMP'

    shared_fs : int, optional (default=32)
        Target sampling frequency (Hz) used to resample all signals
        before feature extraction and sensor fusion.

    participants_id : str
        Unique participant identifier.
        Must correspond to a directory within `path` containing
        task and label annotation files.

    path : str
        Path to the parent directory containing all participant folders.
        Used to load task descriptions and labels.

    timewindow : int, optional (default=30)
        Length of the sliding window (in seconds) used for feature extraction.

    overlap : int, optional (default=15)
        Overlap (in seconds) between consecutive sliding windows.

    Returns
    -------
    list
        A list with the following elements:

        [0] complete_def : pandas.DataFrame
            Fully time-aligned, resampled, and labeled sensor data
            combining all modalities.

        [1] all_tsk_list : list
            List of task-segmented data structures returned by the
            label/task alignment step.

        [2] all_feat : pandas.DataFrame
            Final merged feature table containing all extracted features
            from ACC, BVP, EDA, and TEMP, cleaned and ready for ML modeling.

        [3] solo_feat : list of pandas.DataFrame
            Individual feature tables in the following order:
                [BVP_features, EDA_features, TEMP_features, ACC_features]

        [4] compl_feat : list of pandas.DataFrame
            Task-aligned, labeled signal DataFrames corresponding to:
                [ACC, BVP, EDA, TEMP]

    Notes
    -----
    - This function performs all I/O for label and task alignment internally.
    - It validates paths and participant IDs early to fail fast in parallel runs, use it in a jupyter notebook with ipyparallel
    - Duplicate time indices are removed before feature merging.
    - Designed for embarrassingly parallel execution (one participant per job).


    """


    
    
    
    import empatica_prep
    import empatica_prep.ppg.ppg_functions as bvp
    import empatica_prep.acc.acc_functions as acc
    import empatica_prep.eda.eda_functions as gsr
    import empatica_prep.temp.temp_functions as temp
    import empatica_prep.utils.reading_functions as read
    import pandas as pd
    import numpy as np
    import os 
    
    def pipe_1(participants_data:dict, sampling_rate_dict:dict, shared_fs:int=32, participants_id:str=None, path:str=None)-> tuple[pd.DataFrame, pd.DataFrame]:
        if not path: raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
        try: 
            os.path.isdir(path)
        except Exception:raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
        if not participants_id:
            raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
        try: os.path.isdir(path+'/'+participants_id)
        except Exception:
            raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
        
        acc_clean = acc.clean_acc(participants_data['ACC'], sampling_rate=sampling_rate_dict['ACC'])
        bvp_clean, _ = bvp.ppg(participants_data['BVP'],acc=acc_clean, sampling_freq=sampling_rate_dict['PPG'], path=SAVEPATH, name='PPG')
        bvp_clean = bvp_clean.resample(str(1/64)+'s').ffill(limit = 1).interpolate('pchip', limit_direction = 'both').sort_index()
        eda_clean = gsr.eda(participants_data['EDA'], acc_clean=acc_clean, sampling_freq=sampling_rate_dict['EDA'], show=False)
        temp_clean = temp.temp(participants_data['TEMP'], acc=acc_clean, sampling_freq=sampling_rate_dict['TEMP'])
      
        complete_path = path+'/'+participants_id
       
        temp_clean.name = 'TEMP'
        # get DL Features
        acc_inter = acc_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
        bvp_inter = bvp_clean.resample(str(1/shared_fs)+'s').ffill(limit=2)
        eda_inter =eda_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
        temp_inter = temp_clean.resample(str(1/shared_fs)+'s').interpolate('pchip', limit_direction='both')
      
        

        acc_inter.columns = 'ACC_'+acc_clean.columns
        all_sensor_data = pd.concat([acc_inter, bvp_inter, eda_inter, temp_inter], axis='columns')
        complete_def , all_tsk_list = read.get_label_and_task(signal=all_sensor_data, path=complete_path)
        return complete_def, all_tsk_list, eda_clean, bvp_clean, temp_clean, acc_clean
        
    

    def remove_duplicate_index(df_or_series):
        """
        Remove duplicate indices from a DataFrame or Series.

        Parameters
        ----------
        df_or_series : pandas.DataFrame or pandas.Series
            The DataFrame or Series from which to remove duplicate indices.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            A DataFrame or Series with duplicate indices removed.

        Examples
        --------
        >>> import pandas as pd
        >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        >>> index = ['a', 'b', 'a']
        >>> df = pd.DataFrame(data, index=index)
        >>> df_cleaned = remove_duplicate_index(df)
        >>> print(df_cleaned)
        A  B
        a  1  4
        b  2  5

        >>> series = pd.Series([1, 2, 3], index=['a', 'b', 'a'])
        >>> series_cleaned = remove_duplicate_index(series)
        >>> print(series_cleaned)
        a    1
        b    2
        dtype: int64
        """
        df_or_series = df_or_series[~df_or_series.index.duplicated(keep='first')]
        return df_or_series

    def pipe_2(bvp_clean, eda_clean, temp_clean, acc_clean,sampling_rate_dict:dict, timewindow:int=30, overlap:int=15,participants_id:str=None, path:str=None, acc_option:str='A')-> pd.DataFrame:
    
    # catching errors
        if not path: raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
        try: 
            os.path.isdir(path)
        except Exception:raise ValueError("Please specificy the correct path. It should mount to the parent folder of the pparticipants!")
        if not participants_id:
            raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
        try: os.path.isdir(path+'/'+participants_id)
        except Exception:
            raise ValueError("No valid participant id passed. The ID is required to retrieve the label and the task description!")
        

        # read in the labels for ml labels
        complete_path = path+'/'+participants_id
        acc_complete_df, acc_clean_lst = read.get_label_and_task(signal=acc_clean, path=complete_path)
        bvp_complete_df, bvp_clean_lst = read.get_label_and_task(signal=bvp_clean, path=complete_path)
        eda_complete_df, eda_clean_lst = read.get_label_and_task(signal=eda_clean, path=complete_path)
        temp_complete_df, temp_clean_lst = read.get_label_and_task(signal=temp_clean, path=complete_path)

        # Features
        bvp_features = bvp.features(ppg=bvp_clean_lst, sampling_freq=sampling_rate_dict['PPG'], timewindows=timewindow, overlap=overlap)
        eda_features = gsr.feature(tasks_lst_eda=eda_clean_lst, eda_complete=eda_complete_df, sampling_freq=sampling_rate_dict['EDA'], timewindow=timewindow, overlapp=overlap)
        temp_features = temp.generate_statistical_features(temp_lst=temp_clean_lst, sampling_rate=sampling_rate_dict['TEMP'], timewindow=timewindow, overlap=overlap)
        acc_features = acc.features(acc_list=acc_clean_lst, acc_complete=acc_complete_df, sampling_rate=sampling_rate_dict['ACC'], timewindow=timewindow, overlap=overlap, option=acc_option)

        # Merge
        acc_features = remove_duplicate_index(acc_features)
        eda_features = remove_duplicate_index(eda_features)
        temp_features = remove_duplicate_index(temp_features)
        bvp_features = remove_duplicate_index(bvp_features)
        all_ml_features = pd.concat([bvp_features, eda_features, temp_features, acc_features], axis='columns')
        all_ml_features = all_ml_features.loc[:,~all_ml_features.columns.duplicated()].copy()
        all_ml_features = all_ml_features.dropna(thresh=80, axis='index').dropna(axis='columns', how = 'all')

        #
        return all_ml_features, [bvp_features, eda_features, temp_features,acc_features], [acc_complete_df, bvp_complete_df, eda_complete_df, temp_complete_df] 


    
    complete_def, all_tsk_list,eda_clean, bvp_clean, temp_clean, acc_clean= pipe_1(participants_data, sampling_rate_dict=sampling_rate_dict, shared_fs=shared_fs, participants_id=participants_id, path=path)
    all_feat, solo_feat, compl_feat =pipe_2(bvp_clean=bvp_clean, eda_clean=eda_clean, temp_clean=temp_clean, acc_clean=acc_clean, sampling_rate_dict=sampling_rate_dict, timewindow=timewindow, overlap=overlap, participants_id=participants_id, path=path, acc_option='A')
    return [complete_def, all_tsk_list, all_feat, solo_feat, compl_feat]




def pipeline(participants:dict, sampling_rate_dict:dict, shared_fs:int=32, path:str=None, timewindow:int=30, overlap:int = 15):
    """
    Prepare and execute the Empatica processing pipeline for multiple participants.

    This function serves as a lightweight orchestration wrapper around the
    `run` function. It prepares all participant-specific inputs, aligns shared
    configuration parameters, and sequentially executes the full Empatica
    processing pipeline for each participant.

    The function is intentionally simple and stateless. While it executes
    sequentially itself, it is commonly used to *prepare data structures*
    that can be passed to a parallel backend (e.g., `ipyparallel`) or to
    collect results after parallel execution.

    Parameters
    ----------
    participants : dict
        Dictionary mapping participant IDs to their raw Empatica data.
        Format:
            {
                participant_id (str): participants_data (dict)
            }
        Each `participants_data` dictionary must contain the raw signals
        expected by `run` (ACC, BVP, EDA, TEMP).

    sampling_rate_dict : dict
        Dictionary mapping each sensor to its original sampling rate (in Hz).
        Passed unchanged to all participant runs.

    shared_fs : int, optional (default=32)
        Target sampling frequency (Hz) used to resample all signals
        before feature extraction.

    path : str
        Path to the parent directory containing participant folders with
        task and label annotation files.

    timewindow : int, optional (default=30)
        Length of the sliding window (in seconds) used for feature extraction.

    overlap : int, optional (default=15)
        Overlap (in seconds) between consecutive sliding windows.

    Returns
    -------
    dict
        Dictionary mapping participant IDs to the outputs of `run`.

        Format:
            {
                participant_id (str): [
                    complete_def,
                    all_tsk_list,
                    all_feat,
                    solo_feat,
                    compl_feat
                ]
            }

        Each value corresponds exactly to the return structure of `run`.

    Notes
    -----
    - All shared parameters (sampling rates, window sizes, paths) are replicated
    internally to ensure consistent configuration across participants.
    - This function performs no data cleaning or feature extraction itself;
    all processing logic is delegated to `run`.
    - Suitable for use as a preprocessing step before or after parallel execution.

    Example
    -------
    >>> results = pipeline(
    ...     participants=participants_dict,
    ...     sampling_rate_dict=fs_dict,
    ...     path="/data/empatica"
    ... )
    >>> features_p01 = results["P01"][2]
    """
    id_list = list(participants.keys())
    participants_data_list = list(participants.values())
    sampling_dict_list = [sampling_rate_dict]*len(participants)
    timewindow_list = [timewindow]*len(participants)
    overlap_list = [overlap]*len(participants)
    shared_fs_list  = [shared_fs]*len(participants)
    path_list =[path]*len(participants)
    results = {}
    for i in zip(id_list, participants_data_list, sampling_dict_list,shared_fs_list, path_list, timewindow_list, overlap_list):
       
        results[i[0]] = run(participants_data=i[1], sampling_rate_dict=i[2], shared_fs=i[3],path=i[4], participants_id=i[0], timewindow=i[5], overlap=i[6])

  
    return results 







# dump it

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



preprocessing_results2 = pipeline(participants, sampling_rate_dict,shared_fs, PATH, timewindow, overlap)
dump_files([preprocessing_results2, participants], SAVEPATH)

