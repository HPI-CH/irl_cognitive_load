#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import copy
import pandas as pd
import numpy as np
import scipy.signal
import sklearn
import glob
import pickle
import pathlib
import datetime
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.utils
from typing import Any, Optional
import warnings




def unpickle_data(file_path:str)-> dict: 
        """Unpickle the pickle files, this methods searches for all pickle files and will unpickle the file name participants which is the dict with the processed data from the preprocessing

        Parameters
        ----------
        file_path : str
            Path to the folder containing the pickled files

        Returns
        -------
        dict
            unpickled data, the result of the preprocessing pipeline

        Raises
        ------
        ValueError
            If the path leads to a folder with no pickle files or the wanted pickle file is not there. eye-
        """
        files_list = glob.glob(file_path+r'/*.pickle')
        unpickled_files = {}
        for files in files_list:
            file_open = open(files, 'rb')
            file  = pickle.load(file_open)
            unpickled_files[pathlib.Path(files).stem]= file
          
        return unpickled_files




def extract_elements(results):
    """
    Extract specified elements from a list of lists and return them in a new list.

    Parameters
    ----------
    results : list of lists
        The input list of lists.

    Returns
    -------
    list
        A list containing the extracted elements.

    Examples
    --------
    >>> results = [['finetuning', 'task list', 'ml', 'other', 'clean_with_label'], ['finetuning2', 'task list2', 'ml2', 'other2', 'clean_with_label2']]
    >>> extract_elements(results)
    [['finetuning', 'finetuning2'], ['task list', 'task list2'], ['ml', 'ml2'], ['clean_with_label', 'clean_with_label2']]
    """
    # Initialize lists to store the extracted elements
    dl_finetuning_list = []
    dl_features_list = []
    ml_features_list = []
    clean_with_label_list = []

    # Iterate over each sublist in the input list of lists
    for sublist in results:
        dl_finetuning_list.append(sublist[0])  # finetuning
        dl_features_list.append(sublist[1])  # task list
        ml_features_list.append(sublist[2])  # ml
        clean_with_label_list.append(sublist[-1])  # clean_with_label

    # Return the extracted elements in a new list
    return dl_finetuning_list, dl_features_list, ml_features_list, clean_with_label_list



def clean_up_features(feature_data:pd.DataFrame, columns_to_definitly_keep:list=[], extra_columns_to_drop:list=[])->pd.DataFrame:
        """Cleans up data and removes all useless features such as peak rate and promiences...

        Parameters
        ----------
        feature_data : pd.DataFrame
            data of each task for the handcrafted feature generation
        columns_to_definitly_keep : list, optional
            columns which should stay but would be dropped normally, by default []
        extra_columns_to_drop : list, optional
            columns which should be dropped as well, by default []

        Returns
        -------
        pd.DataFrame
            features with only useful features
        """




        columns_to_drop = ['columns'] 
        if len(columns_to_definitly_keep) >0:
            for i in columns_to_definitly_keep:
                try:
                    columns_to_drop.remove(i)
                except ValueError:
                    pass
        if len(extra_columns_to_drop)>0:
            columns_to_drop = columns_to_drop+extra_columns_to_drop          
            columns_to_drop = list(set(columns_to_drop))
        for drop in columns_to_drop:
            try:
                feature_data.drop(drop, axis='columns', inplace=True) 
            except KeyError:
                pass    
        return feature_data      



def window(data_col: np.array, window_size:float=5, overlapp:float=3,copy:bool=True)->np.array:
    """Creates overlapping time windows in an array

    Parameters
    ----------
    data_col : np.array
        array with the data, the timewindows should be created from
    window_size : float, optional
        size of the window, by default 5
    overlapp : float, optional
        the number of values the windows should overlapp, by default 3
    copy : bool, optional
        if a view of the data should be returned set to false, by default True

    Returns
    -------
    np.array
        windowed array
    """
    sh = (data_col.size -window_size+1, window_size)
    st = data_col.strides*2
    view = np.lib.stride_tricks.as_strided(data_col, strides=st, shape=sh)[0::overlapp]
    if copy: return view.copy(); return view

def make_dataframe_time_windows(data:pd.DataFrame,label:pd.Series, window_size:float, overlapp:float)->pd.DataFrame:
    """Creates the time windows in the dataframes by creating the windows for each column

    Parameters
    ----------
    data : pd.DataFrame
        the data used for the training
    label : pd.Series
        Target values
    window_size : float
        Window size for the time window.
    overlapp : float
        the number of values the windows should overlapp,

    Returns
    -------
    pd.DataFrame
        windowed dataframe
    """
    windowed_frame = pd.DataFrame()
    index = window(data.index.values, window_size=window_size, overlapp=overlapp)[:, -1]
    windowed_frame.index = index
    label_auto =  window(label.values, window_size=window_size, overlapp=overlapp)
    label = pd.Series(window(label.values, window_size=window_size, overlapp=overlapp)[:, -1], name = label.name, index=index)
    
    for cols in data.columns:
        new_col = window(data[cols].values, window_size=window_size, overlapp=overlapp)
        
        windowed_frame[cols] = new_col.tolist()
    windowed_frame['label'] = label
    windowed_frame = windowed_frame[:-1]
    label_auto=label_auto[:-1]
    label = label[:-1]
    return windowed_frame, label, label_auto   


def normalize(train:list[pd.DataFrame]|list[dict], test:pd.DataFrame|dict,name_label:str = 'label', scale_method:Optional[str]='Standard')->tuple[list[dict[pd.DataFrame]], list[pd.DataFrame], list[dict[pd.DataFrame]], list[pd.DataFrame]]|tuple[list[pd.DataFrame], pd.DataFrame]:
    """Normalize the trainings data if sklearns scaler Min-Max or Standard Scaler
        TODO scharf nachdenken ob es eine gute idee ist das in listen zu packen prinzipiell wenn man die listen parallel macht für die Label ist das kein problem. Hinsichtlich 
        der methode: MulTiTaSk LEarNiNg <3 lol wäre es vielleicht praktischer wenn es immer noch in per Task sortiert ist. Just saying
        XXX 
        XXX
        XXX 

    Parameters
    ----------
    train : list[pd.DataFrame] or dict
        Train data
    test : pd.DataFrame or dict
        Testing data of the (one) participant(s) which was elected for the testset
    scale_method : Optional[str], optional
        choose the scaler the options are standard scaler or min-max scaler, by default 'Standard'

    Returns
    -------
    tuple[list[pd.DataFrame], pd.DataFrame]
        A list containing the scaled dataframes for each task form each participant for the training, dataframe of the participant(s) for the testing data





    Parameters
    ----------
    train : list[pd.DataFrame] | list[dict]
        list/dict conatining the dataframes from different tasks from the participants elected for the traininig
    test : pd.DataFrame | dict
        Testing data of the (one) participant(s) which was elected for the testset
    scale_method : Optional[str], optional
        choose the scaler the options are standard scaler or min-max scaler, by default 'Standard'


    Returns
    -------
    tuple[list[dict[pd.DataFrame]], list[pd.DataFrame], list[dict[pd.DataFrame]], list[pd.DataFrame]]|tuple[list[pd.DataFrame], pd.DataFrame]
        List of dicts with keys corresponding to the tasks, the values are the dataframes, list containing the scaled dataframes for each task form each participant for the training, dataframe of the participant(s) for the testing data

    """
    if scale_method.lower() == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    
    
    if type(train) == list[dict]:fit_dataframe = pd.concat(train, axis=0)
    else: 
        
        
        train2 = copy.deepcopy(train)
        fit_dataframe = pd.concat([pd.concat([*x.values()], axis=0, ignore_index=True) for x in train2], ignore_index=True)#list(np.array([[*x.values()] for x in train]).flat)
        fit_dataframe.drop(['cl'], axis='columns', inplace =True) 
        try:
                    fit_dataframe.drop(['task', 'cat_1', 'cat_2'], axis='columns', inplace = True)

        except KeyError:
                    pass     
    scaler.fit(fit_dataframe)
    train_scaled = []
    test_scaled = []
    train_dicts_scaled = []
    test_dicts_scaled = []
    
    if isinstance(train, list):
        for dicts in train:
            for task in dicts.keys():
                
                label = dicts[task][name_label]
                
                dicts[task].drop([name_label], axis='columns', inplace = True)
                try:
                    dicts[task].drop(['PPG_Quality','EDA_Clean_Rolling_Mean'], axis='columns', inplace = True)

                except KeyError:
                    pass     
                columns = dicts[task].columns
                try:
                    dicts[task] = pd.DataFrame(scaler.transform(dicts[task]), columns=columns)
                    dicts[task].interpolate('cubicspline', order=5, limit_direction='both', inplace =True)
               
                    
                except ValueError: pass
                dicts[task]['label'] = label.values
                train_scaled.append(dicts[task])      
            train_dicts_scaled.append(dicts)    
    else:      
        for frame in train:
            label = frame[name_label]
            frame = frame.drop([name_label], axis='columns')
            temp_frame = pd.DataFrame(scaler.transform(frame), columns=columns)
            temp_frame.interpolate('cubicspline', order=5, limit_direction='both', inplace =True)
            temp_frame['label'] = label.values
            train_scaled.append(temp_frame)

    if isinstance(test, list):
        for dicts in test:
            for task in dicts.keys():
                label = dicts[task][name_label]
                
                dicts[task].drop([name_label], axis='columns', inplace = True)
          
                try:
                    dicts[task] = pd.DataFrame(scaler.transform(dicts[task]), columns=columns)
                    dicts[task].interpolate('cubicspline', order=5, limit_direction='both', inplace =True)
                except ValueError: pass    
                dicts[task]['label'] = label.values
               
                test_scaled.append(dicts[task])
            test_dicts_scaled.append(dicts)
        return test_dicts_scaled, test_scaled, train_scaled, train_dicts_scaled     
    else:
        label = frame[name_label]
        frame = frame.drop([name_label], axis='columns')
        test_scaled = pd.DataFrame(scaler.transform(test), columns=columns)
        temp_frame.interpolate('cubicspline', order=5, limit_direction='both', inplace =True)
        test_scaled['label'] = label

    return train_scaled, test_scaled



def remove_and_interpolate_columns(df, threshold):
    """
    Remove columns from the DataFrame that have more than the specified threshold of NaN values,
    and interpolate NaN values in the remaining columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    threshold : float
        The threshold for the percentage of NaN values in a column. Columns with more than this percentage will be removed.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns removed and NaN values interpolated.

    Examples
    --------
    >>> data = {'A': [1, 2, None, 4], 'B': [None, None, None, None], 'C': [1, None, 3, 4], 'D': [None, 2, 3, None]}
    >>> df = pd.DataFrame(data)
    >>> threshold = 50
    >>> remove_and_interpolate_columns(df, threshold)
         A    C    D
    0  1.0  1.0  NaN
    1  2.0  2.0  2.0
    2  3.0  3.0  3.0
    3  4.0  4.0  3.0
    """
   
    nan_percentage = df.isna().mean() * 100
    
   
    columns_to_keep = nan_percentage[nan_percentage <= threshold].index
    
   
    df_filtered = df[columns_to_keep]
    if not 'cl' in df_filtered.index: df_filtered['cl'] = df['cl']
    
  
    df_interpolated = df_filtered.interpolate('cubicspline', limit_direction = 'both')
    
    return df_interpolated



import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_string_columns(df):
    """
    Automatically detect string columns and encode the string features in the DataFrame.
    Generates a dictionary for decoding the features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        The DataFrame with string columns encoded.
    dict
        A dictionary for decoding the encoded features.

    Examples
    --------
    >>> data = {'A': [1, 2, 3], 'B': ['cat', 'dog', 'mouse'], 'C': ['red', 'blue', 'green']}
    >>> df = pd.DataFrame(data)
    >>> df_encoded, decode_dict = encode_string_columns(df)
    >>> df_encoded
       A  B  C
    0  1  0  2
    1  2  1  0
    2  3  2  1
    >>> decode_dict
    {'B': {0: 'cat', 1: 'dog', 2: 'mouse'}, 'C': {0: 'blue', 1: 'green', 2: 'red'}}
    """
    # Initialize the LabelEncoder and the decode dictionary
    le = LabelEncoder()
    decode_dict = {}
    
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is of type object (string)
        if df[col].dtype == 'object':
            # Encode the string values in the column and store the encoder
            df[col] = le.fit_transform(df[col])
            # Create a dictionary for decoding the encoded values
            decode_dict[col] = dict(zip(le.transform(le.classes_), le.classes_))
    
    return df, decode_dict



def generate_decode_dict(df):
    """
    Generate a dictionary for decoding the features from the first DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    dict
        A dictionary for decoding the encoded features.

    Examples
    --------
    >>> data = {'A': [1, 2, 3], 'B': ['cat', 'dog', 'mouse'], 'C': ['red', 'blue', 'green']}
    >>> df = pd.DataFrame(data)
    >>> decode_dict = generate_decode_dict(df)
    >>> decode_dict
    {'B': {0: 'cat', 1: 'dog', 2: 'mouse'}, 'C': {0: 'blue', 1: 'green', 2: 'red'}}
    """



    le = LabelEncoder()
    decode_dict = {}
    
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column is of type object (string)
        if df[col].dtype == 'object':
            # Special handling for IDs starting with "N931m"
            df[col] = df[col].apply(lambda x: 'N931m3' if x.startswith('N931m') else x)
            
            # Encode the string values in the column and store the encoder
            le.fit(df[col])
            # Create a dictionary for decoding the encoded values
            decode_dict[col] = dict(zip(le.transform(le.classes_), le.classes_))
    
    return decode_dict
  

def encode_string_columns(df, decode_dict):
    """
    Encode string columns in the DataFrame using the provided dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    decode_dict : dict
        A dictionary for decoding the encoded features.

    Returns
    -------
    pd.DataFrame
        The DataFrame with string columns encoded.

    Examples
    --------
    >>> data = {'A': [1, 2, 3], 'B': ['cat', 'dog', 'mouse'], 'C': ['red', 'blue', 'green']}
    >>> df = pd.DataFrame(data)
    >>> decode_dict = {'B': {0: 'cat', 1: 'dog', 2: 'mouse'}, 'C': {0: 'blue', 1: 'green', 2: 'red'}}
    >>> df_encoded = encode_string_columns(df, decode_dict)
    >>> df_encoded
       A  B  C
    0  1  0  2
    1  2  1  0
    2  3  2  1
    """
        # Iterate over each column in the DataFrame
    for col in df.columns:
        # Check if the column has an associated decoder in the dictionary
        if col in decode_dict:
            # Special handling for IDs starting with "N931m"
            if col == 'id' or col == 'participant':
                df[col] = df[col].apply(lambda x: 'N931m3' if x.startswith('N931m') else x)
            
            # Encode the string values in the column using the provided dictionary
            reverse_dict = {v: k for k, v in decode_dict[col].items()}
            df[col] = df[col].map(reverse_dict)
    
    return df
    
  





# In[17]:


import pandas as pd

def decode_string_columns(df, decode_dict):
    """
    Decode the encoded string columns in the DataFrame using the provided dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with encoded string columns.
    decode_dict : dict
        A dictionary for decoding the encoded features.

    Returns
    -------
    pd.DataFrame
        The DataFrame with string columns decoded.

    Examples
    --------
    >>> data = {'A': [1, 2, 3], 'B': [0, 1, 2], 'C': [2, 0, 1]}
    >>> df = pd.DataFrame(data)
    >>> decode_dict = {'B': {0: 'cat', 1: 'dog', 2: 'mouse'}, 'C': {0: 'blue', 1: 'green', 2: 'red'}}
    >>> df_decoded = decode_string_columns(df, decode_dict)
    >>> df_decoded
       A      B      C
    0  1    cat    red
    1  2    dog   blue
    2  3  mouse  green
    """
   
    for col in df.columns:
      
        if col in decode_dict:
         
            df[col] = df[col].map(decode_dict[col])
    
    return df



def standardize_ml(test:list[pd.DataFrame], train:list[pd.DataFrame], scale_method:str='standard')->tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Standardizes features and prepares machine learning datasets.

    This function performs the following steps:
    1. Standardizes the features using the specified scaling method.
    2. Adds the label back to the normalized dataframes.

    Parameters
    ----------
    test : list of pd.DataFrame
        List of dataframes containing the testing data.
    train : list of pd.DataFrame
        List of dataframes containing the training data.
    scale_method : str, optional
        Method for scaling the features ('standard' or 'minmax'). Default is 'standard'.

    Returns
    -------
    norm_train : list of pd.DataFrame
        List of normalized training dataframes with labels.
    norm_test : list of pd.DataFrame
        List of normalized testing dataframes with labels.

    Examples
    --------
    >>> test_data = [pd.DataFrame({'feature1': [1, 2, 3], 'cl': [0, 1, 0]})]
    >>> train_data = [pd.DataFrame({'feature1': [4, 5, 6], 'cl': [1, 0, 1]})]
    >>> norm_train, norm_test = standardize_ml(test_data, train_data)
    """
    if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
    else:
            scaler = sklearn.preprocessing.StandardScaler()
    train2 = copy.deepcopy(train)
    
    colmns_test = pd.concat(test).drop('cl', axis='columns').keys()
    
    fit_framing = pd.concat(train2, axis='index').drop('cl',axis='columns')
    col = fit_framing.columns
    colmns_test = list(set(col).intersection(colmns_test))
    for i in ['EDA_SCR_Peaks', 'EDA_SCR_Height', 'EDA_SCR_Recovery', 'EDA_SCR_Onsets', 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'HRV_TINN', 'HRV_SDRMSSD']: #,'HRV_SDANN2', 'HRV_SDNNI1', 'HRV_SDNNI2', 'HRV_SDANN1', 'HRV_VLF'
        try:
            colmns_test.remove(i)
        except ValueError: pass 
             
    scaler.fit(fit_framing[colmns_test])
    norm_test= []
    norm_train = []
   
    for i in test:
        try: 
            drop =i.loc[i['EDA_Clean']<0].index
            i.drop(drop, axis='index', inplace=True)
        except Exception:pass   
        label =i['cl']
        id = i['id']
        index = i.index
        
        i.drop(['cl'], inplace = True, axis='columns')
        cols = i[colmns_test].columns
        scaled = scaler.transform(i[colmns_test])
        scaled = pd.DataFrame(scaled, columns =cols,index=index)
        scaled['label'] =label
        scaled['id'] =id
        norm_test.append(scaled)
    for i in train:
        try: 
            drop =i.loc[i['EDA_Clean']<0].index
            i.drop(drop, axis='index', inplace=True)
        except Exception:pass   
        label =i['cl']
        id = i['id']
        index = i.index
        
        i.drop(['cl'], inplace = True, axis='columns')
        cols = i[colmns_test].columns
        scaled = scaler.transform(i[colmns_test])
        scaled = pd.DataFrame(scaled, columns =cols,index=index)
        scaled['label'] =label
        scaled['id'] =id
        norm_train.append(scaled) 
    return norm_train, norm_test      

def standardize_dl(test:list[pd.DataFrame], train:list[pd.DataFrame], window_size:int=30*32, overlap:int=30*32//2, scale_method:str='standard')->tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Standardizes and creates time windows for deep learning datasets.

    This function performs the following steps:
    1. Standardizes the features using the specified scaling method.
    2. Creates time windows of the specified size with the desired overlap.
    3. Separates labels and auto features from the dataframes.

    Parameters
    ----------
    test : list of pd.DataFrame
        List of dataframes containing the testing data.
    train : list of pd.DataFrame
        List of dataframes containing the training data.
    window_size : int, optional
        Size of the time windows (default is 30*32).
    overlap : int, optional
        Overlap between the time windows (default is 30*32//2).
    scale_method : str, optional
        Method for scaling the features ('standard' or 'minmax'). Default is 'standard'.

    Returns
    -------
    windowed_frames_test : list of pd.DataFrame
        List of dataframes containing the windowed test data.
    windowed_frames_train : list of pd.DataFrame
        List of dataframes containing the windowed training data.
    label_lst_test : list of pd.DataFrame
        List of dataframes containing the test labels.
    label_lst_train : list of pd.DataFrame
        List of dataframes containing the training labels.
    auto_list_test : list of pd.DataFrame
        List of dataframes containing the test auto features.
    auto_list_train : list of pd.DataFrame
        List of dataframes containing the training auto features.

    Examples
    --------
    >>> test_data = [pd.DataFrame({'feature1': [1, 2, 3], 'cl': [0, 1, 0]})]
    >>> train_data = [pd.DataFrame({'feature1': [4, 5, 6], 'cl': [1, 0, 1]})]
    >>> windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train = standardize_dl(test_data, train_data)
    """
    scale_method = 'standard'
    if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
    else:
            scaler = sklearn.preprocessing.StandardScaler()

    fit_framing = pd.concat([pd.concat(x) for x in train], axis='index').drop('cl',axis='columns')
    scaler.fit(fit_framing)
   
    windowed_frames_test = []
    label_lst_test = []
    auto_list_test = []
    windowed_frames_train = []
    label_lst_train = []
    auto_list_train = []
   
    for t in test:
        for i in t:
            label =i['cl']
            id = i['id']
            index = i.index
            
            i.drop(['cl'], inplace = True, axis='columns')
            cols = i.columns
            scaled = scaler.transform(i)
            scaled = pd.DataFrame(scaled, columns =cols,index=index)
            if len(i)<window_size:continue
            frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=window_size,overlapp=overlap)
            frame['id'] = id
            windowed_frames_test.append(frame)
            label_lst_test.append(label)
            auto_list_test.append(labe_lauto)  
       
    for t in train:
        for i in t:
            label =i['cl']
            index = i.index
            id = i['id']
            i.drop(['cl'], inplace = True, axis='columns')
            cols = i.columns
            scaled = scaler.transform(i)
            scaled = pd.DataFrame(scaled, columns =cols,index=index)
            if len(i)<window_size:continue
            frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=window_size,overlapp=overlap)
            frame['id'] = id
            windowed_frames_train.append(frame)
            label_lst_train.append(label)
            auto_list_train.append(labe_lauto)  
       
    return windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train



def preprare_data(path, use_case:str='ml_feature'):
    """
    Prepares data for different use cases by reading, cleaning, and encoding features.

    This function performs the following steps:
    1. Reads in the data from a pickle file.
    2. Drops columns with excessive missing values and interpolates small missing values.
    3. Encodes categorical features using a decode dictionary.
    4. Extracts and processes the desired data for the specified use case.

    Parameters
    ----------
    path : str
        Path to the pickle file containing the data.
    use_case : str, optional
        Specifies the use case for the data processing ('ml_feature', 'dl_fine', 'dl_feat'). Default is 'ml_feature'.

    Returns
    -------
    list_corr : list of pd.DataFrame
        List of processed dataframes for the specified use case.
    decode_dict : dict
        Dictionary used for encoding categorical features.

    Examples
    --------
    >>> path = 'data/preprocessed_data.pkl'
    >>> use_case = 'ml_feature'
    >>> list_corr, decode_dict = preprare_data(path, use_case)
    """
  
    files = unpickle_data(path)
    results=files['preprocessing_results']
   
   
    if isinstance(results, dict): results = [*results.values()]
    elif not isinstance(results, list): raise ValueError('The data has an incorrect format!')
    dl_finetuning_list, dl_features_list, ml_features_list, clean_with_label_list = extract_elements(results)
    if 'dl_fine' in use_case.lower():
        use_ls = dl_finetuning_list
    elif 'dl_feat' in use_case.lower(): use_ls = dl_features_list 
    else: use_ls=ml_features_list; del dl_features_list, clean_with_label_list, dl_finetuning_list

    if 'dl_fine'in use_case.lower() or 'ml_feat' in use_case.lower():
        list_corr = []
        decode_dict = generate_decode_dict(pd.concat(use_ls))
        for ls in use_ls:
            temp = remove_and_interpolate_columns(ls, 0.2)
            temp = encode_string_columns(temp,decode_dict=decode_dict)
            list_corr.append(temp)
        return list_corr, decode_dict    
    elif 'dl_feat' in use_case.lower():
        compl=pd.concat([pd.concat(x) for x in use_ls])
        decode_dict = generate_decode_dict(compl)
        total = []
        for i in use_ls:
            list_corr = []
            for tasks in i:
                temp = remove_and_interpolate_columns(tasks, 0.2)
                temp = encode_string_columns(temp,decode_dict=decode_dict)
                list_corr.append(temp)
            total.append(list_corr)            
        return total, decode_dict
    






def loso_datasets_ml(train_data:list, test_data:list, drop:list=[]):
    """
    Preprocesses and prepares training and testing datasets for machine learning models.

    This function performs the following steps:
    1. Concatenates the dataframes.
    2. Normalizes the features.
    3. Removes unwanted features from the dataframes if specified.
    4. Encodes categorical features.
    5. Separates labels from the dataframes.
    6. Concatenates the lists into single dataframes for training and testing.

    Parameters
    ----------
    train_data : list of pd.DataFrame
        List of dataframes containing the training data.
    test_data : list of pd.DataFrame
        List of dataframes containing the testing data.
    drop : list, optional
        List of feature names to be dropped from the dataframes (default is []).

    Returns
    -------
    train_set : pd.DataFrame
        Concatenated dataframe of normalized training data.
    test_set : pd.DataFrame
        Concatenated dataframe of normalized testing data.
    label_train : pd.Series
        Series containing the concatenated training labels.
    label_test : pd.Series
        Series containing the concatenated testing labels.

    Examples
    --------
    >>> train_data = [pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'label': [0, 1, 0]})]
    >>> test_data = [pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12], 'label': [1, 0, 1]})]
    >>> drop = ['feature2']
    >>> train_set, test_set, label_train, label_test = loso_datasets_ml(train_data, test_data, drop)
    """
    
    
    norm_train, norm_test = standardize_ml(test=test_data, train=train_data, scale_method='standard')
    if len(drop)>0:
        norm_train = clean_up_features(norm_train, extra_columns_to_drop=drop)
        norm_test = clean_up_features(norm_test, extra_columns_to_drop=drop)
    
    # Concatenate the lists into single dataframes
    train_set = pd.concat(norm_train, axis='index')
    test_set = pd.concat(norm_test, axis='index')
    label_train = train_set['label']
    label_test = test_set['label']
    train_set = train_set.drop('label', axis='columns')
    test_set = test_set.drop('label', axis='columns')
    # get the columns/ features which have been calculated in the first place :-(
    cols = list(set(test_set.dropna(axis='columns').keys()).intersection(set(train_set.dropna(axis='columns').keys())))
    return train_set[cols], test_set[cols], label_train, label_test  
    


# In[117]:


def loso_datasets_dl(train_data:list, test_data:list, drop:list=[]):
    """
    Preprocesses and prepares training and testing datasets for deep learning models.

    This function performs the following steps:
    1. Removes unwanted features from the dataframes if specified.
    2. Normalizes the features.
    3. Creates time windows of the specified size with the desired overlap.
    4. Concatenates the resulting lists into single dataframes for training and testing.

    Parameters
    ----------
    train_data : list of pd.DataFrame
        List of dataframes containing the training data.
    test_data : list of pd.DataFrame
        List of dataframes containing the testing data.
    drop : list
        List of feature names to be dropped from the dataframes.

    Returns
    -------
    test_set : pd.DataFrame
        Concatenated dataframe of windowed test data.
    train_set : pd.DataFrame
        Concatenated dataframe of windowed training data.
    label_test : pd.DataFrame
        Concatenated dataframe of test labels.
    label_train : pd.DataFrame
        Concatenated dataframe of training labels.
    auto_test : pd.DataFrame
        Concatenated dataframe of test auto features.
    auto_train : pd.DataFrame
        Concatenated dataframe of training auto features.

    Examples
    --------
    >>> train_data = [pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})]
    >>> test_data = [pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> drop = ['feature2']
    >>> test_set, train_set, label_test, label_train, auto_test, auto_train = loso_datasets_dl(train_data, test_data, drop)
    """
  
  

    if len(drop)>0:
        train_data = clean_up_features(train_data, extra_columns_to_drop=drop)
        test_data = clean_up_features(test_data, extra_columns_to_drop=drop)
    windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train = standardize_dl(test=test_data, train=train_data, window_size=30*32, overlap=15*32, scale_method='standard')
    
    test_set = pd.concat(windowed_frames_test, axis='index')
    train_set = pd.concat(windowed_frames_train, axis='index')
    label_test = pd.concat(label_lst_test, axis='index')
    label_train = pd.concat(label_lst_train,axis='index')
    auto_train = pd.Series(auto_list_train,name='auto_train')
    auto_test = pd.Series(auto_list_test,name='auto_test')
    return test_set, train_set, label_test, label_train, auto_test, auto_train




import sklearn
from sklearn.model_selection import KFold



def personal_finetune_ml(data:pd.DataFrame, drop:list=[]):
    
    train_set, test_set = sklearn.model_selection.train_test_split(data, shuffle=True)
    test_set, train_set,label_test, label_train = standardize_ml_personal(test=test_set, train=train_set,scale_method='standard')
    if len(drop)>0:
        train_set = clean_up_features(train_set, extra_columns_to_drop=drop)
        test_set = clean_up_features(test_set, extra_columns_to_drop=drop)
    return test_set, train_set,label_test, label_train     





def standardize_ml_personal(test:list[pd.DataFrame], train:list[pd.DataFrame], scale_method:str='standard')->tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Standardizes features and prepares machine learning datasets.

    This function performs the following steps:
    1. Standardizes the features using the specified scaling method.
    2. Adds the label back to the normalized dataframes.

    Parameters
    ----------
    test : list of pd.DataFrame
        List of dataframes containing the testing data.
    train : list of pd.DataFrame
        List of dataframes containing the training data.
    scale_method : str, optional
        Method for scaling the features ('standard' or 'minmax'). Default is 'standard'.

    Returns
    -------
    norm_train : list of pd.DataFrame
        List of normalized training dataframes with labels.
    norm_test : list of pd.DataFrame
        List of normalized testing dataframes with labels.

    Examples
    --------
    >>> test_data = [pd.DataFrame({'feature1': [1, 2, 3], 'cl': [0, 1, 0]})]
    >>> train_data = [pd.DataFrame({'feature1': [4, 5, 6], 'cl': [1, 0, 1]})]
    >>> norm_train, norm_test = standardize_ml(test_data, train_data)
    """
    if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
    else:
            scaler = sklearn.preprocessing.StandardScaler()
    train2 = copy.deepcopy(train)
    colmns_test = test.drop('cl', axis='columns').keys()
    train_in = train2.index
    test_in = test.index
    train_cl = train['cl']
    test_cl = test['cl']
    train= train.drop('cl', axis = 'columns')
    test=test.drop('cl', axis = 'columns')
    train_norm =pd.DataFrame(scaler.fit_transform(train[colmns_test]), columns=train[colmns_test].keys(), index=train_in)
    test_norm =pd.DataFrame(scaler.fit_transform(test[colmns_test]), columns=test[colmns_test].keys(), index=test_in)
    
    return test_norm, train_norm, test_cl, train_cl


# In[ ]:


def personal_datasets_ml(train_data:list, test_data:list, drop:list=[]):
    """
    Preprocesses and prepares training and testing datasets for deep learning models.

    This function performs the following steps:
    1. Removes unwanted features from the dataframes if specified.
    2. Normalizes the features.
    3. Creates time windows of the specified size with the desired overlap.
    4. Concatenates the resulting lists into single dataframes for training and testing.

    Parameters
    ----------
    train_data : list of pd.DataFrame
        List of dataframes containing the training data.
    test_data : list of pd.DataFrame
        List of dataframes containing the testing data.
    drop : list
        List of feature names to be dropped from the dataframes.

    Returns
    -------
    test_set : pd.DataFrame
        Concatenated dataframe of windowed test data.
    train_set : pd.DataFrame
        Concatenated dataframe of windowed training data.
    label_test : pd.DataFrame
        Concatenated dataframe of test labels.
    label_train : pd.DataFrame
        Concatenated dataframe of training labels.
    auto_test : pd.DataFrame
        Concatenated dataframe of test auto features.
    auto_train : pd.DataFrame
        Concatenated dataframe of training auto features.

    Examples
    --------
    >>> train_data = [pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})]
    >>> test_data = [pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> drop = ['feature2']
    >>> test_set, train_set, label_test, label_train, auto_test, auto_train = loso_datasets_dl(train_data, test_data, drop)
    """
    # encode cat features
    # normalize
    # timewindows
    # concat 

    if len(drop)>0:
        train_data = clean_up_features(train_data, extra_columns_to_drop=drop)
        test_data = clean_up_features(test_data, extra_columns_to_drop=drop)
    test_set, train_set, label_test, label_train = standardize_ml_personal(test=test_data, train=train_data,scale_method='standard')
    


    return test_set, train_set, label_test, label_train#, auto_test, auto_train



def personal_datasets_dl(train_data:list, test_data:list, drop:list=[]):
    """
    Preprocesses and prepares training and testing datasets for deep learning models.

    This function performs the following steps:
    1. Removes unwanted features from the dataframes if specified.
    2. Normalizes the features.
    3. Creates time windows of the specified size with the desired overlap.
    4. Concatenates the resulting lists into single dataframes for training and testing.

    Parameters
    ----------
    train_data : list of pd.DataFrame
        List of dataframes containing the training data.
    test_data : list of pd.DataFrame
        List of dataframes containing the testing data.
    drop : list
        List of feature names to be dropped from the dataframes.

    Returns
    -------
    test_set : pd.DataFrame
        Concatenated dataframe of windowed test data.
    train_set : pd.DataFrame
        Concatenated dataframe of windowed training data.
    label_test : pd.DataFrame
        Concatenated dataframe of test labels.
    label_train : pd.DataFrame
        Concatenated dataframe of training labels.
    auto_test : pd.DataFrame
        Concatenated dataframe of test auto features.
    auto_train : pd.DataFrame
        Concatenated dataframe of training auto features.

    Examples
    --------
    >>> train_data = [pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})]
    >>> test_data = [pd.DataFrame({'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> drop = ['feature2']
    >>> test_set, train_set, label_test, label_train, auto_test, auto_train = loso_datasets_dl(train_data, test_data, drop)
    """
   
   

    if len(drop)>0:
        train_data = clean_up_features(train_data, extra_columns_to_drop=drop)
        test_data = clean_up_features(test_data, extra_columns_to_drop=drop)
    windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train = standardize_dl_personal(test=test_data, train=train_data, window_size=30*32, overlap=15*32, scale_method='standard')
    
    test_set = pd.concat(windowed_frames_test, axis='index')
    train_set = pd.concat(windowed_frames_train, axis='index')
    label_test = pd.concat(label_lst_test, axis='index')
    label_train = pd.concat(label_lst_train,axis='index')
    auto_train = pd.Series(auto_list_train,name='auto_train')
    auto_test = pd.Series(auto_list_test,name='auto_test')
    return test_set, train_set, label_test, label_train, auto_test, auto_train


# In[ ]:


def standardize_dl_personal(test:list[pd.DataFrame], train:list[pd.DataFrame], window_size:int=30*32, overlap:int=30*32//2, scale_method:str='standard')->tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Standardizes and creates time windows for deep learning datasets.

    This function performs the following steps:
    1. Standardizes the features using the specified scaling method.
    2. Creates time windows of the specified size with the desired overlap.
    3. Separates labels and auto features from the dataframes.

    Parameters
    ----------
    test : list of pd.DataFrame
        List of dataframes containing the testing data.
    train : list of pd.DataFrame
        List of dataframes containing the training data.
    window_size : int, optional
        Size of the time windows (default is 30*32).
    overlap : int, optional
        Overlap between the time windows (default is 30*32//2).
    scale_method : str, optional
        Method for scaling the features ('standard' or 'minmax'). Default is 'standard'.

    Returns
    -------
    windowed_frames_test : list of pd.DataFrame
        List of dataframes containing the windowed test data.
    windowed_frames_train : list of pd.DataFrame
        List of dataframes containing the windowed training data.
    label_lst_test : list of pd.DataFrame
        List of dataframes containing the test labels.
    label_lst_train : list of pd.DataFrame
        List of dataframes containing the training labels.
    auto_list_test : list of pd.DataFrame
        List of dataframes containing the test auto features.
    auto_list_train : list of pd.DataFrame
        List of dataframes containing the training auto features.

    Examples
    --------
    >>> test_data = [pd.DataFrame({'feature1': [1, 2, 3], 'cl': [0, 1, 0]})]
    >>> train_data = [pd.DataFrame({'feature1': [4, 5, 6], 'cl': [1, 0, 1]})]
    >>> windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train = standardize_dl(test_data, train_data)
    """
    scale_method = 'standard'
    if scale_method.lower() == 'standard':
            scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
            scaler = sklearn.preprocessing.MinMaxScaler()
    else:
            scaler = sklearn.preprocessing.StandardScaler()

    fit_framing = pd.concat(train, axis='index').drop('cl',axis='columns')
    scaler.fit(fit_framing)
    
    windowed_frames_test = []
    label_lst_test = []
    auto_list_test = []
    windowed_frames_train = []
    label_lst_train = []
    auto_list_train = []
    
    for i in test:
        
                label =i['cl']
                index = i.index
                
                i.drop(['cl'], inplace = True, axis='columns')
                cols = i.columns
                scaled = scaler.transform(i)
                scaled = pd.DataFrame(scaled, columns =cols,index=index)
                if len(i)<window_size:continue
                frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=window_size,overlapp=overlap)
                windowed_frames_test.append(frame)
                label_lst_test.append(label)
                auto_list_test.append(labe_lauto)  
    for i in train:
       
            label =i['cl']
            index = i.index
            
            i.drop(['cl'], inplace = True, axis='columns')
            cols = i.columns
            scaled = scaler.transform(i)
            scaled = pd.DataFrame(scaled, columns =cols,index=index)
            if len(i)<window_size:continue
            frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=window_size,overlapp=overlap)
            windowed_frames_train.append(frame)
            label_lst_train.append(label)
            auto_list_train.append(labe_lauto)  


    return windowed_frames_test, windowed_frames_train, label_lst_test, label_lst_train, auto_list_test, auto_list_train





def loso_finetuning_ml(data: list[pd.DataFrame], test_size: float = 0.3, scale_method: str = 'standard') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform Leave-One-Subject-Out (LOSO) fine-tuning for machine learning models.

    This function randomly samples data from tasks for train and test splits, and scales the data using the specified scaling method.

    Parameters
    ----------
    data : list of pd.DataFrame
        List of DataFrames, each representing data from a participant.
    test_size : float, optional
        Proportion of the data to include in the test split (default is 0.1).
    scale_method : str, optional
        Scaling method to use ('standard' or 'minmax', default is 'standard').

    Returns
    -------
    tuple of pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        - train_set_n : Scaled training set.
        - test_set_n : Scaled test set.
        - label_train : Labels for the training set.
        - label_test : Labels for the test set.

    Example
    -------
    >>> import pandas as pd
    >>> data = [pd.DataFrame({'cl': [0, 1, 0], 'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
    >>>         pd.DataFrame({'cl': [1, 0, 1], 'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> train_set_n, test_set_n, label_train, label_test = loso_finetuning_ml(data, test_size=0.2, scale_method='minmax')
    """
    train_samples = []
    test_samples = []
    decode_dict = generate_decode_dict(pd.concat(data, axis='index'))
    for df in data:
        df = df.sample(frac=1, random_state=77)
        df = remove_and_interpolate_columns(df=df, threshold = 0.15)  # Shuffle the data but keep the index
        df = encode_string_columns(df,decode_dict=decode_dict)
        split_index = int(len(df) * (1 - test_size))
        train_samples.append(df.iloc[:split_index])
        test_samples.append(df.iloc[split_index:])

    train_set = pd.concat(train_samples)#.sample(frac=1, random_state=77)
    test_set = pd.concat(test_samples)#.sample(frac=1, random_state=77)

    if scale_method.lower() == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()

    # Align columns
    cols = list(set(test_set.dropna(axis='columns').keys()).intersection(set(train_set.dropna(axis='columns').keys())))
    cols
    train_set = train_set[cols]
    label_train = train_set['cl']
    train_set = train_set.drop(['cl'], axis='columns')
    label_test = test_set['cl']
    test_set = test_set[cols]
    test_set = test_set.drop(['cl'], axis='columns')
    col_tr = train_set.columns
    cols_t = test_set.columns
    train_set_n = pd.DataFrame(scaler.fit_transform(train_set), columns=col_tr)
    test_set_n = pd.DataFrame(scaler.transform(test_set), columns=cols_t)

    return train_set_n, test_set_n, label_train, label_test


# In[ ]:


def loso_finetuning_dl(data: list[pd.DataFrame], test_size: float = 0.1, scale_method: str = 'standard', windowsize: int = 32*30, overlap: int = (30*32)//2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform Leave-One-Subject-Out (LOSO) fine-tuning for deep learning models.

    This function randomly samples data from tasks for train and test splits, scales the data using the specified scaling method, and adds time windows of the desired size.

    Parameters
    ----------
    data : list of pd.DataFrame
        List of DataFrames, each representing data from a participant.
    test_size : float, optional
        Proportion of the data to include in the test split (default is 0.1).
    scale_method : str, optional
        Scaling method to use ('standard' or 'minmax', default is 'standard').
    windowsize : int, optional
        Size of the time windows to create (default is 32*30).
    overlap : int, optional
        Overlap between consecutive time windows (default is (30*32)//2).

    Returns
    -------
    tuple of pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series
        - test_set : Scaled test set with time windows.
        - train_set : Scaled training set with time windows.
        - label_test : Labels for the test set.
        - label_train : Labels for the training set.
        - auto_test : Auto-generated labels for the test set.
        - auto_train : Auto-generated labels for the training set.

    Example
    -------
    >>> import pandas as pd
    >>> data = [pd.DataFrame({'cl': [0, 1, 0], 'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
    >>>         pd.DataFrame({'cl': [1, 0, 1], 'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> test_set, train_set, label_test, label_train, auto_test, auto_train = loso_finetuning_dl(data, test_size=0.2, scale_method='minmax', windowsize=32, overlap=16)
    """
    train_samples = []
    test_samples = []
    if scale_method.lower() == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    decode_dict = generate_decode_dict(pd.concat([pd.concat(x) for x in data], axis='index'), axis = 'index')
    for d in data:
        for df in d:
            df = encode_string_columns(df,decode_dict=decode_dict)
            df = remove_and_interpolate_columns(df, 0.15)
            split_index = int(len(df) * (1 - test_size))
            if len(df) - split_index < windowsize:
                split_index = len(df) - (1 * windowsize)
            train_samples.append(df.iloc[:split_index])
            test_samples.append(df.iloc[split_index:])
        
    train_set = pd.concat(train_samples)
    test_set = pd.concat(test_samples)
    cols = list(set(test_set.dropna(axis='columns').keys()).intersection(set(train_set.dropna(axis='columns').keys())))
    cols.append('cl')
    train_set = train_set[cols]
    train_set = train_set.drop(['cl'], axis='columns')
    test_set = test_set[cols]
    scaler.fit(train_set)
    cols.remove('cl')
    
    windowed_frames_test = []
    label_lst_test = []
    auto_list_test = []
    windowed_frames_train = []
    label_lst_train = []
    auto_list_train = []
    conter = 0
    for i in test_samples:
        conter =+1
     
     
        label = i['cl']
        index = i.index
        i.drop(['cl'], inplace=True, axis='columns')
        colus = i[cols].columns
        scaled = scaler.transform(i[cols])
        scaled = pd.DataFrame(scaled, columns=colus, index=index)
        if len(scaled) < windowsize:
            continue
        frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=windowsize, overlapp=overlap)
        windowed_frames_test.append(frame)
        label_lst_test.append(label)
        auto_list_test.append(labe_lauto)
    
    for i in train_samples:
        label = i['cl']
        index = i.index
        i.drop(['cl'], inplace=True, axis='columns')
        colus = i[cols].columns
        scaled = scaler.transform(i[cols])
        scaled = pd.DataFrame(scaled, columns=colus, index=index)
        if len(i) < windowsize:
            continue
        frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=windowsize, overlapp=overlap)
        windowed_frames_train.append(frame)
        label_lst_train.append(label)
        auto_list_train.append(labe_lauto)
    
    test_set = pd.concat(windowed_frames_test, axis='index')
    train_set = pd.concat(windowed_frames_train, axis='index')
    label_test = pd.concat(label_lst_test, axis='index')
    label_train = pd.concat(label_lst_train, axis='index')
    auto_train = pd.Series(auto_list_train,name='auto_train')
    auto_test = pd.Series(auto_list_test,name='auto_test')
    
    return test_set, train_set, label_test, label_train, auto_test, auto_train


# In[122]:


def personal_finetuning_dl(data: list[pd.DataFrame], test_size: float = 0.1, scale_method: str = 'standard', windowsize: int = 32*30, overlap: int = (30*32)//2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform Leave-One-Subject-Out (LOSO) fine-tuning for deep learning models.

    This function randomly samples data from tasks for train and test splits, scales the data using the specified scaling method, and adds time windows of the desired size.

    Parameters
    ----------
    data : list of pd.DataFrame
        List of DataFrames, each representing data from a participant.
    test_size : float, optional
        Proportion of the data to include in the test split (default is 0.1).
    scale_method : str, optional
        Scaling method to use ('standard' or 'minmax', default is 'standard').
    windowsize : int, optional
        Size of the time windows to create (default is 32*30).
    overlap : int, optional
        Overlap between consecutive time windows (default is (30*32)//2).

    Returns
    -------
    tuple of pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series
        - test_set : Scaled test set with time windows.
        - train_set : Scaled training set with time windows.
        - label_test : Labels for the test set.
        - label_train : Labels for the training set.
        - auto_test : Auto-generated labels for the test set.
        - auto_train : Auto-generated labels for the training set.

    Example
    -------
    >>> import pandas as pd
    >>> data = [pd.DataFrame({'cl': [0, 1, 0], 'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
    >>>         pd.DataFrame({'cl': [1, 0, 1], 'feature1': [7, 8, 9], 'feature2': [10, 11, 12]})]
    >>> test_set, train_set, label_test, label_train, auto_test, auto_train = loso_finetuning_dl(data, test_size=0.2, scale_method='minmax', windowsize=32, overlap=16)
    """
    train_samples = []
    test_samples = []
    if scale_method.lower() == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scale_method.lower() == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()
    decode_dict = generate_decode_dict(pd.concat(data), axis = 'index')
    for df in data:
            df = encode_string_columns(df,decode_dict=decode_dict)
            df = remove_and_interpolate_columns(df, 0.15)
            split_index = int(len(df) * (1 - test_size))
            if len(df) - split_index < windowsize:
                split_index = len(df) - (1 * windowsize)
            train_samples.append(df.iloc[:split_index])
            test_samples.append(df.iloc[split_index:])
        
    train_set = pd.concat(train_samples)
    test_set = pd.concat(test_samples)
    cols = list(set(test_set.dropna(axis='columns').keys()).intersection(set(train_set.dropna(axis='columns').keys())))
    #cols.append('cl')
    train_set = train_set[cols]
    train_set = train_set.drop(['cl'], axis='columns')
    test_set = test_set[cols]
    scaler.fit(train_set)
    cols.remove('cl')
    
    windowed_frames_test = []
    label_lst_test = []
    auto_list_test = []
    windowed_frames_train = []
    label_lst_train = []
    auto_list_train = []
    conter = 0
    
    for i in test_samples:
        conter =conter+1
       
        label = i['cl']
        index = i.index
        i.drop(['cl'], inplace=True, axis='columns')
       
        colus = i[cols].columns
        scaled = scaler.transform(i[cols])
        scaled = pd.DataFrame(scaled, columns=colus, index=index)
        if len(scaled) < windowsize:
            continue
        frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=windowsize, overlapp=overlap)
        windowed_frames_test.append(frame)
        label_lst_test.append(label)
        auto_list_test.append(labe_lauto)
    
    for i in train_samples:
        label = i['cl']
        index = i.index
        i.drop(['cl'], inplace=True, axis='columns')
        colus = i[cols].columns
        scaled = scaler.transform(i[cols])
        scaled = pd.DataFrame(scaled, columns=colus, index=index)
        if len(i) < windowsize:
            continue
        frame, label, labe_lauto = make_dataframe_time_windows(scaled, label=label, window_size=windowsize, overlapp=overlap)
        windowed_frames_train.append(frame)
        label_lst_train.append(label)
        auto_list_train.append(labe_lauto)
    
    test_set = pd.concat(windowed_frames_test, axis='index')
    train_set = pd.concat(windowed_frames_train, axis='index')
    label_test = pd.concat(label_lst_test, axis='index')
    label_train = pd.concat(label_lst_train, axis='index')
    auto_train = pd.Series(auto_list_train,name='auto_train')
    auto_test = pd.Series(auto_list_test,name='auto_test')
    
    return test_set, train_set, label_test, label_train, auto_test, auto_train




import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tsne(X_train: pd.DataFrame, y_train: pd.Series, person_labels_train: pd.Series, 
              X_test: pd.DataFrame, y_test: pd.Series, person_labels_test: pd.Series, 
              name: str, path: str) -> None:
    """
    Creates t-SNE plots for the training and test datasets, colored by labels and person labels, and saves them as PDF.

    Parameters
    ----------
    X_train : pd.DataFrame
        DataFrame containing the feature data for training.
    y_train : pd.Series
        Series containing the labels for the training data.
    person_labels_train : pd.Series
        Series containing the person labels for the training data.
    X_test : pd.DataFrame
        DataFrame containing the feature data for testing.
    y_test : pd.Series
        Series containing the labels for the testing data.
    person_labels_test : pd.Series
        Series containing the person labels for the testing data.
    name : str
        Name to be used for the saved PDF files.
    path : str
        Path where the PDF files will be saved.

    Returns
    -------
    None
        This function does not return any value. It displays the t-SNE plots and saves them as PDF files.
    """
    # Apply t-SNE to the combined data
    tsne = TSNE(n_components=2, random_state=0)
    combined_data = pd.concat([X_train, X_test])
    combined_tsne = tsne.fit_transform(combined_data)

    # Split the transformed data back into train and test sets
    X_train_tsne = combined_tsne[:len(X_train)]
    X_test_tsne = combined_tsne[len(X_train):]

    # Create DataFrames for plotting
    tsne_train_df = pd.DataFrame(X_train_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_train_df['Label'] = y_train.values
    tsne_train_df['Person'] = person_labels_train.values

    tsne_test_df = pd.DataFrame(X_test_tsne, columns=['TSNE1', 'TSNE2'])
    tsne_test_df['Label'] = y_test.values
    tsne_test_df['Person'] = person_labels_test.values

    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', data=tsne_train_df)
    plt.title('t-SNE Plot for Training Data by Labels')
    plt.savefig(f"{path}/{name}_train_by_labels.pdf")
    plt.show()
    plt.clf()
    plt.close()

    # Plot t-SNE for training data by person labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Person', data=tsne_train_df)
    plt.title('t-SNE Plot for Training Data by Person Labels')
    plt.savefig(f"{path}/{name}_train_by_persons.pdf")
    plt.show()
    plt.clf()
    plt.close()

    # Plot t-SNE for test data by labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', data=tsne_test_df)
    plt.title('t-SNE Plot for Test Data by Labels')
    plt.savefig(f"{path}/{name}_test_by_labels.pdf")
    plt.show()
    plt.clf()
    plt.close()

    # Plot t-SNE for test data by person labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Person', data=tsne_test_df)
    plt.title('t-SNE Plot for Test Data by Person Labels')
    plt.savefig(f"{path}/{name}_test_by_persons.pdf")
    plt.show()
    plt.clf()
    plt.close()

    plt.close('all')


import os
import pickle
from typing import List

def combine_pickle_dicts(pickle_paths: List[str], output_dir: str, output_filename: str) -> None:
    """
    Loads pickle files containing dictionaries, combines them, and saves the resulting dictionary
    in a separate folder as a pickle file.

    Parameters
    ----------
    pickle_paths : list of str
        List of paths to the pickle files.
    output_dir : str
        Path to the directory where the combined pickle file will be saved.
    output_filename : str
        Name of the resulting pickle file.

    Raises
    ------
    ValueError
        If any of the pickle files do not contain a dictionary.

    Returns
    -------
    None
        This function does not return anything. It saves the combined dictionary as a pickle file.
    """
    combined_dict = {}

    for path in pickle_paths:
        with open(path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, dict):
                combined_dict.update(data)
            else:
                raise ValueError(f"The file {path} does not contain a dictionary.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'wb') as output_file:
        pickle.dump(combined_dict, output_file)

    print(f"The combined dictionary has been successfully saved to {output_path}.")




from typing import List
import pandas as pd

def reduce_dataframes_size(list_of_lists: List[pd.DataFrame], percentage: float) -> List[pd.DataFrame]:
    """
    Reduce the size of DataFrames within a list by a specified percentage.

    This function takes a list containing DataFrames and returns a new list,
    where each DataFrame is reduced to a specified percentage of its original size.
    The function also performs forward fill and interpolation to handle missing values,
    and drops rows with a threshold of missing values.

    Parameters
    ----------
    list_of_lists : List[pd.DataFrame]
        A list containing DataFrames.
    percentage : float
        The percentage of rows to retain in each DataFrame. If greater than 1, it is assumed to be a percentage (0-100).

    Returns
    -------
    List[pd.DataFrame]
        A new list containing the reduced DataFrames.

    Example
    -------
    >>> import pandas as pd
    >>> list_of_lists = [pd.DataFrame({'A': range(100)}), pd.DataFrame({'B': range(100)})]
    >>> reduced_dataframes = reduce_dataframes_size(list_of_lists, 50)
    >>> print(reduced_dataframes)
    """
    if percentage > 1:
        percentage = percentage / 100
    reduced_dataframes = []
    for df in list_of_lists:
        num_rows_to_select = max(1, int(len(df) * percentage))
        df = df.ffill(limit=2).interpolate('pchip', limit_direction='both')
        import random
        # Zufälligen Startindex für den Chunk wählen
        start_index = random.randint(0, len(df) - num_rows_to_select)

        # Chunk auswählen
        reduced_df = df.iloc[start_index:start_index + num_rows_to_select]

        reduced_df = reduced_df.dropna(axis='index', thresh=7)
        reduced_dataframes.append(reduced_df)
    return reduced_dataframes
