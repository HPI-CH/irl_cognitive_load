#!/usr/bin/env python
# coding: utf-8
"""
TEMP Functions Module
======================

This module provides functions for processing temperature data.

Steps
-----
1. Filter outliers using accelerometer (ACC) data.
2. Apply cutoff for unrealistic temperature values.
3. Clean the data by filtering.
4. Detect outliers using Z-score method.
5. Calculate features.

Features
--------
- TEMP_median: Median of the temperature values. Represents the middle value in the sorted temperature data.
- TEMP_max: Maximum temperature value. Indicates the highest recorded temperature.
- TEMP_min: Minimum temperature value. Indicates the lowest recorded temperature.
- TEMP_iqr: Interquartile range of the temperature values. Measures the spread of the middle 50% of the data.
- TEMP_slope: Slope of the temperature values. Represents the rate of change in temperature over time.
- TEMP_minimal_slope: Minimal slope of the temperature values. Indicates the smallest rate of change in temperature.
- TEMP_maximal_slope: Maximal slope of the temperature values. Indicates the largest rate of change in temperature.
- TEMP_mean_slope: Mean slope of the temperature values. Represents the average rate of change in temperature.
- TEMP_mean: Mean of the temperature values. Represents the average temperature.
"""

#imports
import inspect
import pickle
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Any
import copy
import pandas as pd
import os
import numpy as np
import datetime
import pathlib
import neurokit2 as nk
from pandas import DataFrame
import warnings





# In[16]:


def calculate_cutoff(temp:pd.Series)->tuple[pd.Series, pd.Series]:
    """Cuttofff the unrealistic data values of the tempuature and save the index of these datapoints

    Parameters
    ----------
    temp : pd.Series
        temperature

    Returns
    -------
    pd.Series, pd.Index
        filtered temperature, index of the wrong values
    """
    no_values = temp[(temp <= 25) & (temp >= 40)]
    no_values[:] = np.nan
   
    return temp[(temp >= 25) & (temp <= 40)], no_values
    
    


# In[17]:


def calculate_mst(t_values:list|pd.Series, k_values:list)->list:
    """
    Calculate the Mean Skin Temperature (MST) using the given formula:
    MST = sum(t_i * k_i)
    
    Parameters
    ----------
    t_values : list
        List of local skin temperatures (t_i).
    k_values : list
        List of corresponding weighting factors (k_i).
    
    Returns
    -------
    float
        Calculated Mean Skin Temperature (MST).
    
    Raises
    ------
    ValueError
        If the length of t_values and k_values is not the same.

    Notes
    -----
    https://www.sciencedirect.com/science/article/pii/S0360132310002647#sec2    

    """
    if len(t_values) != len(k_values):
        raise ValueError("The length of t_values and k_values must be the same.")
    
    mst = sum(t * k for t, k in zip(t_values, k_values))
    return mst


# In[18]:


from scipy.stats import linregress

def calculate_skin_temperature(temp:pd.Series):
    """
    Calculate the skin temperature parameters TEMP_mean and TEMP_rise.
    
    Parameters
    ----------
    signal : list or np.array
        The measured temperature signal.
    
    Returns
    -------
    dict
        A dictionary containing TEMP_mean and TEMP_rise.

    Notes
    -----    
    https://link.springer.com/chapter/10.1007/978-3-540-75696-5_10    
    """
    time = temp.index
    temp =temp.values
    
   
    TEMP_mean = np.mean(temp)
    
    
    x = np.arange(len(temp))
    slope, intercept, r_value, p_value, std_err = linregress(x, temp)
    TEMP_rise = slope
    
    return {"TEMP_mean": TEMP_mean, "TEMP_rise": TEMP_rise, "Temp_time": time}


# In[19]:


def robust_z_score(acc: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate robust z-scores for each column in the DataFrame using the median and MAD.

        Parameters
        ----------
        acc : DataFrame
            The input DataFrame with accelerometer data.

        Returns
        -------
        DataFrame
            The DataFrame with additional columns for z-scores.
        """
        for cols in acc.columns:
            median = np.median(acc[cols])
            mad = np.median(np.abs(acc[cols] - median))
            acc[cols + '_zscore'] = (acc[cols] - median) / mad
        return acc


# In[20]:


def calculate_rms(acc: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the root mean square (RMS) value for z-score columns in the DataFrame.

        Parameters
        ----------
        acc : DataFrame
            The input DataFrame with z-score columns.

        Returns
        -------
        DataFrame
            The DataFrame with an additional column for RMS values.
        """
        cols = np.zeros(len(acc))
        for col in acc.columns:
                cols = cols + acc[col]**2
        rms = np.sqrt(cols / 3)
        acc['rms'] = rms
        return acc


# 3 
def get_cut_off(acc: pd.DataFrame, col: str = 'rms') -> float:
        """
        Calculate the cutoff values (mean ± standard deviation) for a specified column.

        Parameters
        ----------
        acc : DataFrame
            The input DataFrame.
        col : str, optional
            The column name for which to calculate the cutoff values. Default is 'rms'.

        Returns
        -------
        float
            The upper and lower cutoff values.
        """
        mean = acc[col].mean()
        sd = acc[col].std()
        return mean + sd

# 4

def remove_artifacts(EDA: pd.Series, cutoff: float, rms:pd.Series) -> tuple:
        """
        Remove artifacts from a EDA signal by setting values outside the specified range to NaN.

        Parameters
        ----------
        EDA : Series
            The input EDA signal.
        lower : float
            The lower cutoff value.
        higher : float
            The upper cutoff value.

        Returns
        -------
        tuple
            The cleaned EDA signal and the indices of the removed values.
        """
        missing_values = []
        EDA.loc[rms > cutoff] = np.nan
        missing_values.append(EDA[EDA.TEMP ==np.nan])
        EDA = EDA.dropna(axis='index')
        return EDA, missing_values




# In[21]:


def outlier_zscores(temp:pd.Series, n:int)-> pd.Series:
    z_scores = (temp - temp.mean()) / temp.std()
    no_values = temp[(z_scores < -3) & (z_scores > 3)].index
    temp = temp[(z_scores > -3) & (z_scores < 3)]
    return temp, no_values


# In[22]:


def smoothing_and_remove_noise(temp: pd.Series, window_length: int = 11, polyorder: int = 2, 
                               mode: str = 'interp', delta: float = 1.0, cval: float = 0.0) -> pd.Series:
    """
    Smooth and remove noise from skin temperature data using the Savitzky-Golay filter.
    
    Parameters
    ----------
    temp : pd.Series
        The raw skin temperature data.
    window_length : int, optional
        The length of the filter window (i.e., the number of coefficients). Must be a positive odd integer. Default is 11.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less than `window_length`. Default is 2.
    mode : str, optional
        The mode parameter determines how the signal is extended at the boundaries. Default is 'interp'.
    delta : float, optional
        The spacing of the samples to which the filter is applied. Default is 1.0.
    cval : float, optional
        Value to fill past the edges of the input if `mode` is 'constant'. Default is 0.0.
    
    Returns
    -------
    pd.Series
        The smoothed skin temperature data.
    
    Notes
    -----
    The Savitzky-Golay filter is used to smooth the data and remove noise while preserving the shape of the signal.
    For more information, see: https://www.tandfonline.com/doi/full/10.1080/23744731.2016.1124714?neTEMPccess=true
    """
    smoothed_temp = sc.signal.savgol_filter(temp.values.T,window_length, polyorder, deriv=0, delta=delta, axis=-1, mode=mode, cval=cval)
    return pd.Series(smoothed_temp, index=temp.index)


# In[23]:


def outlier_zscores(temp: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Identify and remove outliers from a Pandas Series using Z-scores.
    
    Parameters
    ----------
    temp : pd.Series
        The raw skin temperature data.
    threshold : float, optional
        The Z-score threshold to identify outliers. Default is 3.0.
    
    Returns
    -------
    pd.Series
        The cleaned skin temperature data with outliers removed.
    pd.Index
        The indices of the removed outliers.
    
    Notes
    -----
    Outliers are defined as data points with Z-scores greater than the specified threshold.
    """
    z_scores = (temp - temp.mean()) / temp.std()
    outlier_indices = temp[(z_scores < -threshold) | (z_scores > threshold)]
    outlier_indices[:] = np.nan
    cleaned_temp = temp[(z_scores >= -threshold) & (z_scores <= threshold)]
    return cleaned_temp, outlier_indices


# In[24]:


def plot_before_and_after(raw_signal: pd.Series|pd.DataFrame, processed_signal:pd.Series|pd.DataFrame, name:str, path:str, colors:list[str]=['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E'], call_plot = True )->None:
        """Plot the signal before the processing and after

        Parameters
        ----------
        raw_signal : pd.Series | pd.DataFrame
            the raw signal
        processed_signal : pd.Series | pd.DataFrame
            the cleansed signal
        name : str
            name of the task and the name of the participant
        path : str
            where to save the image to
        colors : list[str], optional
            colors for plotting, by default ['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E']
        """
        if not call_plot: return
        plt.figure(figsize=(10, 6))
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            for i in processed_signal.columns:

                sns.lineplot(x=processed_signal.index, y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        else: 
            sns.lineplot(x=processed_signal.index, y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])        
        if isinstance(raw_signal,pd. DataFrame):
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=raw_signal.index,y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
        else:
            sns.lineplot(x=raw_signal.index,y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
        plt.ylabel('TEMP Signal in mV')
        plt.title('Processed TEMP signal vs the raw signal of '+ str(name),wrap=True)
        plt.legend()
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_TEMP_'+str(name)+'.pdf') 
        plt.close()   

        # split
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            fig1 = plt.figure()
            for i in processed_signal.columns:

                sns.lineplot(x=processed_signal.index, y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        
            plt.title('Processed Temp signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_solo'+str(name)+'.pdf') 
            plt.close()   
        else: 
            fig12 = plt.figure()
            sns.lineplot(x=processed_signal.index, y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])   
            plt.title('Processed TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_sk'+str(name)+'.pdf') 
            plt.close()        
        if isinstance(raw_signal,pd. DataFrame):
            fig11 = plt.figure()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=raw_signal.index,y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('Raw TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close()    
        else:
            fig22 = plt.figure()
            sns.lineplot(x=raw_signal.index,y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close()


# In[26]:


def fast_plot_before_and_after(raw_signal: pd.Series|pd.DataFrame, processed_signal:pd.Series|pd.DataFrame,sampling_freq:int, name:str, path:str, colors:list[str]=['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E'], call_plot = True )->None:
        """Plot the signal before the processing and after

        Parameters
        ----------
        raw_signal : pd.Series | pd.DataFrame
            the raw signal
        processed_signal : pd.Series | pd.DataFrame
            the cleansed signal
        name : str
            name of the task and the name of the participant
        path : str
            where to save the image to
        colors : list[str], optional
            colors for plotting, by default ['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E']
        """
        if not call_plot: return
        
        plt.figure(figsize=(10, 6))
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            for i in processed_signal.columns:

                sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        else: 
            sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])        
        if isinstance(raw_signal,pd. DataFrame):
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
        else:
            sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
        
        plt.title('Processed TEMP signal vs the raw signal of '+ str(name),wrap=True)
        plt.legend()
        plt.ylabel('Temperature in °C')
        plt.xlabel('Time in seconds')
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_TEMP_'+str(name)+'.pdf') 
        plt.close()   

        # split
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            fig1 = plt.figure()
            for i in processed_signal.columns:

                sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        
            plt.title('Processed TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'processed_signal_solo'+str(name)+'.pdf') 
            plt.close()   
        else: 
            fig12 = plt.figure()
            sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])   
            plt.title('Processed TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'processed_signal_sk'+str(name)+'.pdf') 
            plt.close()        
        if isinstance(raw_signal,pd. DataFrame):
            fig11 = plt.figure()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('Raw TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close()    
        else:
            fig22 = plt.figure()
            sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close()


# In[27]:


def fast_plot(raw_signal: pd.Series|pd.DataFrame,sampling_freq:int, name:str, path:str, colors:list[str]=['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E'], call_plot = True )->None:
        """Plot the signal before the processing and after

        Parameters
        ----------
        raw_signal : pd.Series | pd.DataFrame
            the raw signal
        processed_signal : pd.Series | pd.DataFrame
            the cleansed signal
        name : str
            name of the task and the name of the participant
        path : str
            where to save the image to
        colors : list[str], optional
            colors for plotting, by default ['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E']
        """
        if not call_plot: return

        # split       
        if isinstance(raw_signal,pd. DataFrame):
            x = [i // sampling_freq for i in raw_signal.index]
            fig11 = plt.figure(figsize=(50, 10))
            plt.tight_layout()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=x,y=raw_signal[i].values, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('TEMP signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close(fig11)    
        else:
            fidf= plt.figure(figsize=(50, 10))
            plt.tight_layout()
            x = [i // sampling_freq for i in raw_signal.index]
            sns.lineplot(x=x,y=raw_signal, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('TEMP signal '+ str(name),wrap=True)
            plt.ylabel('Temperature in °C')
            plt.xlabel('Time in seconds')
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close(fidf)
        plt.close('all')


# In[28]:


from scipy.interpolate import PchipInterpolator
def interpolate_nan_c(data: pd.Series) -> pd.Series:
    """
    Interpolates NaN values in a series using monotonic cubic interpolation (PCHIP).

    Parameters
    ----------
    data : pd.Series
        Input series containing NaN values to be interpolated.
    outlier : pd.Series
        Series containing outlier values to be set as NaN.

    Returns
    -------
    pd.Series
        Series with NaN values interpolated.

    Examples
    --------
    >>> data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8])
    >>> outlier = pd.Series([np.nan], index=[2.5])
    >>> interpolate_nan(data, outlier)
    0    1.0
    1    2.0
    2    3.0
    2.5  3.5
    3    4.0
    4    5.0
    5    6.0
    6    7.0
    7    8.0
    dtype: float64
    """
    
    data.sort_index(inplace=True)
    data_index = data.index

   
    valid = ~np.isnan(data.values)
    x_valid = np.where(valid)[0]
    y_valid = data.values[valid]

    
    interpolator = PchipInterpolator(x_valid, y_valid)

    
    data_interpolated = interpolator(np.arange(len(data)))
    data_interpolated = pd.Series(data_interpolated, name=data.name, index=data_index)

    return data_interpolated 


# In[29]:


from scipy.interpolate import PchipInterpolator
def interpolate_nan(data: pd.Series, outlier: pd.Series) -> pd.Series:
    """
    Interpolates NaN values in a series using monotonic cubic interpolation (PCHIP).

    Parameters
    ----------
    data : pd.Series
        Input series containing NaN values to be interpolated.
    outlier : pd.Series
        Series containing outlier values to be set as NaN.

    Returns
    -------
    pd.Series
        Series with NaN values interpolated.

    Examples
    --------
    >>> data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8])
    >>> outlier = pd.Series([np.nan], index=[2.5])
    >>> interpolate_nan(data, outlier)
    0    1.0
    1    2.0
    2    3.0
    2.5  3.5
    3    4.0
    4    5.0
    5    6.0
    6    7.0
    7    8.0
    dtype: float64
    """
    # Set outlier values to NaN
    if isinstance(outlier, list):
        temp = [pd.nan] * len(outlier)
        outlier = pd.Series(temp, index=outlier.index)

    else : outlier.values[:] = np.nan
    name = data.name
    data = pd.concat([outlier, data])
    data.sort_index(inplace=True)
    data_index = data.index

   
    valid = ~np.isnan(data.values)
    x_valid = np.where(valid)[0]
    y_valid = data.values[valid]


    interpolator = PchipInterpolator(x_valid, y_valid)
    data_interpolated = interpolator(np.arange(len(data)))
    data_interpolated = pd.Series(data_interpolated, name=name, index=data_index)

    return data_interpolated  


# In[54]:


# add this one to the other ones
def get_label_and_task(path: str, signal: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Process the CL Excel file and categorize tasks based on timestamps.

    This function reads the CL Excel file, processes the categories of tasks, and splits the tasks based on their 
    timestamps. The function also includes cognitive load information, labels, and categories.

    Parameters
    ----------
    path : str
        The path to the directory containing the CL Excel file.
    signal : pd.DataFrame
        DataFrame containing the signal data with a DatetimeIndex.

    Returns
    -------
    Tuple[pd.DataFrame, List[pd.DataFrame]]
        A tuple containing:
        - A DataFrame with the processed tasks, including their respective categories, labels, and cognitive load.
        - A list of DataFrames, each representing a task with its respective signal data.
    Notes
    -----
    Based on co-author's code for reading the CL.xlsx file.    
    """
    cl_questionnaire = pd.read_excel(path + "/cl.xlsx")
    
    
    cl_questionnaire = cl_questionnaire[cl_questionnaire["start time"].notna()]
    

    
    time = signal.index[0]
    cl_questionnaire["physio_start_time"] = time

  
    cl_questionnaire['start time'] = pd.to_datetime(cl_questionnaire['start time'].astype(str), format="%H:%M:%S")    
    cl_questionnaire['end time'] = pd.to_datetime(cl_questionnaire['end time'].astype(str), format="%H:%M:%S")    
    cl_questionnaire["physio_start_time"] = pd.to_datetime(cl_questionnaire['physio_start_time'].astype(str))
    import pytz
    timezones = pytz.timezone('US/Pacific')
    # Füge die neue Datetime-Spalte hinzu
    cl_questionnaire['start_time'] = cl_questionnaire.apply(lambda row: timezones.localize(datetime.datetime(
        year=row['physio_start_time'].year,
        month=row['physio_start_time'].month,
        day=row['physio_start_time'].day,
        hour=row['start time'].hour,
        minute=row['start time'].minute,
        second=0,  
    )), axis='columns')
    cl_questionnaire['end_time'] = cl_questionnaire.apply(lambda row: timezones.localize(datetime.datetime(
        year=row['physio_start_time'].year,
        month=row['physio_start_time'].month,
        day=row['physio_start_time'].day,
        hour=row['end time'].hour,
        minute=row['end time'].minute,
        second=0,  
    )), axis='columns')
    cl_questionnaire["time_between_start-cl"] = (cl_questionnaire["start_time"] - cl_questionnaire["physio_start_time"]).dt.total_seconds()*32  
    cl_questionnaire["time_between_start-cl"] = cl_questionnaire["time_between_start-cl"].astype(int)  
    cl_questionnaire["time_between_end-cl"] = (cl_questionnaire["end_time"] - cl_questionnaire["physio_start_time"]).dt.total_seconds()*32
    cl_questionnaire["time_between_end-cl"] = cl_questionnaire["time_between_end-cl"].astype(int)
        
    # This part will but the IDs and the labels to the signal data
    reduced = cl_questionnaire[['participant ID', 'start_time', 'end_time', 'task', 'cognitive load', 'cat_1', 'cat_2']]
    tasks_test = []
    for row in range(len(reduced)):
        temp_task = signal.iloc[(signal.index>reduced.start_time.loc[row])&(signal.index<reduced.end_time.loc[row])]
        temp_task['task'] = reduced.task[row]
        temp_task['cat_1'] =reduced.cat_1[row]
        temp_task['cat_2'] =reduced.cat_2[row]
        temp_task['cl'] = reduced['cognitive load'].loc[row]
        temp_task['participant'] = reduced['participant ID'][row]
        tasks_test.append(temp_task)
    complete_df = pd.concat(tasks_test, axis='index')
    return complete_df, tasks_test
        


# In[78]:


def stats_temp(temp:pd.Series, window:float, step:float)-> pd.Series:
        """Calculates the statistical features of the temp including: 
            - Median
            - Mean
            - Max
            - Min
            - IQR

        Parameters
        ----------
        temp : pd.Series
            processed temp data
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            median_temp, max_temp, min_temp, temp_iqr
        """
        mean_temp = temp.rolling(window=window, step=step).mean()
        median_temp = temp.rolling(window=window, step=step).median()
        max_temp = temp.rolling(window=window, step=step).max()
        min_temp = temp.rolling(window=window, step=step).min()
        temp_iqr = temp.rolling(window=window, step=step).quantile(0.75)-temp.rolling(window=window).quantile(0.25)

        return median_temp, max_temp, min_temp, temp_iqr, mean_temp



def get_slope(temp_data:pd.Series)->pd.Series:
        """Computes the slope with the scipy's regession function

        Parameters
        ----------
        temp_data : pd.Series
            cleaned temp data

        Returns
        -------
        pd.Series
            slope of the temp
        """
        x = np.arange(0,len(temp_data), 1)
        
        y = temp_data
        slope,intercept, r_value,p, std_err,  = sc.stats.linregress(x,y)
        return slope

    #tested
def calculate_slope_temp(temp:pd.Series, window:float, step:float)->pd.Series:
        """Calculates the slope of the temp (gradient)

        Parameters
        ----------
        temp : pd.Series
            The processed temp values
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            slope, minimal slope, maximal slope, mean slope
            
        """
        slope = temp.rolling(window=window, step=step).apply(get_slope, raw=True)
        min_slope = slope.rolling(window=window, step=step).min()
        max_slope = slope.rolling(window=window, step=step).max()
        mean_slope = slope.rolling(window=window, step=step).mean()
        return slope, min_slope, max_slope, mean_slope


# In[79]:


def generate_statistical_features(temp_lst: list[pd.DataFrame], sampling_rate: int = 4, timewindow: int = 30, overlap: Optional[int] = 15) -> pd.DataFrame:
    """
    Generate statistical features from a list of PPG DataFrames.

    Parameters
    ----------
    temp_lst : list[pd.DataFrame]
        List of DataFrames containing PPG data.
    sampling_rate : int, optional
        Sampling rate of the PPG data, by default 64.
    timewindow : int, optional
        Time window in seconds for calculating features, by default 30.
    overlap : Optional[int], optional
        Overlap in seconds between consecutive windows, by default 15.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated statistical features.

    Notes
    -----
    The function calculates the following features for each PPG DataFrame:
    - TEMP_median
    - TEMP_max
    - TEMP_min
    - TEMP_iqr
    - TEMP_slope
    - TEMP_minimal_slope
    - TEMP_maximal_slope
    - TEMP_mean_slope
    - TEMP_mean
    """
    stat_features = []
    timewindow = timewindow * sampling_rate
    overlap = overlap * sampling_rate
    for task in temp_lst:
        median_temp, max_temp, min_temp, temp_iqr,mean_temp = stats_temp(task.TEMP, window=timewindow, step=overlap)
        slope, minimal_slope, maximal_slope, mean_slope = calculate_slope_temp(task.TEMP, window=timewindow, step=overlap)
        emp_stat_frame = pd.concat([median_temp, max_temp, min_temp, temp_iqr, slope, minimal_slope, maximal_slope, mean_slope, mean_temp], axis='columns').dropna(how='all')
        emp_stat_frame.columns = ['TEMP_median', 'TEMP_max', 'TEMP_min', 'TEMP_iqr', 'TEMP_slope', 'TEMP_minimal_slope', 'TEMP_maximal_slope', 'TEMP_mean_slope', 'TEMP_mean']
        labels= task.loc[emp_stat_frame.index][['task', 'cat_1', 'cat_2', 'participant']]
        emp_stat_frame = pd.concat([emp_stat_frame, labels], axis='columns')
        stat_features.append(emp_stat_frame)
    return pd.concat(stat_features, axis='index').sort_index()#.dropna(how='all')



def temp(temp: pd.Series, acc: pd.DataFrame, sampling_freq: int = 64, path: str = '', name: str = 'TEMP') -> pd.DataFrame:
    """
    Process temperature data by filtering outliers, applying cutoff, cleaning, and calculating features.

    Parameters
    ----------
    temp : pd.Series
        Raw temperature data.
    acc : pd.DataFrame
        Accelerometer data for filtering outliers.
    sampling_freq : int, optional
        Sampling frequency in Hz, by default 64.
    path : str, optional
        Path to save the processed data, by default ''.
    name : str, optional
        Name for the saved data, by default 'TEMP'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed temperature data.

    Notes
    -----
    The function performs the following steps:
    1. Resample and interpolate the accelerometer data using cubic spline method.
    2. Calculate the root mean square (RMS) of the accelerometer data.
    3. Determine cutoff values based on RMS of accelerometer data.
    4. Remove artifacts from the temperature data using the cutoff values.
    5. Concatenate the filtered temperature data and artifacts, then sort by index.
    6. Apply cutoff for unrealistic temperature values and concatenate with artifacts.
    7. Interpolate missing values in the temperature data using custom interpolation method.
    8. Smooth and remove noise from the interpolated temperature data.
    9. Detect outliers using Z-score method.
    10. If outliers are detected, concatenate the outlier data and artifacts, then interpolate using PCHIP method.
    11. Return the processed temperature data.

    Examples
    --------
    >>> import pandas as pd
    >>> temp_data = pd.Series([36.5, 36.7, 36.8, 37.0, 36.6])
    >>> acc_data = pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4, 0.5], 'y': [0.1, 0.2, 0.3, 0.4, 0.5], 'z': [0.1, 0.2, 0.3, 0.4, 0.5]})
    >>> processed_temp = temp(temp_data, acc_data)
    >>> print(processed_temp)
    """
   
    acc_down = acc.resample(str(1/sampling_freq)+'s').interpolate(method='cubicspline', limit_direction='both')
    acc_rms = calculate_rms(acc=acc_down)
    up = get_cut_off(acc=acc_rms)
    temp_f, mv_ca1 = remove_artifacts(temp, up, acc_rms.rms)
    temp_cut_off, mv2 = calculate_cutoff(temp_f)
    mv_ca1= pd.concat(mv_ca1, axis='index'); mv_ca1[:]=np.nan
    mv_ca1 = mv_ca1.loc[~mv_ca1.index.duplicated()]
    temp_inter = pd.concat([temp_cut_off, mv2, mv_ca1]).sort_index()
    temp_inter = temp_inter.loc[~temp_inter.index.duplicated()]
    temp_inter_i = interpolate_nan_c(temp_inter.TEMP)
    temp_filter = smoothing_and_remove_noise(temp=temp_inter_i)
    temp_outlier, mv3 = outlier_zscores(temp_filter)
    if len(mv3) > 0:
        mv3[:] = np.nan
        temp_outlier_con = pd.concat([temp_outlier, mv3])
        temp_outlier_con = temp_outlier_con.loc[~temp_outlier_con.index.duplicated()].sort_index()
        try:
            temp_outlier_coni = temp_outlier_con.resample(str(1/4)+'s').interpolate('pchip', limit_direction = 'both')
        except (ValueError,AttributeError, TypeError) as e:
             try: 
                  temp_outlier_coni=temp_outlier_con.resample(str(1/4)+'s').ffill(limit=2).interpolate('cubicspline', order=3, limit_direction = 'both')
             except (ValueError,AttributeError, TypeError) as e:   
                  temp_outlier_coni=temp_outlier_con.resample(str(1/4)+'s').ffill(limit=2).interpolate('linear')

        
       
        return temp_outlier_coni
    else:
        temp_filter = temp_filter.resample(str(1/4)+'s').interpolate('pchip', limit_direction = 'both')
        return temp_filter
