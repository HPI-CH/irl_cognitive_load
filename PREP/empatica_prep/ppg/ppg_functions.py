#!/usr/bin/env python
# coding: utf-8
"""
PPG/BVP Functions Module
=========================

This module provides functions for cleaning and feature extraction of Photoplethysmogram (PPG) or Blood Volume Pulse (BVP) signals.

Steps
-----
1. Resample accelerometer (ACC) data.
2. Apply X-RLS filter.
3. Determine thresholds.
4. Interpolate missing values using monotone cubic method from scipy and stored values.
5. Clean PPG/BVP signal:
    1. Improve signal quality.
    2. Clean using BioSPPy with cutoff frequency: [0.5, 12].
    3. Remove trend.
    4. Calculate heart rate (HR).
    5. Interpolate (optional).
6. Calculate TPP (correct HR).
7. Extract HRV/RRI features.
8. Split the data into its tasks.
9. Calculate features:
    1. Derivatives (1-3).
    2. Systolic peak, pulse onset, dicrotic notch, diastolic peak, min and max of first derivative, min and max points of second derivative.
    3. Time-domain features using BioSPPy.
    4. Frequency-domain features.

Features
--------
- Frequency-domain features
- Time-domain features (for signals of length 30 seconds)
- Non-linear features (for signals of length 90 seconds)

The resulting features in detail:
- 'PPG_Clean': Cleaned PPG signal
- 'PPG_Pulse_Peaks': Detected pulse peaks in the PPG signal
- 'hr': Heart rate
- 'cl': Confidence level
- 'PPG_Peak_Onset': Onset of PPG peaks
- 'PPG_Systolic_Peaks': Systolic peaks in the PPG signal
- 'cat_1': Category 1 (custom feature)
- 'cat_2': Category 2 (custom feature)
- 'id': Identifier for the task
- 'task': Task identifier
- 'hr_min': Minimum heart rate
- 'hr_max': Maximum heart rate
- 'hr_minmax': Range of heart rate (max - min)
- 'hr_mean': Mean heart rate
- 'hr_median': Median heart rate
- 'rr_min': Minimum RR interval
- 'rr_max': Maximum RR interval
- 'rr_minmax': Range of RR interval (max - min)
- 'rr_mean': Mean RR interval
- 'rr_median': Median RR interval
- 'rmssd': Root mean square of successive differences
- 'nn50': Number of pairs of successive RR intervals that differ by more than 50 ms
- 'pnn50': Proportion of NN50 divided by total number of RR intervals
- 'ulf_pwr': Ultra-low frequency power
- 'ulf_peak': Ultra-low frequency peak
- 'ulf_rpwr': Relative power in the ultra-low frequency range
- 'vlf_pwr': Very low frequency power
- 'vlf_peak': Very low frequency peak
- 'vlf_rpwr': Relative power in the very low frequency range
- 'lf_pwr': Low frequency power
- 'lf_peak': Low frequency peak
- 'lf_rpwr': Relative power in the low frequency range
- 'hf_pwr': High frequency power
- 'hf_peak': High frequency peak
- 'hf_rpwr': Relative power in the high frequency range
- 'vhf_pwr': Very high frequency power
- 'vhf_peak': Very high frequency peak
- 'vhf_rpwr': Relative power in the very high frequency range
- 'lf_hf': Ratio of low frequency to high frequency power
- 'lf_nu': Normalized low frequency power
- 'hf_nu': Normalized high frequency power
- 'PPG_median': Median of the PPG signal
- 'PPG_max': Maximum of the PPG signal
- 'PPG_min': Minimum of the PPG signal
- 'PPG_iqr': Interquartile range of the PPG signal
- 'PPG_slope': Slope of the PPG signal
- 'PPG_minimal_slope': Minimal slope of the PPG signal
- 'PPG_maximal_slope': Maximal slope of the PPG signal
- 'PPG_mean_slope': Mean slope of the PPG signal
- 'PPG_HR_mean': Mean heart rate derived from the PPG signal
"""


import os
import seaborn as sns
import inspect
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from numpy.lib.stride_tricks import sliding_window_view
import neurokit2 as nk
from typing import Optional, Any
import copy
import warnings
import biosppy
#import hrv_ppg_biosspy
from . import hrv_ppg_biosspy
from . import pyppg_functions
from ..utils import plotting_biosppy
from ..utils import reading_functions
import gc



BEAT_START_DERIVATIVE_THRESHOLD = 0.01
HIGH_QUALITY_THRESHOLD = 0.65
QUALITY_WINDOW_SIZE = 3.0




# helper
def delete_file(file_path: str) -> None:
    """
    Delete a specified file if it exists.

    Parameters
    ----------
    file_path : str
        The path to the file that should be deleted.

    Returns
    -------
    None

    Notes
    -----
    This function checks if the file exists at the specified path and deletes it.
    If the file does not exist, it prints a message indicating so.

    Examples
    --------
    >>> delete_file('pfad/zur/datei.txt')
    Datei pfad/zur/datei.txt wurde erfolgreich gelöscht.
    """
    if os.path.exists(file_path):
        os.remove(file_path)





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
        temp = [np.nan] * len(outlier)
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


# In[160]:




# helper
def amount_of_dirty_data(data: pd.Series) -> float:
    """
    Calculate the amount of remaining data after filtering out artifacts and outliers.

    Parameters
    ----------
    data : pd.Series
        A pandas Series where artifacts and outliers have been set to NaN.

    Returns
    -------
    float
        The proportion of remaining data that is not NaN.

    Notes
    -----
    This function calculates the proportion of data points that are not NaN 
    in the input pandas Series and prints the result.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
    >>> amount = amount_of_dirty_data(data)
    >>> print(amount)
    0.6
    """
    amount = np.sum(~np.isnan(data)) / len(data)
    print(f'{amount:.4f} Amount of remaining data')
    return amount


  
    # tested 
def improve_quality(sampling_freq: int, data: np.array) -> np.array:
        """
        Improving quality while perserving the structure of the signal

        Parameters
        ---------
        sampling_freq: int
            sampling frequency of the signal
        data: np.array
            signal which should be improved    
        
            
        Notes
        -----
        padlen = 3 * max(len(cheby4a), len(cheby4b)), in case the of insufficent length of the sample    
        
        """
        cheby4a, cheby4b = scipy.signal.cheby2(4,117,[(2*0.15)/sampling_freq], btype= 'highpass')
        if 3 * max(len(cheby4a), len(cheby4b)) > len(data): pad_len = len(data)-1; 
        else: pad_len = 3 * max(len(cheby4a), len(cheby4b));
        return scipy.signal.filtfilt(cheby4a, cheby4b,data, padlen=pad_len)
    



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
        plt.ylabel('PPG Signal in mV')
        plt.title('Processed PPG signal vs the raw signal of '+ str(name),wrap=True)
        plt.legend()
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_ppg_'+str(name)+'.pdf') 
        plt.close()   

        # split
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            fig1 = plt.figure()
            for i in processed_signal.columns:

                sns.lineplot(x=processed_signal.index, y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        
            plt.title('Processed PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_solo'+str(name)+'.pdf') 
            plt.close()   
        else: 
            fig12 = plt.figure()
            sns.lineplot(x=processed_signal.index, y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])   
            plt.title('Processed PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_sk'+str(name)+'.pdf') 
            plt.close()        
        if isinstance(raw_signal,pd. DataFrame):
            fig11 = plt.figure()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=raw_signal.index,y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close()    
        else:
            fig22 = plt.figure()
            sns.lineplot(x=raw_signal.index,y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close()



def detrend_signal(signal:np.array)-> np.array:
        """ 

        Remove linear trend along axis from data
        
        Parameters
        ----------

        data: numpy.array
            The PPG signal saved in numpy array

        Returns
        -------
        numpy.array
            the Signal with the removed linear trend along the axis

        Notes
        -----
        The signal will be added to the dataframe again it should have the right format

        """
        return scipy.signal.detrend(signal)




# 1
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


# In[173]:


# 3 # changed
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



# changed !!!
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
        missing_values.append(EDA.loc[rms > cutoff])
        missing_values.append(EDA[EDA ==np.nan])
        EDA.loc[rms > cutoff] = np.nan
        EDA = EDA.dropna()
        return EDA, missing_values


# In[175]:


# 1
def dc_removal(ppg_signal:pd.Series, beta:float=0.99)->pd.Series:
        """
        Remove the DC component from the PPG signal using a high-pass filter.

        Parameters
        ----------
        ppg_signal : pd.Series
            The raw PPG signal data.
        beta : float, optional
            The smoothing factor for the filter (default is 0.99). Note that setting beta to 1 is ineffective.

        Returns
        -------
        pd.Series
            The PPG signal with the DC component removed.

        Notes
        -----
        The gain is calculated as (1.0 + beta) / 2.0. The previous output and input values are used to compute the new signal values.
        Setting beta to 1 is ineffective because it results in no filtering effect.

        """

        gain = (1.0+beta)/2.0
        prev_output =0
        prev_input = 0
        new_gsr = []
        for i in ppg_signal.values:
            out = gain*(i-prev_input)+beta*prev_output
            new_gsr.append(out)
            prev_output=out
            prev_input=i
        return pd.Series(new_gsr, index=ppg_signal.index)


# In[176]:


#2
def rls_filter(ppg_signal:pd.Series, acceleration_data:pd.Series, delta:float=1.0, filter_order:int=4)->tuple[np.array, np.array, np.array]:
        """
        Apply Recursive Least Squares (RLS) adaptive filter to remove motion artifacts from PPG signal.

        Parameters
        ----------
        ppg_signal : pandas.Series
            The raw PPG signal data.
        acceleration_data : pandas.Series
            The acceleration data used as a reference for motion artifacts.
        delta : float, optional
            The initial value for the diagonal elements of the P matrix (default is 1.0).
        filter_order : int, optional
            The order of the filter, indicating how many previous values are considered (default is 4).

        Returns
        -------
        noise_signal : numpy.ndarray
            The estimated noise signal due to motion artifacts.
        cleaned_signal : numpy.ndarray
            The PPG signal with motion artifacts removed.
        theta : numpy.ndarray
            The final filter coefficients.

        Notes
        -----
        The filter order determines how many previous values are considered. A higher filter order increases the computational complexity.
        Based on the paper: "An implementation of motion artifacts elimination for PPG signal processing based on recursive least squares adaptive filter"
        DOI: 10.1109/BIOCAS.2017.8325141

        """



        n = len(ppg_signal)
        theta = np.zeros((filter_order, 1))  
        p = delta * np.eye(filter_order)  
        noise_signal = np.zeros(n)
        cleaned_signal = np.zeros(n)

        for t in range(filter_order, n):
          
            if t > n - filter_order - 1:
                noise_signal[t] = ppg_signal.iloc[t]
                cleaned_signal[t] = ppg_signal.iloc[t]
                continue
            if not ((t+1)-(t-filter_order+1) == filter_order): 
                noise_signal[t] = ppg_signal.iloc[t]
                cleaned_signal[t] = ppg_signal.iloc[t]
                continue
                
            phi = acceleration_data[t-filter_order+1:t+1].values.reshape(-1, 1)  #
            try:
                n_hat = np.dot(theta.T, phi)[0, 0]  # noise 
            except ValueError: 
                noise_signal[t] = ppg_signal.iloc[t]
                cleaned_signal[t] = ppg_signal.iloc[t]
                continue  
            error = ppg_signal.iloc[t] - n_hat 
            
            
            beta = np.dot(p, phi) / (1 + np.dot(phi.T, np.dot(p, phi)))
            
           
            theta = theta + beta * error
            
           
            p = p - np.dot(beta, np.dot(phi.T, p))
            
            noise_signal[t] = n_hat
            cleaned_signal[t] = error

        return noise_signal, cleaned_signal, theta








def stats_ppg(ppg:pd.Series, window:float, step:float)-> pd.Series:
        """Calculates the statistical features of the ppg including: 
            - Median
            - Mean
            - Max
            - Min
            - IQR

        Parameters
        ----------
        ppg : pd.Series
            processed ppg data
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            median_ppg, max_ppg, min_ppg, ppg_iqr
        """
        median_ppg = ppg.rolling(window=window, step=step).median().dropna()
        max_ppg = ppg.rolling(window=window, step=step).max().dropna()
        min_ppg = ppg.rolling(window=window, step=step).min().dropna()
        ppg_iqr = ppg.rolling(window=window, step=step).quantile(0.75)-ppg.rolling(window=window).quantile(0.25).dropna()

        return median_ppg, max_ppg, min_ppg, ppg_iqr



def get_slope(ppg_data:pd.Series)->pd.Series:
        """Computes the slope with the scipy's regession function

        Parameters
        ----------
        ppg_data : pd.Series
            cleaned ppg data

        Returns
        -------
        pd.Series
            slope of the ppg
        """
        x = np.arange(0,len(ppg_data), 1)
        
        y = ppg_data
        slope,intercept, r_value,p, std_err,  = scipy.stats.linregress(x,y)
        return slope

    #tested
def calculate_slope_ppg(ppg:pd.Series, window:float, step:float)->pd.Series:
        """Calculates the slope of the ppg (gradient)

        Parameters
        ----------
        ppg : pd.Series
            The processed ppg values
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            slope, minimal slope, maximal slope, mean slope
            
        """
        slope = ppg.rolling(window=window, step=step).apply(get_slope, raw=True).dropna()
        min_slope = slope.rolling(window=window, step=step).min().dropna()
        max_slope = slope.rolling(window=window, step=step).max().dropna()
        mean_slope = slope.rolling(window=window, step=step).mean().dropna()
        return slope, min_slope, max_slope, mean_slope




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
        
        plt.figure(figsize=(30, 10))
        plt.tight_layout()
        
        if isinstance(processed_signal,pd. DataFrame):
            x = [i // sampling_freq for i in processed_signal.index]
            columns_name = processed_signal.columns.values.tolist()
            for i in processed_signal.columns:

                sns.lineplot(x=x, y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        else: 
            sns.lineplot(x=x, y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])        
        if isinstance(raw_signal,pd. DataFrame):
            x = [i // sampling_freq for i in raw_signal.index]
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=x,y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
        else:
            sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
        plt.ylabel('PPG Signal in mV')
        plt.title('Processed PPG signal vs the raw signal of '+ str(name),wrap=True)
        plt.legend()
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_ppg_'+str(name)+'.pdf') 
        plt.close()   

        # split
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            fig1 = plt.figure(figsize=(30, 10))
            plt.tight_layout()
            x = [i // sampling_freq for i in processed_signal.index]
            for i in processed_signal.columns:

                sns.lineplot(x=x, y=processed_signal[i].values, label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        
            plt.title('Processed PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_solo'+str(name)+'.pdf') 
            plt.close(fig1)   
        else: 
            fig12 = plt.figure(figsize=(30, 10))
            plt.tight_layout()
            sns.lineplot(x=x, y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])   
            plt.title('Processed PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_sk'+str(name)+'.pdf') 
            plt.close(fig12)        
        if isinstance(raw_signal,pd. DataFrame):
            x = [i // sampling_freq for i in raw_signal.index]
            fig11 = plt.figure(figsize=(30, 10))
            plt.tight_layout()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=x,y=raw_signal[i].values, alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close(fig11)    
        else:
            fidf= plt.figure(figsize=(30, 10))
            plt.tight_layout()
            x = [i // sampling_freq for i in raw_signal.index]
            sns.lineplot(x=x,y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close(fidf)
        plt.close('all')



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
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close(fig11)    
        else:
            fidf= plt.figure(figsize=(50, 10))
            plt.tight_layout()
            x = [i // sampling_freq for i in raw_signal.index]
            sns.lineplot(x=x,y=raw_signal, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw PPG signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close(fidf)
        plt.close('all')



import datetime
def get_time_back(ts: np.array, orig_signal: pd.Series) -> list[datetime.datetime]:
    """
    Reconstructs timestamps based on a starting point and a list of time deltas.

    Parameters
    ----------
    ts : np.array
        An array of time deltas in seconds.
    orig_signal : pd.Series
        A Pandas Series whose index contains the starting timestamp.

    Returns
    -------
    list of datetime.datetime
        A list of reconstructed timestamps.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import datetime
    >>> ts = np.array([0, 10, 20, 30])
    >>> orig_signal = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3, freq='S'))
    >>> get_time_back(ts, orig_signal)
    [Timestamp('2023-01-01 00:00:00'), Timestamp('2023-01-01 00:00:10'), Timestamp('2023-01-01 00:00:20'), Timestamp('2023-01-01 00:00:30')]
    """
    
    start_time = orig_signal.index[0]
    reconstructed_timestamps = [start_time + pd.Timedelta(seconds=t) for t in ts]
    return reconstructed_timestamps


def plot_signal_with_peaks(indices, values, filename):
    """
    Plots a signal with specified peaks and saves the plot as a PDF.

    Parameters:
    indices (list of int): Indices of the peaks.
    values (list of int or float): Signal values.
    filename (str): Name of the PDF file to save the plot.

    The signal is plotted as a pink line, and the peaks are marked as blue crosses.
    """
    # Signal plotten
    plt.plot(values, color='pink', label='Signal')

    # Peaks plotten
    plt.scatter(indices, [values[i] for i in indices], color='blue', marker='x', label='Peaks')

    # Labels und Legende hinzufügen
    plt.xlabel('Index')
    plt.ylabel('Wert')
    plt.title('Signal mit Peaks')
    plt.legend()

    # Plot speichern als PDF
    plt.savefig(filename, format='pdf')


def ppg(ppg:pd.Series, acc:pd.DataFrame, sampling_freq:int=64, path:str='', name:str='PPG', use_xls=False)-> pd.DataFrame:
    """
    Process photoplethysmogram (PPG) and accelerometer (ACC) data to remove artifacts, filter noise, and extract heart rate variability (HRV) features.

    Parameters:
    -----------
    ppg : pd.Series
        The raw PPG signal data.
    acc : pd.DataFrame
        The accelerometer data with columns for x, y, and z axes.
    sampling_freq : int, optional
        The sampling frequency of the data in Hz (default is 64).
    path : str, optional
        The path to save the results (default is an empty string).
    name : str, optional
        The name for the output data (default is 'PPG').

    Returns:
    --------
    ppg_compact : pd.DataFrame
        DataFrame containing the raw PPG signal, cleaned PPG signal, and detected pulse peaks.
    hr_compact : pd.DataFrame
        DataFrame containing the heart rate data.
    peaks_comp : pd.DataFrame
        DataFrame containing the detected onsets and systolic peaks.

    Notes:
    ------
    This function performs the following steps:
    1. Resamples and interpolates the accelerometer data.
    2. Calculates the root mean square (RMS) of the accelerometer data.
    3. Removes artifacts from the PPG signal.
    4. Applies DC removal and RLS filtering to the PPG signal. (Optional)
    5. Improves the quality of the PPG signal.
    6. Extracts HRV features using the hrv_ppg_biosspy library.
    7. Aligns the timestamps of the processed data.
    8. Returns the processed PPG and HR data in compact DataFrames.
    """
   
    acc_down = acc.resample(str(1/sampling_freq)+'s').interpolate(method='pchip',limit_direction='both')
    acc_rms = calculate_rms(acc=acc_down)
    up= get_cut_off(acc=acc_rms)
    ppg_f, mv = remove_artifacts(ppg, up, acc_rms.rms)
    if use_xls:
        ppg_dc = dc_removal(ppg_f, beta=1)
        noise, ppg_rls, theta= rls_filter(ppg_dc, acceleration_data=acc['x'])
        ppg_rls = pd.Series(ppg_rls, index=ppg_f.index, name='BVP')
        ppg_rls = improve_quality(sampling_freq=sampling_freq, data=ppg_rls.values)
        ppg_f=pd.Series(ppg_rls, index=ppg_f.index, name='BVP')
    mv1 = pd.concat(mv, axis='index')
    mv1[:] = np.nan
    ppg_i = pd.concat([ppg_f, mv1],axis='index').sort_index()
   
    try: 
         ppg_inter = ppg_i.interpolate('pchip', limit_direction='both')
         print(ppg_inter)
    except ValueError:
         ppg_inter = ppg_i.ffill(limit=2).interpolate('pchip', limit_direction='both')  
    ts1, clean_ppg1, pulse_peaks1, template_ts1, template_hr1, hr_ts1, hr1 = hrv_ppg_biosspy.ppg(signal = ppg_inter.values.flatten(), sampling_rate=sampling_freq, units='mV', path = path, name= 'name',ftype='cheby2', cutoff=[0.5, 12.0])
    
    peaksPy, onsetsPy = pyppg_functions.get_peak_onset('PPGdet', ppg=clean_ppg1, fs=sampling_freq)
    print(peaksPy)
    print(onsetsPy)
    final_peaks = np.concatenate((pulse_peaks1, peaksPy))
    final_peaks = np.sort(final_peaks)
    final_peaks = np.unique(final_peaks)
    print('final_peaks: ', final_peaks)
    
    rpeaks = np.array(peaksPy)

    # difference of R-peaks converted to ms
    rri = np.diff(rpeaks) / sampling_freq*1000.0
    
    hr = 60/(rri/1000)
    print('Heartrate vorher',hr.mean())
    hr = pd.Series(hr.flatten(), index=[peaksPy[:len(hr)]])
    ps = pd.Series((rri/1000).flatten(), index = [peaksPy[:len(rri)]])
    j = process_peaks(hr,ps )
    j.set_index('idx', inplace=True)
    j['PPG_fixed_peaks']= j['PPG_fixed_peaks'].fillna(0)
    clean_time = get_time_back(ts1, ppg_i)
    
    clean_df = pd.DataFrame(clean_ppg1, columns=['PPG_Clean'])
    clean_df = pd.concat([clean_df, j], axis='columns')
    print('clean_df',clean_df)
    clean_df['PPG_fixed_peaks'] = clean_df['PPG_fixed_peaks'].fillna(0)
    clean_df.index=clean_time
    clean_df['PPG_Pulse_Peaks']=0
   
    clean_df['PPG_Pulse_Peaks'].iloc[pulse_peaks1] =1
    time_hr = get_time_back(hr_ts1, ppg_f)
    hr_compact = pd.DataFrame({'PPG_Hr':hr1}, index=time_hr)
    complete_ppg = pd.concat([clean_df, hr_compact], axis='columns')
    complete_ppg.sort_index(inplace=True)
    return complete_ppg, hr_compact




def generate_features_for_timewindows(signal_task: list[pd.DataFrame], timewindow: int, overlapp: Optional[int] = None, sampling_rate:int =64, method:str='nk') -> pd.DataFrame:
    
    """
    Generate features for specified time windows from signal data.

    This function calculates features for each specified time window (e.g., 30 seconds) from the provided signal data.
    The features are computed for each time window and not for the entire signal. If an overlap is specified, the 
    function will use the overlap to determine the start of the next time window.

    Parameters
    ----------
    signal_task : list[pd.DataFrame]
        List of DataFrames containing the signal data with a DatetimeIndex.
    timewindow : int
        The length of the time window in seconds for which features are to be calculated.
    overlapp : Optional[int], optional
        The overlap in seconds between consecutive time windows. If not specified, the default overlap is half of the time window length.
    sampling_rate : int, optional
        The sampling rate of the signal, by default 64 Hz.
    method : str, optional
        The method used for feature extraction, by default 'biosppy'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated features for each time window.

    Notes
    -----
    The function calculates features for each time window and includes the following:
    - PPG_Peak_Onset: Onset of PPG peaks.
    - PPG_Systolic_Peaks: Systolic peaks of PPG.
    - HRV features: Various heart rate variability features calculated using the specified method.
    - Mean values of other signal columns within the time window.
    - Category and participant information.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from copy import deepcopy
    >>> import neurokit2 as nk
    >>> signal_task = [pd.DataFrame({
    ...     'PPG_Clean': np.random.rand(1000),
    ...     'PPG_fixed_peaks': np.random.randint(0, 2, size=1000),
    ...     'cat_1': ['cat1']*1000,
    ...     'cat_2': ['cat2']*1000,
    ...     'participant': ['participant1']*1000,
    ...     'task': ['task1']*1000
    ... }, index=pd.date_range('2023-01-01', periods=1000, freq='S'))]
    >>> features = generate_features_for_timewindows(signal_task, timewindow=30, overlapp=15, sampling_rate=64, method='biosppy')
    >>> print(features)
    """

    features = []
    if not overlapp: overlapp = timewindow // 2

    for task in signal_task:
        task.sort_index(inplace=True, axis='index')
        starttime = task.index[0]
        finaltime = task.index[-1]
        temp2 =copy.deepcopy(task).reset_index()
        

        while starttime + pd.Timedelta(seconds=timewindow) <= finaltime:
            endtime = starttime + pd.Timedelta(seconds=timewindow)
            
            st =temp2[(temp2['index'] == starttime)].index.item()
            #print(temp2[(temp2['index'] == endtime)].index)
            et =temp2[(temp2['index'] == endtime)].index.item()
            temp22 = temp2.loc[st:et]
            tztz = temp22[temp22.PPG_fixed_peaks ==1].index.values
            if 'biosppy' in method:
                print(tztz)
                onsets1d, systolic_peaks1d, segement_locs1d = hrv_ppg_biosspy.ppg_segmentation(signal=task.PPG_Clean.loc[starttime:endtime].values, sampling_rate=sampling_rate,peaks=tztz)
                task['PPG_Peak_Onset']=0
                task['PPG_Peak_Onset'].iloc[onsets1d] = 1
                task['PPG_Systolic_Peaks']=0
                task['PPG_Systolic_Peaks'].iloc[systolic_peaks1d] = 1
                
                
                #hrv_features1= hrv_ppg_biosspy.hrv(rpeaks=systolic_peaks1d, sampling_rate=sampling_rate, parameters='auto', outliers='filter', detrend_rri=True, features_only=True, show=False, show_individual=False, name='')
                hrv_features1= hrv_ppg_biosspy.hrv(rpeaks=tztz, sampling_rate=sampling_rate, parameters='auto', outliers='filter', detrend_rri=True, features_only=True, show=False, show_individual=False, name='')
                l = task.loc[starttime:endtime].mean(numeric_only=True)
                lt = pd.DataFrame(l).T
                lt.index = [endtime]
                lt['cat_1'] = task.cat_1.loc[endtime]
                lt['cat_2'] = task.cat_2.loc[endtime]
                lt['id'] =  task.participant.loc[endtime]
                lt['task'] = task.task.loc[endtime]
                features_hrv = pd.DataFrame(dict(hrv_features1), index=[endtime])
                mean_feature_c = pd.concat([lt, features_hrv], axis='columns')
                mean_feature_c.index = lt.index
                features.append(mean_feature_c)
                
                
            else:
                try:
                     hrv_time = nk.hrv_time(tztz, sampling_rate = sampling_rate)
                except Exception:
                    time =[
                       
                        'HRV_RMSSD',
                        'HRV_SDNN',
                        'HRV_pNN50',
                        'HRV_MeanNN',
                        'HRV_SDSD',
                        'HRV_CVNN',
                        'HRV_pNN20',
                        'HRV_MedianNN',
                        'HRV_TINN',
                        'HRV_MadNN',
                        'HRV_HCVNN',
                        'HRV_IQRNN',
                        'HRV_HTI',
                        'HRV_MCVNN',
                        'HRV_SDRMSSD',
                        'HRV_Prc20NN',
                        'HRV_Prc80NN',
                        'HRV_MinNN',
                        'HRV_MaxNN'
                    ]
                    
                    hrv_time = pd.DataFrame(
                         columns=time, index=[0]
                    )
                try:
                    hrv_freq= nk.hrv_frequency(tztz, sampling_rate= sampling_rate)
                except Exception:
                    freq =[   'HRV_LF',
                        'HRV_HF',
                        'HRV_HFn',
                        'HRV_ULF',
                        'HRV_VLF',
                        'HRV_VHF',
                        'HRV_LFn',
                        'HRV_LnHF',
                    ]

                    hrv_freq = pd.DataFrame(
                         columns=freq, index=[0]
                    )
                lt = pd.concat([hrv_time, hrv_freq], axis='columns')    
                lt.index = [endtime]
                k = task.loc[starttime:endtime].mean(numeric_only=True)
                print(k)
                k= pd.DataFrame(k).T
                k.index = [endtime]
                lt['cat_1'] = task.cat_1.loc[endtime]
                lt['cat_2'] = task.cat_2.loc[endtime]
                lt['id'] =  task.participant.loc[endtime]
                lt['task'] = task.task.loc[endtime]
                lt=pd.concat([lt, k], axis = 'columns')
                features.append(lt)
            
            
            starttime = starttime + pd.Timedelta(seconds=overlapp)
            #endtime = endtime + pd.Timedelta(seconds=overlapp)
            
    
    return pd.concat(features, axis='index').sort_index()


 


def generate_statistical_features(ppg_lst: list[pd.DataFrame], sampling_rate: int = 64, timewindow: int = 30, overlap: Optional[int] = 15) -> pd.DataFrame:
    """
    Generate statistical features from a list of PPG DataFrames.

    Parameters
    ----------
    ppg_lst : list[pd.DataFrame]
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
    - PPG_median
    - PPG_max
    - PPG_min
    - PPG_iqr
    - PPG_slope
    - PPG_minimal_slope
    - PPG_maximal_slope
    - PPG_mean_slope
    - PPG_HR_mean
    """
    stat_features = []
    timewindow = timewindow * sampling_rate
    overlap = overlap * sampling_rate
    for task in ppg_lst:
        median_ppg, max_ppg, min_ppg, ppg_iqr = stats_ppg(task.PPG_Clean, window=timewindow, step=overlap)
        hr = pd.Series(task.PPG_Hr.rolling(window=timewindow, step=overlap).mean(), name='PPG_HR_mean')
        slope, minimal_slope, maximal_slope, mean_slope = calculate_slope_ppg(task.PPG_Clean, window=timewindow, step=overlap)
        emp_stat_frame = pd.concat([median_ppg, max_ppg, min_ppg, ppg_iqr, slope, minimal_slope, maximal_slope, mean_slope, hr], axis='columns')
        emp_stat_frame.columns = ['PPG_median', 'PPG_max', 'PPG_min', 'PPG_iqr', 'PPG_slope', 'PPG_minimal_slope', 'PPG_maximal_slope', 'PPG_mean_slope', 'PPG_HR_mean']
        stat_features.append(emp_stat_frame)
    return pd.concat(stat_features, axis='index').sort_index().dropna(how='all')



def features(ppg: list[pd.DataFrame], sampling_freq: int = 64, timewindows: int = 30, show: bool = False, overlap: Optional[int] = None):
    """
    Generate frequency-domain, time-domain, and non-linear features from PPG data.

    Parameters
    ----------
    ppg : list[pd.DataFrame]
        List of DataFrames containing PPG data.
    sampling_freq : int, optional
        Sampling frequency of the PPG data, by default 64.
    timewindows : int, optional
        Time window in seconds for calculating features, by default 30.
    show : bool, optional
        Whether to display the features, by default False.
    overlap : Optional[int], optional
        Overlap in seconds between consecutive windows, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated features.

    Notes
    -----
    The function generates the following types of features:
    - Frequency-domain features
    - Time-domain features (for signals of length 30 seconds)
    - Non-linear features (for signals of length 90 seconds)

    The resulting DataFrame includes the following columns:
    - 'PPG_Clean': Cleaned PPG signal
    - 'PPG_Pulse_Peaks': Detected pulse peaks in the PPG signal
    - 'hr': Heart rate
    - 'cl': Confidence level
    - 'PPG_Peak_Onset': Onset of PPG peaks
    - 'PPG_Systolic_Peaks': Systolic peaks in the PPG signal
    - 'cat_1': Category 1 (custom feature)
    - 'cat_2': Category 2 (custom feature)
    - 'id': Identifier for the task
    - 'task': Task identifier
    - 'PPG_median': Median of the PPG signal
    - 'PPG_max': Maximum of the PPG signal
    - 'PPG_min': Minimum of the PPG signal
    - 'PPG_iqr': Interquartile range of the PPG signal
    - 'PPG_slope': Slope of the PPG signal
    - 'PPG_minimal_slope': Minimal slope of the PPG signal
    - 'PPG_maximal_slope': Maximal slope of the PPG signal
    - 'PPG_mean_slope': Mean slope of the PPG signal
    - 'PPG_HR_mean': Mean heart rate derived from the PPG signal

    Frequency-domain HRV metrics:
    - HRV_ULF: The spectral power of ultra low frequencies (by default, .0 to .0033 Hz). Very long signals are required for this index to be extracted, otherwise, will return NaN.
    - HRV_VLF: The spectral power of very low frequencies (by default, .0033 to .04 Hz).
    - HRV_LF: The spectral power of low frequencies (by default, .04 to .15 Hz).
    - HRV_HF: The spectral power of high frequencies (by default, .15 to .4 Hz).
    - HRV_VHF: The spectral power of very high frequencies (by default, .4 to .5 Hz).
    - HRV_TP: The total spectral power.
    - HRV_LFHF: The ratio obtained by dividing the low frequency power by the high frequency power.
    - HRV_LFn: The normalized low frequency, obtained by dividing the low frequency power by the total power.
    - HRV_HFn: The normalized high frequency, obtained by dividing the high frequency power by the total power.
    - HRV_LnHF: The log transformed HF.

    Time-domain HRV metrics:
    - MeanNN: The mean of the RR intervals.
    - SDNN: The standard deviation of the RR intervals.
    - SDANN1, SDANN2, SDANN5: The standard deviation of average RR intervals extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
    - SDNNI1, SDNNI2, SDNNI5: The mean of the standard deviations of RR intervals extracted from n-minute segments of time series data (1, 2 and 5 by default). Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) and will be silently skipped if the data provided is too short.
    - RMSSD: The square root of the mean of the squared successive differences between adjacent RR intervals. It is equivalent (although on another scale) to SD1, and therefore it is redundant to report correlations with both (Ciccone, 2017).
    - SDSD: The standard deviation of the successive differences between RR intervals.
    - CVNN: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR intervals (MeanNN).
    - CVSD: The root mean square of successive differences (RMSSD) divided by the mean of the RR intervals (MeanNN).
    - MedianNN: The median of the RR intervals.
    - MadNN: The median absolute deviation of the RR intervals.
    - MCVNN: The median absolute deviation of the RR intervals (MadNN) divided by the median of the RR intervals (MedianNN).
    - IQRNN: The interquartile range (IQR) of the RR intervals.
    - SDRMSSD: SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High Frequency (LF/HF) Ratio (Sollers et al., 2007).
    - Prc20NN: The 20th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
    - Prc80NN: The 80th percentile of the RR intervals (Han, 2017; Hovsepian, 2015).
    - pNN50: The percentage of absolute differences in successive RR intervals greater than 50 ms (Bigger et al., 1988; Mietus et al., 2002).
    - pNN20: The percentage of absolute differences in successive RR intervals greater than 20 ms (Mietus et al., 2002).
    - MinNN: The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).
    - MaxNN: The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).
    - TINN: A geometrical parameter of the HRV, or more specifically, the baseline width of the RR intervals distribution obtained by triangular interpolation, where the error of least squares determines the triangle. It is an approximation of the RR interval distribution.
    - HTI: The HRV triangular index, measuring the total number of RR intervals divided by the height of the RR intervals histogram.
    """

    features_1_ppg = generate_features_for_timewindows(signal_task=ppg, timewindow=timewindows, overlapp=overlap, sampling_rate=sampling_freq)
    features_2_ppg = generate_statistical_features(ppg_lst=ppg, sampling_rate=sampling_freq, timewindow=timewindows, overlap=overlap)
    print(features_1_ppg)
    return pd.concat([features_1_ppg, features_2_ppg], axis='columns').dropna(how='all')



import pandas as pd
import numpy as np
import neurokit2 as nk

def process_peaks(hr_series, tpp_series):
    
    """
    Process raw heart rate (HR) and time-per-peak (TPP) series to identify,
    filter, and correct physiological peaks using rule-based constraints and
    NeuroKit's `signal_fixpeaks` method.

    This function combines HR and TPP values, applies a sequence of checks to
    identify valid peaks (based on physiological plausibility of inter-beat
    intervals and HR changes), filters out invalid or noisy peaks, and finally
    uses NeuroKit's peak-correction algorithm to refine the detected peak
    indices. The function then computes corrected TPP values and the
    corresponding heart rates derived from the corrected peak indices.

    Parameters
    ----------
    hr_series : pandas.Series
        A 1D time-indexed series containing raw heart rate (HR) estimates.
        Values should be in beats per minute (bpm).
    tpp_series : pandas.Series
        A 1D time-indexed series representing the time-per-peak (TPP) values,
        typically derived from photoplethysmography (PPG) or similar signals.
        Values are expected in seconds.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the corrected peak indices and recalculated
        physiological metrics with the following columns:
        
        - ``idx`` : int  
            Corrected peak locations (sample indices).
        - ``PPG_fixed_peaks`` : int  
            Indicator variable marking the corrected peak positions.
        - ``PPG_tpp_new`` : float  
            Corrected time-per-peak intervals derived from the fixed indices.
        - ``PPG_hr_new`` : float  
            Corrected heart rate values computed as ``60 / PPG_tpp_new``.

    Notes
    -----
    The rule-based filtering includes:
    - Checking whether TPP values fall within a physiologically plausible
      range (0.4–1.8 s).
    - Ensuring HR changes between consecutive samples remain within 
      allowed relative differences.
    - Using consecutive and non-consecutive peak checks based on missing
      or invalid entries.

    After rule-based filtering, the peak list is refined using:
    ``nk.signal_fixpeaks(..., method="neurokit")``

    The function was originally created by Charlotte Brandebusemeyer.  
    For methodological details, see Paper DOI: 10.1109/ICSE-SEIP66354.2025.00008

    Examples
    --------
    >>> hr_series = pd.Series([80, 82, 85, 83, 81, 79, 78])
    >>> tpp_series = pd.Series([1.2, 1.1, 1.3, 1.2, 1.1, 1.0, 1.2])
    >>> result_df = process_peaks(hr_series, tpp_series)
    >>> result_df.head()
    """


    # Create a DataFrame from the input series
    tpp_df = pd.concat([hr_series, tpp_series], axis='columns')
    tpp_df.columns = ['hr', 'tpp']
    
    bpm_peaks_within_range = []
    idx_list = []
    jump = 0
    
    for i in range(len(tpp_df)-1):
        # peaks are consecutive
        if jump == 0:
            # if the previous and current tpp is within the bpm range of 30-200 and the change between the hr is not too large --> peak
            if ((tpp_df["tpp"].iloc[i-1] < 1.8) & (tpp_df["tpp"].iloc[i-1] > 0.4)):
                if (((tpp_df["tpp"].iloc[i] < 1.8) & (tpp_df["tpp"].iloc[i] > 0.4)) & ((tpp_df["hr"].iloc[i-1] == tpp_df["hr"].iloc[i]) | ((tpp_df["hr"].iloc[i-1] < tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 0.8 <= tpp_df["hr"].iloc[i-1])) | ((tpp_df["hr"].iloc[i-1] > tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 1.2 >= tpp_df["hr"].iloc[i-1])))):
                    bpm_peaks_within_range.append(1)
                    idx_list.append(tpp_df.iloc[i].name)
                    jump = 0
                else:
                    bpm_peaks_within_range.append(0)
                    jump += 1
            else:
                if len(idx_list) == 0:
                    bpm_peaks_within_range.append(0) 
                    jump = 0
                else:
                    bpm_peaks_within_range.append(0)
                    
        # peaks are not consecutive
        else:
            if len(idx_list) == 0:
                bpm_peaks_within_range.append(0) 
                jump = 0
            else:
                # if there is only one peak, that was skipped 
                if jump == 1:
                    # if current tpp is within bpm range and the change to previous valid hr is not too large --> peak
                    last_valid_hr = tpp_df["hr"].loc[idx_list[-1] == tpp_df.index].values.item()
                    if (((tpp_df["tpp"].iloc[i] < 1.8) & (tpp_df["tpp"].iloc[i] > 0.4)) & ((last_valid_hr == tpp_df["hr"].iloc[i]) | ((last_valid_hr < tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 0.8 <= last_valid_hr)) | ((last_valid_hr > tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 1.2 >= last_valid_hr)))):
                        bpm_peaks_within_range.append(1)
                        print('should append: ',tpp_df.iloc[i].name )
                        idx_list.append(tpp_df.iloc[i].name)
                        jump = 0
                    else:
                        bpm_peaks_within_range.append(0)
                        jump += 1
                # larger jumps
                else:
                    # if current and next tpp are within bpm range and the change is no too large --> peak
                    if (((tpp_df["tpp"].iloc[i] < 1.8) & (tpp_df["tpp"].iloc[i] > 0.4)) & ((tpp_df["tpp"].iloc[i+1] < 2.0) & (tpp_df["tpp"].iloc[i+1] > 0.3)) & ((tpp_df["hr"].iloc[i+1] == tpp_df["hr"].iloc[i]) | ((tpp_df["hr"].iloc[i+1] < tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 0.8 <= tpp_df["hr"].iloc[i+1])) | ((tpp_df["hr"].iloc[i+1] > tpp_df["hr"].iloc[i]) & (tpp_df["hr"].iloc[i] * 1.2 >= tpp_df["hr"].iloc[i+1])))):
                        bpm_peaks_within_range.append(1)
                        idx_list.append(tpp_df.iloc[i].name)
                        jump = 0
                    else:
                        bpm_peaks_within_range.append(0)
                        jump += 1
    
    if len(tpp_df) > len(bpm_peaks_within_range):
        bpm_peaks_within_range.append(np.nan)
    tpp_df["filtered_peaks"] = bpm_peaks_within_range
    
    idx_list = np.array(idx_list)
    print(idx_list)
    print(tpp_df)
    info, PPG_fixed_peaks = nk.signal_fixpeaks(idx_list.flatten(), sampling_rate=64, iterative=True, show=False, interval_min=0.4, interval_max=1.8, robust=True, method="neurokit")
    
    PPG_fixed_peaks = pd.DataFrame(PPG_fixed_peaks)
    PPG_fixed_peaks["PPG_fixed_peaks"] = 1
    PPG_fixed_peaks.columns = ["idx", "PPG_fixed_peaks"]
    
    index_list = pd.DataFrame(PPG_fixed_peaks.iloc[:,0])
    
    PPG_tpp_new = ((index_list - index_list.shift(1))/64)
    PPG_tpp_new.loc[len(PPG_tpp_new)] = [np.nan]
    PPG_tpp_new = PPG_tpp_new.drop(0)
    PPG_hr_new = 60/PPG_tpp_new  # calculate hr from new tpp
    
    PPG_fixed_peaks = pd.concat([PPG_fixed_peaks, PPG_tpp_new.set_index(PPG_fixed_peaks.index), PPG_hr_new.set_index(PPG_fixed_peaks.index)], ignore_index=True, axis=1)
    PPG_fixed_peaks.columns = ["idx", "PPG_fixed_peaks", "PPG_tpp_new", "PPG_hr_new"]
    
   
    
    return PPG_fixed_peaks


# %%
