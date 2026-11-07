#!/usr/bin/env python
# coding: utf-8
"""
EDA Functions Module
====================

This module provides functions for cleaning and feature extraction of Electrodermal Activity (EDA) signals.

Steps
-----
0. Remove/save values below 0, as EDA cannot be negative.
1. Filter EDA signal using accelerometer (ACC) data.
2. Experiment with different filter options.
3. Clean the signal using a 3D Hamming filter.
4. Clean the signal using custom methods:
    1. Sanitize signal using NeuroKit.
    2. Clean EDA signal using NeuroKit with BioSPPy version.
    3. Remove EDA values that deviate more than 3*MAD (Median Absolute Deviation) and replace with median (consider interpolation).
    4. Interpolate missing values using monotone cubic method.
5. Decompose the signal using sparseEDA.
6. Calculate EDA peaks.
7. Generate EDA epochs based on tasks (windows).
8. Generate intervals (windows/features).
9. Calculate features such as scl_mean, rise_time, recovery_time, accumulative_eda, magnitude, skin conductance rate.

Features
--------
- amp: Amplitude of the EDA signal.
- sd_tonic: Standard deviation of the tonic component.
- symp: Sympathetic activity measure.
- sympN: Normalized sympathetic activity measure.
- skin conductance rate: Rate of change in skin conductance
    
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
import pywt





def plot_before_and_after(raw_signal: pd.Series|pd.DataFrame, processed_signal:pd.Series|pd.DataFrame, name:str, path:str, colors:list[str]=['#E6BFCC' ,'#E98DAD' ,'#C74777', '#7A042A', '#4F021B','#663845','#8B4E5E'], call_plot=False)->None:
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
        if not call_plot:return
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

        plt.title('Processed EDA signal vs the raw signal of '+ str(name),wrap=True)
        plt.ylabel('EDA signals in µS')
        plt.legend()
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_eda_'+str(name)+'.pdf')
        plt.close()


# In[8]:


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
        plt.ylabel('EDA Signal in mV')
        plt.title('Processed EDA signal vs the raw signal of '+ str(name),wrap=True)
        plt.legend()
        plt.savefig(path+'/'+'processed_signal_vs_raw_signal_EDA_'+str(name)+'.pdf') 
        plt.close()   

        # split
        if isinstance(processed_signal,pd. DataFrame):
            columns_name = processed_signal.columns.values.tolist()
            fig1 = plt.figure()
            for i in processed_signal.columns:

                sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal[i], label = 'Processed signal '+i, color=colors[columns_name.index(i)])
        
            plt.title('Processed EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_solo'+str(name)+'.pdf') 
            plt.close()   
        else: 
            fig12 = plt.figure()
            sns.lineplot(x=[i // sampling_freq for i in range(len(processed_signal))], y=processed_signal, label = 'Processed signal '+processed_signal.name, color=colors[2])   
            plt.title('Processed EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'processed_signal_sk'+str(name)+'.pdf') 
            plt.close()        
        if isinstance(raw_signal,pd. DataFrame):
            fig11 = plt.figure()
            columns_name = raw_signal.columns.values.tolist()
            for i in raw_signal.columns:
                sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal[i], alpha= 0.3, label='Raw signal '+i,color=colors[columns_name.index(i)])   
            plt.title('Raw EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close()    
        else:
            fig22 = plt.figure()
            sns.lineplot(x=[i // sampling_freq for i in range(len(raw_signal))],y=raw_signal, alpha= 0.3, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('Raw EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close()


# In[9]:


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
                sns.lineplot(x=x,y=raw_signal[i].values, label=i,color=colors[columns_name.index(i)])   
            plt.title('EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('EDA in µS')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'raw_signal_solo'+str(name)+'.pdf') 
            plt.close(fig11)    
        else:
            fidf= plt.figure(figsize=(50, 10))
            plt.tight_layout()
            x = [i // sampling_freq for i in raw_signal.index]
            sns.lineplot(x=x,y=raw_signal, label='Raw signal '+raw_signal.name, color = colors[1])
            plt.title('EDA signal '+ str(name),wrap=True)
            plt.legend()
            plt.ylabel('EDA in µS')
            plt.xlabel('Time in seconds')
            plt.savefig(path+'/'+'raw_signal_sk'+str(name)+'.pdf') 
            plt.close(fidf)
        plt.close('all')


# In[10]:


def estimate_scale_parameter(x:np.array)-> float:
        """The function will estimate the scale parameter b_hat, to do the laplace fitting of the coefficient of the wavelet coeffiecents. This parameter will later be 
            estimated for every level j from the wavelet coefficients of the original EDA signal using the Maximum-Likelihood estimation:
            - b_hat = 1/(Number of coefficients)*sum(|x_i -µ_hat|) -> µ_hat of a wavelet distrubution is zero, therefore
            - b_hat = 1/(Number of coefficients)*sum(|x_i|), x_i is the ith coefficient
            The function is part of the motion artefact removal of the EDA.

        Parameters
        ----------
        x : pd.Series | np.array
            wavelet coefficients, output of the wavelet function

        Returns
        -------
        float
            scale parameter b_hat
        
        Notes
        -----
        This is the implementation of the paper:  Efficient wavelet-based artifact removal for electrodermal activity in
        real-world applications with the DOI: https://doi.org/10.1016/j.bspc.2018.01.009    
        """
        n = len(x)
        if isinstance(x, pd.Series): b_hat = 1/n*(np.sum(np.absolute(x.values)))
        else: b_hat = 1/n*(np.sum(np.absolute(x)))
        return b_hat




def generate_threshold(b_hat:float, i:float)->tuple[float, float]:
        """Generates the wavelet coefficient shrinkage threshold with the b_hat and a value i which is the estimated portion of noise from the signal. Set i to low value and than adapt it.
            Calculations:
                - upper threshold = b_hat * ln (i)
                - lower threshold = - upper threshold

        Parameters
        ----------
        b_hat : float
            scalar parameter calculated from the wavelet coefficients 
        i : float
            estimated noise of the signal

        Returns
        -------
        tuple[float, float]
            upper threshold, lower threshold
        """
        upper  = b_hat*np.log(i)
        lower = -upper
        return upper, lower



def apply_treshold_wavelet(coeff:np.array, upper:float, lower:float)->np.array:
        """Applies the upper and lower threshold to the wavelet coefficients which will be later used to retransform the signal with inverse wavelet.
            - is the filtering step
            - cuttoff of alle coefficients which are higher or lower than the threshold.

        Parameters
        ----------
        coeff : np.array
            coefficients of the wavelet transformation
        upper : float
            upper threshold
        lower : float
            lower threshold

        Returns
        -------
        np.array
            filtered coefficients of the wavelet
            
        """
        return np.where((coeff >= lower)&(coeff <= upper),coeff,0)



# verändert # XXX
def denoise_signal(eda:pd.Series, i:float, level:int, wavelet:str='haar')->np.array:
        """This function serves partly as a wrapper function. The function will denoise the EDA signal using the stationary wavelet function (swt).
                - transforms the EDA signal using the wavelet transformation, default haar
                - estimates for each level of the coefficients the scale parameter b_hat 
                - calculates upper and lower threshold based on b_hat and a scaler i, which denotes the estimated noise of the signal 
                - applies threshold to coefficients of the leves
                - inverse transforms signal

        Parameters
        ----------
        eda : pd.Series
            raw EDA data
        i : float
            estimated noise of the signal
        level : int
            how many levels the wavelet decomposition has
        wavelet : str, optional
            kind of wavelet, by default 'haar'

        Returns
        -------
        array
            denoised EDA signal
        """
        max_level = pywt.swt_max_level(len(eda))
        if max_level-level >=3: max_level=max_level-level
        coeffs = pywt.swt(eda, wavelet, max_level)
        thresholded_coeffs = []
        for cA, cD in coeffs:
            b_hat = estimate_scale_parameter(cD)
            higher, lower= generate_threshold(b_hat, i)
            thresholded_cD = apply_treshold_wavelet(cD, higher,lower)
            thresholded_coeffs.append((cA, thresholded_cD))
        eda_denoised = pywt.iswt(thresholded_coeffs, wavelet)
        return eda_denoised



def cclean_with_hamming(eda_clean:pd.Series)->pd.Series:

    """
    Clean EDA signal using a Hamming filter.

    This function applies a high-pass Hamming filter to clean the Electrodermal Activity (EDA) signal.
    The filtered signal is then returned as a DataFrame with the same index as the input.

    Parameters
    ----------
    eda_clean : pd.Series
        A pandas Series containing the EDA signal data to be cleaned.

    Returns
    -------
    pd.Series
        A pandas Series containing the cleaned EDA signal.

    Notes
    -----
    The Hamming filter used in this function is designed with the following specifications:
    - Filter order: 3
    - Cutoff frequency: 0.05
    - Pass type: High-pass
    - Sampling frequency: 4 Hz

    The function was created implemented after the paper by Brandebusemeyer et al, for details see Paper Doi: 10.1109/ICSE-SEIP66354.2025.00008

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import scipy.signal as sc
    >>> eda_clean = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> clean_signal = cclean_with_hamming(eda_clean)
    >>> print(clean_signal)
    """

    ham_filter = sc.signal.firwin(3, cutoff=0.05, pass_zero="highpass", fs=4)
    eda_clean1 = np.array([np.convolve(x, ham_filter, mode="valid") for x in eda_clean.iloc[:]])
    return pd.DataFrame(eda_clean1, index=eda_clean.index)



def estimate_event_related_SCR(cleaned_eda:pd.DataFrame,eda_base_pre:pd.DataFrame, eda_base_post:pd.DataFrame)-> float:
        """Estimates the event ralted SCR

        Parameters
        ----------
        src_task : pd.DataFrame
            dataframe with SCR componet of the eda signal from a task recording
        eda_base_pre : pd.DataFrame
            dataframe with SCR from the baseline recording
        eda_base_post : pd.DataFrame
            dataframe with SCR from the baseline recording    

        Returns
        -------
        float
            ERSCR
        """

        evcr = cleaned_eda['EDA_SCR_Peaks'].mean()/len(cleaned_eda)
        evr_post = eda_base_post['EDA_SCR_Peaks'].mean()/len(eda_base_post)
        evr_pre = eda_base_pre['EDA_SCR_Peaks'].mean()/len(eda_base_pre)
        return evr_post, evr_pre


# In[18]:


def calculate_skin_conductance_magintude(scr_task: pd.Series, time:pd.Series, name: Optional[str] = '', path: Optional[str] = '', call_plot:bool=False) -> np.array:
        """Calculates the magnitude of the signal by cumulative integrating the signal using trapezoid rule.

        Parameters
        ----------
        scr_task : pd.Series
            The scr signal of the EDA signal
        time : pd.Series
            Timestamps
        name : Optional[str], optional
            Name of the participant and the task/baseline for saving the images/csv files, by default ''
        path : Optional[str], optional
            Path to the folder where the image/csv file should be saved., by default ''

        Returns
        -------
        pd.Series
           magnitude of the signal
        """
        time_numeric = np.linspace(0, len(scr_task), len(scr_task))
        magnitude = sc.integrate.cumulative_trapezoid(scr_task, x=time_numeric)
        if call_plot:
            ig3 =plt.figure() 
            plt.axis('off')
            plt.setp(plt.axes().get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.title('Magnitude_SCR'+'_'+name,wrap=True)
            plt.plot(time[:-1], magnitude)
            if path=='':plt.savefig('Magnitude_SCR'+'_'+name+'.pdf', bbox_inches = 'tight')
            else:plt.savefig(path+'/Magnitude_SCR'+'_'+name+'.pdf', bbox_inches = 'tight')
            plt.close(ig3)
        return pd.Series(magnitude, index=time[:-1], name='EDA_Magnitude')



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


# 3 # get changed
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
        EDA.loc[EDA.EDA <0] = np.nan
       
        missing_values.append(EDA[EDA ==np.nan]) 
        EDA = EDA.dropna(axis='index')
        return EDA, missing_values


# In[21]:
def eda(raw_eda: pd.Series, acc_clean: pd.DataFrame, sampling_freq: int = 4, show: bool = False, wavelet: str = 'haar', SAVEPATH: str = None, name: str = 'EDA processed') -> pd.DataFrame:
    """
    Process Electrodermal Activity (EDA) signal by cleaning, denoising, and decomposing into tonic and phasic components.

    Parameters
    ----------
    raw_eda : pd.Series
        Raw EDA signal.
    acc_clean : pd.DataFrame
        Cleaned accelerometer data.
    sampling_freq : int, optional
        Sampling frequency in Hz, by default 4.
    show : bool, optional
        Whether to plot the results, by default False.
    wavelet : str, optional
        Type of wavelet to use for denoising, by default 'haar'.
    SAVEPATH : str, optional
        Path to save the plot, by default None.
    name : str, optional
        Name for the saved plot, by default 'EDA processed'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the following columns:
        - 'EDA_Tonic': Tonic component of the EDA signal.
        - 'EDA_Phasic': Phasic component of the EDA signal.
        - 'EDA_Clean': Cleaned EDA signal.
        - 'EDA_Raw': Raw EDA signal.

    Notes
    -----
    The function performs the following steps:
    1. Downsample and interpolate accelerometer data.
    2. Calculate RMS of accelerometer data and determine cut-off.
    3. Remove artifacts from raw EDA signal using accelerometer data.
    4. Denoise the EDA signal using wavelet transform.
    5. Sanitize the denoised EDA signal.
    6. Apply Hamming filter to clean the sanitized EDA signal.
    7. Remove outliers from the cleaned EDA signal.
    8. Interpolate missing values in the cleaned EDA signal.
    9. Decompose the cleaned EDA signal into tonic and phasic components using the 'sparseda' method.
    10. Concatenate the tonic, phasic, cleaned, and raw EDA signals into a single DataFrame.

    If `show` is True and `SAVEPATH` is provided, the function will plot the results and save the plot to the specified path.
    """
    # clean with acc
    acc_down = acc_clean.resample(str(1/sampling_freq)+'s').interpolate(method='cubicspline',limit_direction='both')
    acc_rms = calculate_rms(acc=acc_down)
    up= get_cut_off(acc=acc_rms)
    eda_f, mv = remove_artifacts(raw_eda, up, acc_rms.rms)
    if len(eda_f)%2==0:
        eda_d = denoise_signal(eda_f.EDA[:], i=0.25, level=3, wavelet=wavelet)
        eda_d = pd.Series(eda_d, index=eda_f[:].index, name='EDA')
    else: 
        eda_d = denoise_signal(eda_f.EDA[:-1], i=0.25, level=3, wavelet=wavelet)  
        eda_d = pd.Series(eda_d, index=eda_f[:-1].index, name='EDA')
    
    eda_s= nk.signal.signal_sanitize(eda_d)
    try:
        eda_ham = cclean_with_hamming(eda_clean=pd.Series(eda_s, index=eda_f.index))
    except ValueError:
        eda_ham = cclean_with_hamming(eda_clean=pd.Series(eda_s, index=eda_f.index[:-1]))     
    eda_ham= eda_ham[1];eda_ham.name = 'EDA_Clean'
    eda_m= eda_ham.apply(lambda x: np.nan if x > eda_ham.mean() + 3 * eda_ham.std() else x) 
    mv1 = pd.concat(mv, axis='index')
    mv1[:]=np.nan
    mv1.columns = ['EDA_Clean']
    mv1 = mv1.loc[~mv1.index.duplicated()]
    eda_i = pd.concat([eda_m,mv1]).sort_index()
    eda_i = eda_i.loc[~eda_i.index.duplicated()]
    eda_ii = interpolate_nan_c(eda_i.EDA_Clean)
    eda_decomp = nk.eda_phasic(
                        eda_ii,
                        sampling_rate=sampling_freq,
                        method='sparseda', 
                    )
    eda_decomp.index =eda_ii.index
    eda_compl1 = pd.concat([eda_decomp, eda_ii, raw_eda], axis='columns'); eda_compl1.index = eda_ii.index; eda_compl1.columns = ['EDA_Tonic', 'EDA_Phasic', 'EDA_Clean', 'EDA_Raw']
    if show and SAVEPATH:
        fast_plot(eda_compl1[['EDA_Clean', 'EDA_Raw','EDA_Tonic', 'EDA_Phasic']].reset_index(drop=True), 4, name=name, path = SAVEPATH)




    return eda_compl1    
    


 
def calculate_accumulative_eda(eda_signal: pd.Series) -> pd.Series:
    """
    Calculate the accumulative EDA signal.

    Parameters
    ----------
    eda_signal : pd.Series
        Series containing the EDA signal.

    Returns
    -------
    pd.Series
        Series containing the accumulative EDA signal.
    """
    accumulative_eda = copy.deepcopy(eda_signal)
    for i in range(1, len(eda_signal)):
        accumulative_eda[i] = accumulative_eda[i-1] + eda_signal[i]
    return accumulative_eda




def feature(tasks_lst_eda:list[pd.DataFrame],eda_complete:pd.DataFrame, sampling_freq:int=4, timewindow:int=30, overlapp:Optional[int]=None)->tuple[pd.DataFrame, pd.DataFrame]:
        """
            Calculate features for EDA signals within specified time windows and overlap.

            This function computes various statistical and physiological features from Electrodermal Activity (EDA) signals.
            It processes the signals in specified time windows with overlap and returns the cleaned features.

            Parameters
            ----------
            tasks_lst_eda : list[pd.DataFrame]
                List of pandas DataFrames containing EDA signal data for different tasks.
            eda_complete : pd.DataFrame
                DataFrame containing complete EDA signal data with labels.
            sampling_freq : int, optional
                Sampling frequency of the EDA signal, by default 4 Hz.
            timewindow : int, optional
                Size of the time window in seconds for feature calculation, by default 30 seconds.
            overlapp : Optional[int], optional
                Overlap size in seconds for the time windows, by default half of the timewindow.

            Returns
            -------
            tuple[pd.DataFrame, pd.DataFrame]
                Tuple containing two DataFrames:
                - Cleaned features DataFrame.
                - Labels DataFrame.

            Notes
            -----
            The function calculates the following features:
            - **EDA_mean**: Mean value of the EDA signal.
            - **EDA_slope**: Slope of the EDA signal, indicating the rate of change.
            - **EDA_median**: Median value of the EDA signal.
            - **EDA_IQR**: Interquartile range of the EDA signal, representing variability.
            - **EDA_min**: Minimum value of the EDA signal.
            - **EDA_max**: Maximum value of the EDA signal.
            - **EDA_symp**: Sympathetic activity derived from the EDA signal (if timewindow > 64).
            - **SCR_Onsets**: Onsets of skin conductance responses.
            - **SCR_Peaks**: Peaks of skin conductance responses.
            - **SCR_Height**: Height of skin conductance responses.
            - **SCR_Amplitude**: Amplitude of skin conductance responses.
            - **SCR_RiseTime**: Rise time of skin conductance responses.
            - **SCR_Recovery**: Recovery of skin conductance responses.
            - **SCR_RecoveryTime**: Recovery time of skin conductance responses.
            - **EDA_Acum**: Accumulative EDA signal.
            - **EDA_magnitude**: Magnitude of skin conductance.
            - **EDA_Tonic**: Tonic component of the EDA signal.
            - **EDA_Phasic**: Phasic component of the EDA signal.

            Example
            -------
            >>> import pandas as pd
            >>> import numpy as np
            >>> import neurokit2 as nk
            >>> tasks_lst_eda = [pd.DataFrame({'EDA_Clean': [0.1, 0.2, 0.3, 0.4, 0.5], 'EDA_Tonic': [0.05, 0.1, 0.15, 0.2, 0.25]})]
            >>> eda_complete = pd.DataFrame({'task': ['task1'], 'cat_1': ['cat1'], 'cat_2': ['cat2'], 'participant': ['participant1']})
            >>> clean_features, labels_eda = feature(tasks_lst_eda, eda_complete)
            >>> print(clean_features)
            >>> print(labels_eda)
        """

        if not overlapp: overlapp=timewindow//2
        stat_edas=generate_statistical_features(tasks_lst_eda, sampling_rate=sampling_freq,timewindow=timewindow,overlap=overlapp).dropna(how='all')
        f_eda_compl = []
        eda_feature = []
        for task in tasks_lst_eda:
                if len(task) ==0: continue
                if timewindow > 64:
                        
                        starttime = task.index[0]
                        
                        while starttime + pd.Timedelta(seconds=timewindow) <= task.index[-1]:
                                endtime = starttime + pd.Timedelta(seconds=timewindow)
                                neurokit_eda = nk.eda.eda_sympathetic(task.loc[starttime:endtime].EDA_Clean, sampling_rate=4, frequency_band=[0.045, 0.25], method='posada', show=False)
                                nk_eda = pd.DataFrame(neurokit_eda, index=[endtime])
                                eda_feature.append(nk_eda
                                )
                                starttime = starttime + pd.Timedelta(seconds=overlapp)
                
                eda_magnitude = calculate_skin_conductance_magintude(task.EDA_Tonic, time=task.index)
                eda_acum = calculate_accumulative_eda(eda_signal=task.EDA_Clean)
                eda_acum.name = 'EDA_Acum'
                print(len(task.EDA_Tonic))
                try:
                    signals, info = nk.eda.eda_peaks(task.EDA_Tonic.values, sampling_rate=sampling_freq, method='neurokit')
                except ValueError:signals= pd.DataFrame(columns=['SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude',
                                    'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime'] ,index=[x for x in range(len(task))])  
                new_Cols = 'EDA_'+signals.columns
                signals.columns = new_Cols
                signals.index = task.index
                        
        
                
                eda_compl=pd.concat([task, signals, eda_acum, eda_magnitude], axis='columns')
                f_eda_compl.append(eda_compl.rolling(window=timewindow*sampling_freq, step=overlapp*sampling_freq).mean(numeric_only=True).dropna(how='all'))
        
        feature_eda_compl = pd.concat(f_eda_compl).dropna(how='all')   
        feature_eda_compl = feature_eda_compl.loc[~feature_eda_compl.index.duplicated()]
        stat_edas = stat_edas.loc[~stat_edas.index.duplicated()]
        f_eda_compl_stat = pd.concat([feature_eda_compl, stat_edas], axis='columns')
        clean_features = f_eda_compl_stat.dropna(axis='columns', how='all')

        for i in ['EDA_Raw'#,'EDA_SCR_Onsets', 'EDA_SCR_Peaks', 'EDA_SCR_Height',
                # 'EDA_SCR_Amplitude', 'EDA_SCR_RiseTime', 'EDA_SCR_Recovery',
                #'EDA_SCR_RecoveryTime'
                ]:
                try:
                        clean_features = clean_features.drop(i, axis='columns')
                except KeyError: pass
        print(clean_features.index)

        
        if len(eda_feature)>0:
                eda_feature = pd.concat(eda_feature, axis='index') 
                labels_eda=eda_complete.loc[clean_features.index][['task', 'cat_1','cat_2', 'participant']]
                clean_features = pd.concat([clean_features, eda_feature, labels_eda],axis='columns')        
        else: 
                labels_eda=eda_complete.loc[clean_features.index][['task', 'cat_1','cat_2', 'participant']]
                clean_features = pd.concat([clean_features, labels_eda],axis='columns')
        return clean_features
  
  














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

    # Interpolate NaN values
    data_interpolated = interpolator(np.arange(len(data)))
    data_interpolated = pd.Series(data_interpolated, name=data.name, index=data_index)

    return data_interpolated 


# In[52]:


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
    if len(outlier) ==0: return data
    # Set outlier values to NaN
    if isinstance(outlier, list):
        if len(np.concatenate(outlier))==0: return data
        if len(outlier) ==0: return data
        temp = [np.nan] * len(outlier)
        outlier = pd.Series(temp, index=outlier)

    else : 
        outlier.values[:] = np.nan
    name = data.name
    data = pd.concat([outlier, data])
    data.sort_index(inplace=True)
    data_index = data.index

    # Get indices of valid (non-NaN) values
    valid = ~np.isnan(data.values)
    x_valid = np.where(valid)[0]
    y_valid = data.values[valid]

    # Create PCHIP interpolator
    interpolator = PchipInterpolator(x_valid, y_valid)

    # Interpolate NaN values
    data_interpolated = interpolator(np.arange(len(data)))
    data_interpolated = pd.Series(data_interpolated, name=name, index=data_index)

    return data_interpolated  




def stats_eda(eda:pd.Series, window:float, step:float)-> pd.Series:
        """Calculates the statistical features of the eda including: 
            - Median
            - Mean
            - Max
            - Min
            - IQR

        Parameters
        ----------
        eda : pd.Series
            processed eda data
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            median_eda, max_eda, min_eda, eda_iqr
        """
        median_eda = eda.rolling(window=window, step=step).median().dropna()
        max_eda = eda.rolling(window=window, step=step).max().dropna()
        min_eda = eda.rolling(window=window, step=step).min().dropna()
        eda_iqr = eda.rolling(window=window, step=step).quantile(0.75)-eda.rolling(window=window).quantile(0.25).dropna()

        return median_eda, max_eda, min_eda, eda_iqr




def get_slope(eda_data:pd.Series)->pd.Series:
        """Computes the slope with the scipy's regession function

        Parameters
        ----------
        eda_data : pd.Series
            cleaned eda data

        Returns
        -------
        pd.Series
            slope of the eda
        """
        x = np.arange(0,len(eda_data), 1)
        
        y = eda_data
        slope,intercept, r_value,p, std_err,  = sc.stats.linregress(x,y)
        return slope

    
def calculate_slope_eda(eda:pd.Series, window:float, step:float)->pd.Series:
        """Calculates the slope of the eda (gradient)

        Parameters
        ----------
        eda : pd.Series
            The processed eda values
        window : float
            window size of the sliding window
        step : float
            number of steps the sliding window should move

        Returns
        -------
        pd.Series
            slope, minimal slope, maximal slope, mean slope
            
        """
        slope = eda.rolling(window=window, step=step).apply(get_slope, raw=True).dropna()
        min_slope = slope.rolling(window=window, step=step).min().dropna()
        max_slope = slope.rolling(window=window, step=step).max().dropna()
        mean_slope = slope.rolling(window=window, step=step).mean().dropna()
        return slope, min_slope, max_slope, mean_slope


# In[55]:


def generate_statistical_features(eda_lst: list[pd.DataFrame], sampling_rate: int = 64, timewindow: int = 30, overlap: Optional[int] = 15) -> pd.DataFrame:
    """
    Generate statistical features from a list of PPG DataFrames.

    Parameters
    ----------
    eda_lst : list[pd.DataFrame]
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
    - EDA_median
    - EDA_max
    - EDA_min
    - EDA_iqr
    - EDA_slope
    - EDA_minimal_slope
    - EDA_maximal_slope
    - EDA_mean_slope
    - EDA_HR_mean
    """
    stat_features = []
    timewindow = timewindow * sampling_rate
    overlap = overlap * sampling_rate
    for task in eda_lst:
        median_eda, max_eda, min_eda, eda_iqr = stats_eda(task.EDA_Clean, window=timewindow, step=overlap)
        slope, minimal_slope, maximal_slope, mean_slope = calculate_slope_eda(task.EDA_Clean, window=timewindow, step=overlap)
        emp_stat_frame = pd.concat([median_eda, max_eda, min_eda, eda_iqr, slope, minimal_slope, maximal_slope, mean_slope], axis='columns')
        emp_stat_frame.columns = ['EDA_median', 'EDA_max', 'EDA_min', 'EDA_iqr', 'EDA_slope', 'EDA_minimal_slope', 'EDA_maximal_slope', 'EDA_mean_slope']
        stat_features.append(emp_stat_frame)
    return pd.concat(stat_features, axis='index').sort_index().dropna(how='all')


# In[56]:


def create_time_windows_ml(signal_task: list[pd.DataFrame], timewindow: int, overlapp: Optional[int] = None) -> pd.DataFrame:
    """
    Create time windows for machine learning from signal data.

    This function generates time windows (e.g., 30 seconds) from the provided signal data and calculates the mean 
    value for each time window. The features are computed for each time window and not for the entire signal. If an 
    overlap is specified, the function will use the overlap to determine the start of the next time window.

    Parameters:
    signal_task (list[pd.DataFrame]): List of DataFrames containing the signal data with a DatetimeIndex.
    timewindow (int): The length of the time window in seconds for which features are to be calculated.
    overlapp (Optional[int]): The overlap in seconds between consecutive time windows. If not specified, 
                              the default overlap is half of the time window length.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated mean values for each time window.
    """
    features = []
    if not overlapp: overlapp = timewindow // 2

    for tasks in signal_task:
        starttime = tasks.index[0]
    
        while starttime + pd.Timedelta(seconds=timewindow) <= tasks.index[-1]:
            endtime = starttime + pd.Timedelta(seconds=timewindow)
            features.append(tasks.loc[starttime:endtime].mean())
            starttime = starttime + pd.Timedelta(seconds=overlapp)
    
    return pd.concat([features], axis='index')

