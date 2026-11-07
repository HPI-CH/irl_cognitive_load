#!/usr/bin/env python
# coding: utf-8
"""
ACC Functions Module
=====================

This module provides functions for processing acceleration data.

Steps
-----
1. Remove noise and non-human frequencies using either Chebyshev (method='my') or Butterworth filter of 4th order.
2. Segment data into time windows and separate into tasks.
3. Calculate features.

Features
--------
- Magnitude: The overall magnitude of the acceleration signal.
- Mean signal: The average value of the acceleration signal.
- Minimum signal: The lowest value of the acceleration signal.
- Maximum signal: The highest value of the acceleration signal.
- Interquartile range (IQR): The range between the first and third quartile of the acceleration signal, indicating the spread of the middle 50% of the data.
- Activity index: A measure of the activity level based on the acceleration signal.
- Signal magnitude: The magnitude of the acceleration signal.
- Vector magnitude: The magnitude of the acceleration vector, calculated as the square root of the sum of the squares of the individual axis values.
"""


# Imports
import copy
import inspect
import os
from matplotlib import pyplot as plt
import pandas as pd
from typing import Optional, Any
import scipy
import numpy as np
from . import acc_function_biosppy
import datetime
from ..utils import reading_functions




# NO USE # XXX die empatica daten befinden wsich bereits in der linearen Form
def  remove_gravity_acc(signal_with_acc: pd.Series,sampling_freq: float, use_mean: Optional[bool] = False, use_filter: Optional[bool] = True, window_size : Optional[float]=10, cut_off: Optional[float] =0.1) -> pd.Series:
        """remove the gravitiy form the accerloration data

        Parameters
        ----------
        signal_with_acc : pd.Series
            accelorater axis and its values
        sampling_freq : float
            sampling frequency of the signal
        use_mean : Optional[bool], optional
            use the mean of the singal to remove the gravitity, not recommended, by default False
        use_filter : Optional[bool], optional
            use the filtered values to get the gravitiy and subtract it from the acceleration data, by default True
        window_size : Optional[float], optional
            window size for the rolling mean, by default 10
        cut_off : Optional[float], optional
            choosen cut off frequency of the lowpass filter should be between [0.1, 0.5], by default 0.1

        Returns
        -------
        pd.Series
            linear acceleration data
        """
        if use_mean:
                signal_with_acc = signal_with_acc- signal_with_acc-signal_with_acc.rolling(window=window_size).mean()
                
        elif use_filter:
                butter_a, butter_b = scipy.signal.butter(N =1,Wn=(2*cut_off/sampling_freq),btype= 'low', fs = sampling_freq)
                name = signal_with_acc.name

                signal_with_acc= pd.Series(signal_with_acc - scipy.signal.filtfilt(butter_a,butter_b,signal_with_acc, padlen = 0),name = name, index=signal_with_acc.index)
        elif not use_filter and not use_mean:
                signal_with_acc = remove_gravity_acc(signal_with_acc, sampling_freq, use_mean=False, use_filter=True, window_size=7, cut_off=0.1)

        return signal_with_acc    






def filter_acc(acc: pd.Series, sampling_freq: float, high: Optional[float] = 2.5, low: Optional[float] = 0.25)-> pd.Series:
        """Filters the acceleration signal with a bandpass filter

        Parameters
        ----------
        acc : pd.Series
            acceloration data of a axis
        sampling_freq : float
            Sampling frequency of the signal
        high : Optional[float], optional
            the upper cut off frequency, by default 2.5
        low : Optional[float], optional
            the lower cut off frequency, by default 0.25

        Returns
        -------
        pd.Series
            filtered acceleration signal

        Notes
        -----
        Implemented accoring the paper: https://doi.org/10.1016/j.procs.2014.07.009.    
        """
        a, b, = scipy.signal.cheby2(4, 25, [low/(0.5*sampling_freq), high/(0.5*sampling_freq)], btype='bandpass')
       
       
        name = acc.name
        if 3 * max(len(a), len(b)) > len(acc): pad_len = len(acc)-1; 
        else: pad_len = 3 * max(len(a), len(b));
        return pd.Series(scipy.signal.filtfilt(a,b, acc, padlen = pad_len), name= name, index=acc.index)


def filter_acc_chebyshev_sos(acc: pd.Series, sampling_freq: float, high: Optional[float] = 2.5, low: Optional[float] = 0.25)-> pd.Series:
        """Filters the acceleration signal with a bandpass filter

        Parameters
        ----------
        acc : pd.Series
            acceloration data of a axis
        sampling_freq : float
            Sampling frequency of the signal
        high : Optional[float], optional
            the upper cut off frequency, by default 2.5
        low : Optional[float], optional
            the lower cut off frequency, by default 0.25

        Returns
        -------
        pd.Series
            filtered acceleration signal

        Notes
        -----
        Implemented accoring the paper: https://doi.org/10.1016/j.procs.2014.07.009.    
        """
        sos= scipy.signal.cheby2(4, 25, [low/(0.5*sampling_freq), high/(0.5*sampling_freq)], btype='bandpass', output='sos')
       
       
     
        return pd.Series(scipy.signal.sosfilt(sos, acc), index=acc.index, name=acc.name)





def filter_acc_brandebusemeyers_version(acc: pd.Series) -> pd.Series:
    """
    Filter 3-axis acceleration signals using a bandpass filter.

    Parameters
    ----------
    acc : pd.Series
        A pandas Series containing 3-axis acceleration signals.

    Returns
    -------
    pd.Series
        The filtered acceleration signals as a pandas Series.

    Notes
    -----
    This function applies a bandpass filter with a low cutoff frequency of 0.5 Hz 
    and a high cutoff frequency of 4 Hz to the input acceleration signals. The 
    sampling frequency is set to 32 Hz.
    The function was implemented from the paper by Brandebusemeyer et al., for details see Paper Doi: 10.1109/ICSE-SEIP66354.2025.00008

    Examples
    --------
    >>> import pandas as pd
    >>> import scipy.signal
    >>> acc = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> filtered_acc = filter_acc_brandebusemeyers_version(acc)
    >>> print(filtered_acc)
    """
    sos = scipy.signal.butter(Wn=[0.5,4], N =4, btype='bandpass', fs=32, output='sos')
    return pd.Series(scipy.signal.sosfilt(sos, acc), index=acc.index, name=acc.name)




def find_peaks(acc:pd.Series)-> tuple[pd.Series,pd.Series]:
        """Finds the peaks in a given signal, uses ``scipy.find_peaks()`` function

        Parameters
        ----------
        acc : pd.Series
            acceleration signal/gyroskop signal

        Returns
        -------
        tuple[pd.Series,pd.Series]
            contains the peaks, where peaks are 1 all other values are 0, Promience of each of the peaks
            
        """
        prom = scipy.signal.find_peaks(acc)
        max_prom = np.max(acc[prom[0]])/3
        peaks = scipy.signal.find_peaks(acc, prominence=max_prom)
        peak= pd.Series(np.zeros(len(acc)-1), name = acc.name+'_peaks', index=acc.index[:-1])

        prominances = pd.Series(np.zeros(len(acc)-1), name = acc.name+'_prominences', index=acc.index[:-1])
        peak.iloc[peaks[0]] = 1
        prominance = scipy.signal.peak_prominences(acc, peaks[0])
        for i in range(len(peaks[0])):
    
            prominances[peaks[0][i]] = prominance[0][i]
        return peak, prominances, peaks[0]




def retrieve_name(var)->str:
        

    """
    Return the name of a variable by inspecting outer stack frames.

    This function searches through the outer stack frames to identify
    the name of the variable passed as an argument. It inspects frames
    from the outermost scope inward until a matching variable reference
    is found.

    Parameters
    ----------
    var : any
        The variable whose name should be retrieved.

    Returns
    -------
    str
        The name of the variable as a string. If no matching name is
        found, an empty string may be returned.

    Notes
    -----
    This function relies on stack inspection, which may not work in all
    contexts (e.g., optimized runtimes, interactive shells, or when
    multiple variables reference the same object).
    """


    for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                if names[0].startswith('_') and names[0].endswith('_') and len(names) > 1:
                    names[0] = names[1]
                return names[0]




def plot_frame(data:pd.DataFrame, name:str, path:str, is_acc = True)->None:
        """Plots the processed signal after the preprocessing and saves the image into a pdf file

        Parameters
        ----------
        data : pd.DataFrame
            The signal whci should be plotted
        name : str
            Name of the signal, under which the image should be saved. Also it used as a title
        path : str
            Path to folder/place where the image should be saved
        """
        
        figws = plt.figure()
        plt.tight_layout()
        plt.plot(data, label = data.columns.values)
        n = retrieve_name(data)
        if not name: name = n
        plt.title('Processed '+name,wrap=True)
        plt.ylabel('Signals in g') if is_acc else plt.ylabel('Signals in rad/s')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(path+'/processed_'+n+'_'+name+'.pdf', bbox_inches = 'tight')
        plt.close(figws)





import seaborn as sns
def fast_plot_frame(data:pd.DataFrame, name:str, path:str,sampling_freq:int=32, is_acc = True)->None:
        """Plots the processed signal after the preprocessing and saves the image into a pdf file

        Parameters
        ----------
        data : pd.DataFrame
            The signal whci should be plotted
        naerfassenme : str
            Name of the signal, under which the image should be saved. Also it used as a title
        path : str
            Path to folder/place where the image should be saved
        """
        
        figws = plt.figure(figsize=(30,10))
        plt.tight_layout()
        index = data.index
        x = [i /sampling_freq  for i in index]
        if is_acc: 
                for c in data.columns:
                        sns.lineplot(x=x, y=data[c].values, label = c)
        else: sns.lineplot(x=x, y=data.values, label = data.columns.values)
       
        n = retrieve_name(data)
        if not name: name = n
        plt.title(name,wrap=True)
        plt.ylabel('Signals in g') if is_acc else plt.ylabel('Signals in rad/s')
        plt.xlabel('time in s')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(path+'/processed_'+n+'_'+name+'.pdf', bbox_inches = 'tight')
        plt.close(figws)




def stats_acc(acc1:pd.DataFrame, window:float, step:float)->pd.DataFrame:
        """Calculates the statistical features 

        Parameters
        ----------
        acc : pd.DataFrame
            Acceleration data with all the columns
        window : float
            Size of the time window in which the staticsital features should be calculated
        step : float
            Step the slindig window should move. 

        Returns
        -------
        pd.DataFrame
             with Magintude, mean signal, minimum signal, maxium signal, interqantile
        """
        acc = copy.deepcopy(acc1)
        name = acc.columns[0]+ ' '+retrieve_name(acc1)
        names = []
        acc['ACC_norm_'+name] =  np.sqrt(np.square(acc).sum(axis=1)) # euklidische Norm / RMS #XXX
        names.append('ACC_norm_'+name)
        for cols in acc.columns:
            acc[cols+'_mean']= acc[cols].rolling(window=window, step=step).mean()
            names.append(cols+'_mean')
            acc[cols+'_min'] = acc[cols].rolling(window=window, step=step).min()
            names.append(cols+'_min')
            acc[cols+'_max'] = acc[cols].rolling(window, step=step).max()
            names.append(cols+'_max')
            acc[cols+'_iqr'] = acc[cols].rolling(window=window, step=step).quantile(0.75)-acc[cols].rolling(window=window, step=step).quantile(0.25)
            names.append(cols+'_iqr')

        return acc[names]



def get_attributes_acc(acc: pd.DataFrame, sampling_freq: int = 32, show: bool = True, path: Optional[str] = None, name: Optional[str] = 'Summary plot acc') -> tuple[pd.DataFrame, dict, dict]:
    """
    Calculate various attributes of an acceleration signal.

    This function calculates the signal magnitude, vector magnitude, frequency features, and amplitude features 
    of an acceleration signal. Optionally, it can display and save a summary plot of the acceleration signal.

    Parameters
    ----------
    acc : pd.DataFrame
        DataFrame containing the acceleration signal with columns ['x', 'y', 'z'].
    sampling_freq : int, optional
        The sampling frequency of the acceleration signal in Hz (default is 32).
    show : bool, optional
        If True, displays and saves a summary plot of the acceleration signal (default is True).
    path : Optional[str], optional
        The path where the summary plot will be saved (default is None, which uses the current directory).
    name : Optional[str], optional
        The title of the summary plot (default is 'Summary plot acc').

    Returns
    -------
    Tuple[pd.DataFrame, dict, dict]
        A tuple containing:
        - A DataFrame with the calculated magnitudes and resampled frequency and amplitude features.
        - A dictionary with the frequency features.
        - A dictionary with the amplitude features.
    """
    ts, _, vm, sm, freq_features, amp_features = acc_function_biosppy.acc(acc[['x', 'y', 'z']], sampling_rate=sampling_freq, units='g', path=path, show=show, name=name)
    
    if show:
        if not path or path == '': path = os.path.curdir
        plt.savefig(path + '/' + name + '_summary_acc_plot.pdf', bbox_inches='tight')
        plt.close('all')
    
    magnitudes = pd.DataFrame({'ACC_signal_magnitude': sm, 'ACC_Vector_magnitude': vm})
    time = get_time_back(ts, acc.x)
    magnitudes.index = time
    
    try:
        magnitudes['ACC_pos_freq_x'] = scipy.signal.resample(freq_features['x'], len(magnitudes), axis=-1)
        magnitudes['ACC_pos_freq_y'] = scipy.signal.resample(freq_features['y'], len(magnitudes), axis=-1)
        magnitudes['ACC_pos_freq_z'] = scipy.signal.resample(freq_features['z'], len(magnitudes), axis=-1)

        magnitudes['ACC_norm_amp_x'] = scipy.signal.resample(amp_features['x'], len(magnitudes), axis=-1)
        magnitudes['ACC_norm_amp_y'] = scipy.signal.resample(amp_features['y'], len(magnitudes), axis=-1)
        magnitudes['ACC_norm_amp_z'] = scipy.signal.resample(amp_features['z'], len(magnitudes), axis=-1)
    except Exception:
        pass
    
    return magnitudes, freq_features, amp_features

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


def calculate_activity_index(acc: pd.DataFrame, sampling_rate: int = 32, show: bool = False, name: Optional[str] = 'Activity index of acceleration signal in Hz', path: Optional[str] = '') -> pd.Series:
    """
    Calculate the activity index of an acceleration signal.

    This function calculates the activity index of an acceleration signal using the specified sampling rate. 
    Optionally, it can display and save a plot of the activity index.

    Parameters
    ----------
    acc : pd.DataFrame
        DataFrame containing the acceleration signal with columns ['x', 'y', 'z'].
    sampling_rate : int, optional
        The sampling rate of the acceleration signal in Hz (default is 32).
    show : bool, optional
        If True, displays and saves a plot of the activity index (default is False).
    name : Optional[str], optional
        The title of the plot (default is 'Activity index of acceleration signal in Hz').
    path : Optional[str], optional
        The path where the plot will be saved (default is the current directory).

    Returns
    -------
    pd.Series
        A Series containing the activity index with the corresponding timestamps.
    """
    ts1, activity_index = acc_function_biosppy.activity_index(acc[['x', 'y', 'z']], sampling_rate=sampling_rate)
    time = get_time_back(ts=ts1 / sampling_rate, orig_signal=acc.x)
    act_acc = pd.Series(activity_index, index=time, name='ACC_activity_index')
    
    if show:
        fshd = plt.figure(figsize=(30, 10))
        plt.tight_layout()
        sns.lineplot(x=ts1, y=activity_index, label='Activity index')
        plt.ylabel('Activity index in Hz')
        plt.xlabel('time in s')
        plt.title(name, wrap=True)
        if not path or path == '': path = os.path.curdir
        plt.savefig(path + '/' + name + '_activity_plot.pdf', bbox_inches='tight')
        plt.close('all')
    
    return act_acc



def generate_features_for_timewindows(signal_task: list[pd.DataFrame], timewindow: int, overlapp: Optional[int] = None) -> pd.DataFrame:
    
    """
    Generate features for fixed-length time windows from signal data.

    This function computes features for each consecutive time window
    (e.g., 30 seconds) extracted from the provided signal data. Features
    are calculated independently for each time window, rather than for
    the entire signal. If an overlap is specified, the starting point of
    each subsequent window is shifted accordingly. If no overlap is
    provided, it defaults to half of the time window length.

    Parameters
    ----------
    signal_task : list of pandas.DataFrame
        List of DataFrames containing signal data. Each DataFrame must
        have a ``DatetimeIndex`` and at least one numerical column
        representing signal values.
    timewindow : int
        Length of each time window in seconds for which features are
        computed.
    overlapp : int, optional
        Overlap between consecutive time windows in seconds. If ``None``,
        the overlap defaults to half of ``timewindow``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the computed features for each time window.
        Each row corresponds to one time window, and columns represent
        the generated feature values derived from the signal.

    Notes
    -----
    This function is intended for preprocessing time-series data for
    machine learning tasks, where features must be computed over fixed,
    sliding, or overlapping windows rather than over the entire signal.
    """

    features = []
    if not overlapp: overlapp = timewindow // 2

    for tasks in signal_task:
        starttime = tasks.index[0]
    
    while starttime + pd.Timedelta(seconds=timewindow) < tasks.index[-1]:
        endtime = starttime + pd.Timedelta(seconds=timewindow)
        features.append(get_attributes_acc(acc=tasks.loc[starttime:endtime], show=False)[0].mean())
        starttime = starttime + pd.Timedelta(seconds=overlapp)
    
    return pd.concat([features], axis='index')




def create_time_windows_ml(signal_task: list[pd.DataFrame], timewindow: int, overlapp: Optional[int] = None) -> pd.DataFrame:
  
    """
    Create fixed-length time windows from signal data for machine learning.

    This function generates consecutive time windows (e.g., 30 seconds) from
    the provided signal data and computes the mean value for each window.
    Features are calculated per time window, not for the entire signal.
    If an overlap is specified, the start of each subsequent window is shifted
    accordingly. If no overlap is provided, the default is half of the
    time window length.

    Parameters
    ----------
    signal_task : list of pandas.DataFrame
        List of DataFrames containing signal data. Each DataFrame must have a
        ``DatetimeIndex`` and at least one column of numerical values.
    timewindow : int
        Length of each time window in seconds for which features should
        be computed.
    overlapp : int, optional
        Overlap between consecutive windows in seconds. If ``None``,
        the overlap defaults to half of ``timewindow``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the computed mean values for each time window.
        Each row corresponds to one window, and columns represent the mean
        values of the respective signal features within that window.

    Notes
    -----
    This function is intended for preprocessing signal data for machine learning,
    where features are computed over fixed intervals rather than entire time series.
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


def clean_acc(acc: pd.DataFrame, sampling_rate: int = 32, methods: str = 'my', high: Optional[float] = 2.5, low: Optional[float] = 0.25) -> pd.DataFrame:
    """
    Cleans accelerometer data using specified filtering methods.

    Parameters
    ----------
    acc : pd.DataFrame
        DataFrame containing accelerometer data with columns 'x', 'y', and 'z'.
    sampling_rate : int, optional
        Sampling rate of the accelerometer data in Hz (default is 32).
    methods : str, optional
        Method to use for filtering ('my' for custom Chebyshev filter, otherwise uses brandebusemeyer's Butterworth filter) (default is 'my').
    high : float, optional
        High cutoff frequency for the Chebyshev filter (default is 2.5).
    low : float, optional
        Low cutoff frequency for the Chebyshev filter (default is 0.25).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cleaned accelerometer data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime
    >>> acc_data = pd.DataFrame({
    ...     'x': np.random.randn(100),
    ...     'y': np.random.randn(100),
    ...     'z': np.random.randn(100)
    ... }, index=pd.date_range(start='2023-01-01', periods=100, freq='S'))
    >>> clean_acc(acc_data)
    """
    
    if 'my' in methods:
        acc1_x = filter_acc_chebyshev_sos(acc['x'], sampling_freq=sampling_rate, high=high, low=low)
        acc1_y = filter_acc_chebyshev_sos(acc['y'], sampling_freq=sampling_rate, high=high, low=low)
        acc1_z = filter_acc_chebyshev_sos(acc['z'], sampling_freq=sampling_rate, high=high, low=low)
        acc_cleaned = pd.DataFrame({'x': acc1_x, 'y': acc1_y, 'z': acc1_z}, index=acc1_x.index)
    else:
        acc3_x_c = filter_acc_brandebusemeyers_version(acc['x'], order=4, lowcut=0.5, highcut=4)
        acc3_y_c = filter_acc_brandebusemeyers_version(acc['y'], order=4, lowcut=0.5, highcut=4)
        acc3_z_c = filter_acc_brandebusemeyers_version(acc['z'], order=4, lowcut=0.5, highcut=4)
        acc_cleaned = pd.DataFrame({'x_c': acc3_x_c, 'y_c': acc3_y_c, 'z_c': acc3_z_c}, index=acc3_x_c.index)
              
    return acc_cleaned



def features(acc_list: list[pd.DataFrame], acc_complete: pd.DataFrame, sampling_rate: int = 32, timewindow: int = 30, overlap: Optional[int] = None, option: Optional[str] = 'A') -> pd.DataFrame:
    """
    Extracts features from accelerometer data using specified methods and time windows.

    Parameters
    ----------
    acc_list : list of pd.DataFrame
        List of DataFrames containing accelerometer data with columns 'x', 'y', and 'z'.
    acc_complete : pd.DataFrame
        DataFrame containing complete accelerometer data with columns 'x', 'y', and 'z'.
    sampling_rate : int, optional
        Sampling rate of the accelerometer data in Hz (default is 32).
    timewindow : int, optional
        Time window in seconds for feature extraction (default is 30).
    overlap : int, optional
        Overlap in seconds between consecutive time windows (default is None).
    option : str, optional
        Option to choose the feature extraction method ('A' for detailed stats, otherwise uses biosppy features) (default is 'A').

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted features.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> acc_data1 = pd.DataFrame({
    ...     'x': np.random.randn(100),
    ...     'y': np.random.randn(100),
    ...     'z': np.random.randn(100)
    ... }, index=pd.date_range(start='2023-01-01', periods=100, freq='S'))
    >>> acc_data2 = pd.DataFrame({
    ...     'x': np.random.randn(100),
    ...     'y': np.random.randn(100),
    ...     'z': np.random.randn(100)
    ... }, index=pd.date_range(start='2023-01-02', periods=100, freq='S'))
    >>> acc_list = [acc_data1, acc_data2]
    >>> acc_complete = pd.concat(acc_list)
    >>> features(acc_list, acc_complete)

    Notes
    -----
    Generates the following features:
    - Magnitude: The overall magnitude of the acceleration signal.
    - Mean signal: The average value of the acceleration signal.
    - Minimum signal: The lowest value of the acceleration signal.
    - Maximum signal: The highest value of the acceleration signal.
    - Interquartile range (IQR): The range between the first and third quartile of the acceleration signal, indicating the spread of the middle 50% of the data.
    - Activity index: A measure of the activity level based on the acceleration signal.
    - Signal magnitude: The magnitude of the acceleration signal.
    - Vector magnitude: The magnitude of the acceleration vector, calculated as the square root of the sum of the squares of the individual axis values.


    """
    
    if option == 'A':
        feature_all_stats = []
        for task in acc_list:
            acc_stat = stats_acc(task[['x', 'y', 'z']], window=timewindow * sampling_rate, step=overlap * sampling_rate)
            magnitudes, freq_features, amp_features = get_attributes_acc(task[['x', 'y', 'z']], sampling_freq=sampling_rate)
           # acct_acc = calculate_activity_index(acc=task[['x', 'y', 'z']], show=False, sampling_rate=sampling_rate)acct_acc,
            all_acc = pd.concat([task,  magnitudes], axis='columns').dropna(how='all')
            all_acc = all_acc.rolling(window=timewindow*sampling_rate, step=overlap*sampling_rate).mean(numeric_only=True).dropna(how='all')
            labels_eda = task.loc[all_acc.index][['task', 'cat_1', 'cat_2', 'participant']]
            all_f = pd.concat([all_acc, labels_eda, acc_stat], axis='columns').dropna(how='all')
            feature_all_stats.append(all_f)
        clean_features = pd.concat(feature_all_stats).dropna(how='all')
    else:
        feature_all_stats = []
        for task in acc_list:
            acc_stat = stats_acc(task[['x', 'y', 'z']], window=timewindow * sampling_rate, overlap=overlap * sampling_rate)
            feature_all_stats.append(acc_stat)
        features_biosppy = generate_features_for_timewindows(acc_list, timewindow=timewindow, overlapp=overlap)
        feature_stats = pd.concat([feature_all_stats], axis='index').dropna(how='all')
        labels_eda = acc_complete.loc[feature_all_stats.index][['task', 'cat_1', 'cat_2', 'participant']]
        clean_features = pd.concat([features_biosppy, feature_stats, labels_eda], axis='columns').dropna(how='all')
         
    return clean_features       
               

