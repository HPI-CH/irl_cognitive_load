# -*- coding: utf-8 -*-
"""
biosppy.signals.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.


:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

Notes
-----
Adapted the hrv script and merged the ppg calculations into it to avoid the code will stop, it will now just save the images as pdf and will not display them.



"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range
import os

# 3rd party
import numpy as np
import warnings
from scipy.interpolate import interp1d
from scipy.signal import welch
from typing import Optional, Any

# # local
from ..utils import plotting_biosppy as plotting

from biosppy import utils
from biosppy import tools as st

import matplotlib
import matplotlib.pyplot as plt

# Global variables
FBANDS = {'ulf': [0, 0.003],
          'vlf': [0.003, 0.04],
          'lf': [0.04, 0.15],
          'hf': [0.15, 0.4],
          'vhf': [0.4, 0.5]
          }

NOT_FEATURES = ['rri', 'rri_trend', 'outliers_method', 'rri_det', 'hr', 'bins',
                'q_hist', 'fbands', 'frequencies', 'powers', 'freq_method']


def hrv(rpeaks=None, sampling_rate=1000., rri=None, parameters='auto',
        outliers='interpolate', detrend_rri=True, features_only=True,
        show=True,path=None, show_individual=False, name:Optional[str]=None, **kwargs):
    """Extracts the RR-interval sequence from a list of R-peak indexes and
    extracts HRV features.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz). Default: 1000.0 Hz.
    rri : array, optional
        RR-intervals (ms). Providing this parameter overrides the computation of
        RR-intervals from rpeaks.
    parameters : str, optional
        If 'auto' computes the recommended HRV features. If 'time' computes
        only time-domain features. If 'frequency' computes only
        frequency-domain features. If 'non-linear' computes only non-linear
        features. If 'all' computes all available HRV features. Default: 'auto'.
    outliers : str, optional
        Determines the method to handle outliers. If 'interpolate', replaces
        the outlier RR-intervals
        with cubic spline interpolation based on a local threshold. If 'filter',
        the RR-interval sequence
        is cut at the outliers. If None, no correction is performed. Default:
        'interpolate'.
    detrend_rri : bool, optional
        Whether to detrend the RRI sequence with the default method smoothness
        priors. Default: True.
    features_only : bool, optional
        Whether to return only the hrv features. Default: True.
    show : bool, optional
        Whether to show the HRV summary plot. Default: True.
    show_individual : bool, optional
        Whether to show the individual HRV plots. Default: False.
    name : str, Optional
        name of the image incl. path where to save the image, by default None    
    kwargs : dict, optional
        fbands : dictionary of frequency bands (Hz) to use.

    Returns
    -------
    rri : array
        RR-intervals (ms).
    rri_det : array
        Detrended RR-interval sequence (ms), if detrending was applied.
    hrv_features : dict
        The set of HRV features extracted from the RRI data. The number of
        features depends on the chosen parameters.
    """

    # check inputs
    if rpeaks is None and rri is None:
        raise TypeError("Please specify an R-Peak or RRI list or array.")

    parameters_list = ['auto', 'time', 'frequency', 'non-linear', 'all']
    if parameters not in parameters_list:
        raise ValueError(f"'{parameters}' is not an available input. Enter one"
                         f"from: {parameters_list}.")

    # ensure input format
    sampling_rate = float(sampling_rate)

    # initialize outputs
    out = utils.ReturnTuple((), ())
    hrv_td, hrv_fd, hrv_nl = None, None, None

    # compute RRIs
    if rri is None:
        rpeaks = np.array(rpeaks, dtype=float)
        rri = compute_rri(rpeaks=rpeaks, sampling_rate=sampling_rate,
                          filter_rri=False)

    # compute duration
    duration = np.sum(rri) / 1000.  # seconds

    # handle outliers
    if outliers is None:
        pass
    elif outliers == 'interpolate':
        rri = rri_correction(rri)
    elif outliers == 'filter':
        rri = rri_filter(rri)

    # add rri to output
    out = out.append([rri, str(outliers)], ['rri', 'outliers_method'])

    # detrend rri sequence
    if detrend_rri:
        rri_det, rri_trend = detrend_window(rri)
        # add to output
        out = out.append([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    else:
        rri_det = None
        rri_trend = None

    # plot
    if show_individual:
        plotting.plot_rri(rri, rri_trend, show=show_individual)
        if name:
            if not path: path = os.path.curdir()
            name = name + '_induvidual_rri.pdf'
            plt.savefig(path+'/'+name, bbox_inches='tight')
            plt.close('all')

    # extract features
    if parameters == 'all':
        duration = np.inf

    # compute time-domain features
    if parameters in ['time', 'auto', 'all']:
        try:
            hrv_td = hrv_timedomain(rri=rri,
                                    duration=duration,
                                    detrend_rri=detrend_rri,
                                    show=show_individual,
                                    rri_detrended=rri_det)
            out = out.join(hrv_td)

        except ValueError as e:
            print('WARNING: Time-domain features not computed. Check input.')
            print(e)

            pass

    # compute frequency-domain features
    if parameters in ['frequency', 'auto', 'all']:
        try:
            hrv_fd = hrv_frequencydomain(rri=rri,
                                         duration=duration,
                                         detrend_rri=detrend_rri,
                                         show=show_individual,
                                         fbands=kwargs.get('fbands', None))
            out = out.join(hrv_fd)
        except ValueError as e:
            print('WARNING: Frequency-domain features not computed. Check input.')
            print(e)
            pass

    # compute non-linear features
    if parameters in ['non-linear', 'auto', 'all']:
        try:
            hrv_nl = hrv_nonlinear(rri=rri,
                                   duration=duration,
                                   detrend_rri=detrend_rri,
                                   show=show_individual)
            out = out.join(hrv_nl)
                
        except ValueError as e:
            print('WARNING: Non-linear features not computed. Check input.')
            print(e)
            pass

    # plot summary
    if show:
        if hrv_td is not None and hrv_fd is not None and hrv_nl is not None:
            plotting.plot_hrv(rri=rri,
                              rri_trend=rri_trend,
                              td_out=hrv_td,
                              nl_out=hrv_nl,
                              fd_out=hrv_fd,
                              path=path,
                              saving_format=name,
                              show=True                              )
            # if name:
            #     if not path: path = os.path.curdir
            #     name = name + '_summary_hrv.pdf'
            #     plt.savefig(path+'/'+name, bbox_inches='tight')
            #     plt.close('all')
        else:
            warning = "Not all features were computed. To show the summary " \
                      "plot all features must be computed. Set " \
                      "'show_individual' to True to show the individual " \
                      "plots, or use parameters='all' to compute all features."
            warnings.warn(warning)

    # clean output if features_only
    if features_only:
        for key in NOT_FEATURES:
            try:
                out = out.delete(key)
            except ValueError:
                pass

    return out



def compute_rri(rpeaks, sampling_rate=1000., filter_rri=True, show=False, name:Optional[str]=None):
    """Computes RR intervals in milliseconds from a list of R-peak indexes.

    Parameters
    ----------
    rpeaks : list, array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    filter_rri : bool, optional
        Whether to filter the RR-interval sequence. Default: True.
    show : bool, optional
        Plots the RR-interval sequence. Default: False.

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # ensure input format
    rpeaks = np.array(rpeaks)

    # difference of R-peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # filter rri sequence
    if filter_rri:
        rri = rri_filter(rri)

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters."
                      "Check input values.")

    if show:
        plotting.plot_rri(rri, show=False)
        if name:
            path = os.getcwd()
            name = name + '_induvidual_rri.pdf'
            plt.savefig(path+'/'+name, bbox_inches='tight')
            plt.close('all')

    return rri

# js


def rri_filter(rri=None, threshold=1200):
    """Filters an RRI sequence based on a maximum threshold in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).

    Returns
    -------
    rri_filt : array
        Filtered RR-intervals (ms).
    """

    # ensure input format
    rri = np.array(rri, dtype=float)

    # filter rri values
    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt



def rri_correction(rri=None, threshold=250):
    """Corrects artifacts in an RRI sequence based on a local average threshold.
    Artifacts are replaced with cubic spline interpolation.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Local average threshold (ms). Default: 250.

    Returns
    -------
    rri : array
        Corrected RR-intervals (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure input format
    rri = np.array(rri, dtype=float)

    # compute local average
    rri_filt, _ = st.smoother(signal=rri, kernel='median', size=3)

    # find artifacts
    artifacts = np.abs(rri - rri_filt) > threshold

    # replace artifacts with cubic spline interpolation
    rri[artifacts] = interp1d(np.where(~artifacts)[0], rri[~artifacts],
                              kind='cubic')(np.where(artifacts)[0])

    return rri



def hrv_timedomain(rri, duration=None, detrend_rri=True, show=False, name:Optional[str]=None, **kwargs):
    """Computes the time domain HRV features from a sequence of RR intervals
    in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    hr : array
        Instantaneous heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rates (bpm).
    hr_mean : float
        Mean heart rate (bpm).
    hr_median : float
        Median heart rate (bpm).
    rr_min : float
        Minimum value of RR intervals (ms).
    rr_max : float
        Maximum value of RR intervals (ms).
    rr_minmax :  float
        Difference between the highest and the lowest values of RR intervals (ms).
    rr_mean : float
        Mean value of RR intervals (ms).
    rr_median : float
        Median value of RR intervals (ms).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # detrend
    if detrend_rri:
        if 'rri_detrended' in kwargs:
            rri_det = kwargs['rri_detrended']
        else:
            rri_det = detrend_window(rri)['rri_det']
        print('Time domain: the rri sequence was detrended.')
    else:
        rri_det = rri

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise ValueError("Signal duration must be greater than 10 seconds to "
                         "compute time-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # compute the difference between RRIs
    rri_diff = np.diff(rri_det)

    if duration >= 10:
        # compute heart rate features
        hr = 60 / (rri / 1000)  # bpm
        hr_min = hr.min()
        hr_max = hr.max()
        hr_minmax = hr.max() - hr.min()
        hr_mean = hr.mean()
        hr_median = np.median(hr)

        out = out.append([hr, hr_min, hr_max, hr_minmax, hr_mean, hr_median],
                         ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean',
                          'hr_median'])

        # compute RRI features
        rr_min = rri.min()
        rr_max = rri.max()
        rr_minmax = rri.max() - rri.min()
        rr_mean = rri.mean()
        rr_median = np.median(rri)
        rmssd = (rri_diff ** 2).mean() ** 0.5

        out = out.append([rr_min, rr_max, rr_minmax, rr_mean, rr_median, rmssd],
                         ['rr_min', 'rr_max', 'rr_minmax', 'rr_mean',
                          'rr_median', 'rmssd'])

    if duration >= 20:
        # compute NN50 and pNN50
        th50 = 50
        nntot = len(rri_diff)
        nn50 = len(np.argwhere(abs(rri_diff) > th50))
        pnn50 = 100 * (nn50 / nntot)

        out = out.append([nn50, pnn50], ['nn50', 'pnn50'])

    if duration >= 60:
        # compute SDNN
        sdnn = rri_det.std()

        out = out.append(sdnn, 'sdnn')

    if duration >= 90:
        # compute geometrical features (histogram)
        out_geom = compute_geometrical(rri=rri, show=show)

        out = out.join(out_geom)

    return out


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT',
                        fbands=None, detrend_rri=True, show=False,name:Optional[str]=None, path:Optional[str]=None, **kwargs):
    """Computes the frequency domain HRV features from a sequence of RR
    intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method.
    fbands : dict, optional
        Dictionary specifying the desired HRV frequency bands.
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.
    kwargs : dict, optional
        frs : resampling frequency for the RRI sequence (Hz).
        nperseg : Length of each segment in Welch periodogram.
        nfft : Length of the FFT used in Welch function.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency for each frequency band (Hz).
    {fbands}_pwr : float
        Absolute power for each frequency band (ms^2).
    {fbands}_rpwr : float
        Relative power for each frequency band (nu).
    lf_hf : float
        Ratio of LF-to-HF power.
    lf_nu : float
        Ratio of LF to LF+HF power (nu).
    hf_nu :  float
        Ratio of HF to LF+HF power (nu).
    total_pwr : float
        Total power.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    freq_methods = ['FFT']
    if freq_method not in freq_methods:
        raise ValueError(f"'{freq_method}' is not an available input. Choose"
                         f"one from: {freq_methods}.")

    if fbands is None:
        fbands = FBANDS

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # ensure minimal duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise ValueError("Signal duration must be greater than 20 seconds to "
                         "compute frequency-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())
    out = out.append(fbands, 'fbands')

    # resampling with cubic interpolation for equidistant samples
    frs = kwargs['frs'] if 'frs' in kwargs else 4
    t = np.cumsum(rri)
    t -= t[0]
    rri_inter = interp1d(t, rri, 'cubic')
    t_inter = np.arange(t[0], t[-1], 1000. / frs)
    rri_inter = rri_inter(t_inter)

    # detrend
    if detrend_rri:
        rri_inter = detrend_window(rri_inter)['rri_det']
        print('Frequency domain: the rri sequence was detrended.')

    if duration >= 20:

        # compute frequencies and powers
        if freq_method == 'FFT':
            nperseg = kwargs['nperseg'] if 'nperseg' in kwargs else int(len(rri_inter)/4.5)
            nfft = kwargs['nfft'] if 'nfft' in kwargs else (256 if nperseg < 256 else 2**np.ceil(np.log(nperseg)/np.log(2)))

            frequencies, powers = welch(rri_inter, fs=frs, scaling='density',
                                        nperseg=nperseg, nfft=nfft)

            # add to output
            out = out.append([frequencies, powers, freq_method],
                             ['frequencies', 'powers', 'freq_method'])

        # compute frequency bands
        fb_out = compute_fbands(frequencies=frequencies, powers=powers, show=False)

        out = out.join(fb_out)

        # compute LF/HF ratio
        lf_hf = fb_out['lf_pwr'] / fb_out['hf_pwr']

        out = out.append(lf_hf, 'lf_hf')

        # compute LF and HF power in normal units
        lf_nu = fb_out['lf_pwr'] / (fb_out['lf_pwr'] + fb_out['hf_pwr'])
        hf_nu = 1 - lf_nu

        out = out.append([lf_nu, hf_nu], ['lf_nu', 'hf_nu'])

        # plot
        if show:
            legends = {'LF/HF': lf_hf}
            for key in out.keys():
                if key.endswith('_rpwr'):
                    legends[key] = out[key]

            plotting.plot_hrv_fbands(frequencies, powers, fbands, freq_method,
                                     legends, show=False)
            if name:
                if not path: path = os.path.curdir()
                name = name + '_hrvfbands.pdf'
                plt.savefig(path+'/'+name, bbox_inches='tight')
                plt.close('all')

    return out



def hrv_nonlinear(rri=None, duration=None, detrend_rri=True, show=False, name:Optional[str]=None):
    """Computes the non-linear HRV features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
    sampen : float
        Sample entropy.
    appen : float
        Approximate entropy.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # check duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 90:
        raise ValueError("Signal duration must be greater than 90 seconds to "
                         "compute non-linear features.")

    # detrend
    if detrend_rri:
        rri = detrend_window(rri)['rri_det']
        print('Non-linear domain: the rri sequence was detrended.')

    # initialize outputs
    out = utils.ReturnTuple((), ())

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        cp = compute_poincare(rri=rri, show=show, name=name)
        out = out.join(cp)

        # compute sample entropy
        sampen = sample_entropy(rri)
        out = out.append(sampen, 'sampen')

    if len(rri) >= 800 or duration == np.inf:
        # compute approximate entropy
        appen = approximate_entropy(rri)
        out = out.append(appen, 'appen')

    return out



def compute_fbands(frequencies, powers, fbands=None, method_name=None,
                   show=False, name:Optional[str]=None, path:Optional[str]=None):
    """Computes frequency domain features for the specified frequency bands.

    Parameters
    ----------
    frequencies : array
        Frequency axis.
    powers : array
        Power spectrum values for the frequency axis.
    fbands : dict, optional
        Dictionary containing the limits of the frequency bands.
    method_name : str, optional
        Method that was used to compute the power spectrum. Default: None.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency of the frequency band (Hz).
    {fbands}_pwr : float
        Absolute power of the frequency band (ms^2).
    {fbands}_rpwr : float
        Relative power of the frequency band (nu).
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    df = frequencies[1] - frequencies[0]  # frequency resolution
    total_pwr = np.sum(powers) * df

    if fbands is None:
        fbands = FBANDS

    # compute power, peak and relative power for each frequency band
    for fband in fbands.keys():
        band = np.argwhere((frequencies >= fbands[fband][0]) & (frequencies <= fbands[fband][-1])).reshape(-1)

        # check if it's possible to compute the frequency band
        if len(band) == 0:
            continue

        pwr = np.sum(powers[band]) * df
        peak = frequencies[band][np.argmax(powers[band])]
        rpwr = pwr / total_pwr

        out = out.append([pwr, peak, rpwr], [fband + '_pwr', fband + '_peak',
                                             fband + '_rpwr'])

    # plot
    if show:
        # legends
        freq_legends = {}
        for key in out.keys():
            if key.endswith('_rpwr'):
                freq_legends[key] = out[key]

        plotting.plot_hrv_fbands(frequencies=frequencies,
                                 powers=powers,
                                 fbands=fbands,
                                 method_name=method_name,
                                 legends=freq_legends,
                                 show=False)
        if name:
            if not path: path = os.path.curdir()
            name = name + '_induvidual_rri.pdf'
            plt.savefig(path+'/'+name, bbox_inches='tight')
            plt.close('all')

    return out



def compute_poincare(rri, show=False, name:Optional[str]=None,path:Optional[str]=None ):
    """Compute the Poincaré features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    show : bool, optional
        If True, show the Poincaré plot.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    x = rri[:-1]
    y = rri[1:]

    # compute SD1, SD2 and S
    x1 = (x - y) / np.sqrt(2)
    x2 = (x + y) / np.sqrt(2)
    sd1 = x1.std()
    sd2 = x2.std()
    s = np.pi * sd1 * sd2

    # compute sd1/sd2 and sd2/sd1 ratio
    sd12 = sd1 / sd2
    sd21 = sd2 / sd1

    # output
    out = out.append([s, sd1, sd2, sd12, sd21], ['s', 'sd1', 'sd2', 'sd12',
                                                 'sd21'])

    if show:
        legends = {'SD1/SD2': sd12, 'SD2/SD1': sd21}
        plotting.plot_poincare(rri=rri,
                               s=s,
                               sd1=sd1,
                               sd2=sd2,
                               legends=legends,
                               show=False)
        if name:
            if not path: path = os.path.curdir()
            name = name + 'poincare.pdf'
            plt.savefig(path+'/'+name, bbox_inches='tight')
            plt.close('all')

    return out



def compute_geometrical(rri, binsize=1/128, show=False, name:Optional[str]=None,path:Optional[str]=None):
    """Computes the geometrical features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    show : bool, optional
        If True, show the RRI histogram. Default: False.

    Returns
    -------
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).

    """
    binsize = binsize * 1000  # to ms

    # create histogram
    tmin = rri.min()
    tmax = rri.max()
    bins = np.arange(tmin, tmax + binsize, binsize)
    nn_hist = np.histogram(rri, bins)

    # histogram peak
    max_count = np.max(nn_hist[0])
    peak_hist = np.argmax(nn_hist[0])

    # compute HTI
    hti = len(rri) / max_count

    # possible N and M values
    n_values = bins[:peak_hist]
    m_values = bins[peak_hist + 1:]

    # find triangle with base N and M that best approximates the distribution
    error_min = np.inf
    n = 0
    m = 0
    q_hist = None

    for n_ in n_values:

        for m_ in m_values:

            t = np.array([tmin, n_, nn_hist[1][peak_hist], m_, tmax + binsize])
            y = np.array([0, 0, max_count, 0, 0])
            q = interp1d(x=t, y=y, kind='linear')
            q = q(bins)

            # compute the sum of squared differences
            error = np.sum((nn_hist[0] - q[:-1]) ** 2)

            if error < error_min:
                error_min = error
                n, m, q_hist = n_, m_, q

    # compute TINN
    tinn = m - n

    # plot
    if show:
        plotting.plot_hrv_hist(rri=rri,
                               bins=bins,
                               q_hist=q_hist,
                               hti=hti,
                               tinn=tinn,
                               show=False)
        if name:
            if not path: path = os.path.curdir() 
            name = name + '_induvidual_rri.pdf'
            plt.savefig(path+'/'+name, bbox_inches='tight')
            plt.close('all')

    # output
    out = utils.ReturnTuple([hti, tinn, bins, q_hist], ['hti', 'tinn',
                                                        'bins', 'q_hist'])

    return out



def detrend_window(rri, win_len=2000, **kwargs):
    """Facilitates RRI detrending method using a signal window.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    win_len : int, optional
        Length of the window to detrend the RRI signal. Default: 2000.
    kwargs : dict, optional
        Parameters of the detrending method.

    Returns
    -------
    rri_det : array
        Detrended RRI signal.
    rri_trend : array
        Trend of the RRI signal.

    """

    # check input type
    win_len = int(win_len)

    # extract parameters
    smoothing_factor = kwargs['smoothing_factor'] if 'smoothing_factor' in kwargs else 500

    # detrend signal
    if len(rri) > win_len:
        # split the signal
        splits = int(len(rri)/win_len)
        rri_splits = np.array_split(rri, splits)

        # compute the detrended signal for each split
        rri_det = []
        for split in rri_splits:
            split_det = st.detrend_smoothness_priors(split, smoothing_factor)['detrended']
            rri_det.append(split_det)

        # concantenate detrended splits
        rri_det = np.concatenate(rri_det)
        rri_trend = None
    else:
        rri_det, rri_trend = st.detrend_smoothness_priors(rri, smoothing_factor)

    # output
    out = utils.ReturnTuple([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    return out



def sample_entropy(rri, m=2, r=0.2):
    """Computes the sample entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    sampen :  float
        Sample entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Sample_entropy
    """

    # redefine r
    r = r * rri.std()

    n = len(rri)

    # Split time series and save all templates of length m
    xmi = np.array([rri[i: i + m] for i in range(n - m)])
    xmj = np.array([rri[i: i + m] for i in range(n - m + 1)])

    # Save all matches minus the self-match, compute B
    b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([rri[i: i + m] for i in range(n - m + 1)])

    a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(a / b)



def approximate_entropy(rri, m=2, r=0.2):
    """Computes the approximate entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    appen :  float
        Approximate entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Approximate_entropy
    """

    # redefine r
    r = r * rri.std()

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[rri[j] for j in range(i, i + m - 1 + 1)] for i in range(n - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (n - m + 1.0)
            for x_i in x
        ]
        return (n - m + 1.0) ** (-1) * sum(np.log(C))

    n = len(rri)

    return _phi(m) - _phi(m + 1)


import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def plot_signal_with_peaks(indices, values, start, end, filename):
    """
    Plots a signal with specified peaks within a given range and saves the plot as a PDF.

    Parameters:
    indices (list of int): Indices of the peaks.
    values (list of int or float): Signal values.
    start (int): Start index of the range to plot.
    end (int): End index of the range to plot.
    filename (str): Name of the PDF file to save the plot.

    The signal is plotted as a pink line, and the peaks are marked as blue crosses.
    """
    # Extract the specified range
    range_indices = list(range(start, end + 1))
    range_values = values[start:end + 1]
    
    # Adjust peak indices for the specified range
    range_peaks = [i for i in indices if start <= i <= end]
    plt.figure(figsize=(40,20))
    # Signal plotten
    plt.plot(range_indices, range_values, color='pink', label='Signal')

    # Peaks plotten
    plt.scatter(range_peaks, [values[i] for i in range_peaks], color='blue', marker='x', label='Peaks')

    # Labels und Legende hinzufügen
    plt.xlabel('Index')
    plt.ylabel('Wert')
    plt.title('Signal mit Onsets')
    plt.legend()

    # Plot speichern als PDF
    plt.savefig(filename +'.pdf', format='pdf')


def ppg(signal=None, sampling_rate=64., units=None, show=True,path:Optional[str]=None, name:Optional[str]=None, ftype:Optional[str]='butter', band:Optional[str]='bandpass',order:Optional[int]=4, cutoff:Optional[list[2]]=[1,8]):
    """Process a raw PPG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    units : str, optional
        The units of the input signal. If specified, the plot will have the
        y-axis labeled with the corresponding units.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered PPG signal.
    peaks : array
        Indices of PPG pulse peaks.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    # filtered, _, _ = st.filter_signal(signal=signal,
    #                                   ftype=ftype,
    #                                   band=band,
    #                                   order=order,
    #                                   frequency=cutoff,
    #                                   sampling_rate=sampling_rate, rs=20)
    import scipy as sc
    #scipy.signal.cheby2(N, rs, Wn, btype='low', analog=False, output='ba', fs=None)
    # a,b = sc.signal.cheby2(4, rs=20,Wn=[0.50001, 12.0], btype='bandpass',fs=sampling_rate)
    # filtered = sc.signal.filtfilt(a,b, signal)
    sos = sc.signal.butter(3, Wn=[0.5, 8], btype ='bandpass', fs=sampling_rate, output='sos')
    filtered=sc.signal.sosfiltfilt(sos, signal)


    # find peaks
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(50,20))
    # plt.plot(filtered[:3*64*60])
    # plt.show()
    filtered = filtered[~np.isnan(filtered)]
    filtered = filtered[~np.isinf(filtered)]
    print(filtered)
    peaks, _ = find_onsets_elgendi2013(signal=filtered,
                                       sampling_rate=sampling_rate)
    #import neurokit as nk
    # peaks = nk.
    # extract templates
    #peaks =  nk.ppg_peaks(ppg, sampling_rate=sampling_rate, method="elgendi", show=True)
    print(peaks)
    peaks = peaks[~np.isinf(peaks)]
    peaks = peaks[~np.isnan(peaks)]
    onsets, peaks, segments_loc = ppg_segmentation(filtered, sampling_rate,
                                                   peaks)
    plot_signal_with_peaks(onsets, filtered,0, 7680*2, 'onsets')
    templates_ts, templates = _extract_templates(filtered, sampling_rate,
                                                 onsets, peaks, segments_loc)
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    # compute heart rate
    try:
        hr_idx, hr = st.get_heart_rate(beats=onsets,
                                    sampling_rate=sampling_rate,
                                    smooth=True,
                                    size=3)

        # get time vectors
        
        ts_hr = ts[hr_idx]

        # plot
        if show:
            if not name: name = "summary_plot_ppg.pdf"
            if '.' in name: pass
            plotting.plot_ppg(ts=ts,
                            raw=signal,
                            filtered=filtered,
                            peaks=peaks,
                            templates_ts=templates_ts,
                            templates=templates,
                            heart_rate_ts=ts_hr,
                            heart_rate=hr,
                            units=units,
                            path=path,
                            savingformat=name,
                            show=False)
            
            if name:
                if not path: path = os.path.curdir()
                name = name + '_summary_ppg.pdf'
                plt.savefig(path+'/'+name, bbox_inches='tight')
                plt.close('all')

        # output
        args = (ts, filtered, peaks, templates_ts, templates,
                ts_hr, hr)
        names = ('ts', 'filtered', 'peaks', 'templates_ts', 'templates',
                'heart_rate_ts', 'heart_rate')
    except (ValueError, TypeError, AttributeError) as e:
        print('Heart rate could not be calculated')
        # plot
        # kann nicht plotten weil sonst value error 
        # if show:
        #     plotting.plot_ppg(ts=ts,
        #                     raw=signal,
        #                     filtered=filtered,
        #                     peaks=peaks,
        #                     templates_ts=templates_ts,
        #                     templates=templates,
        #                     units=units,
        #                     path=name,
        #                     show=False)
            
        #     if name:
        #         name = name + '_induvidual_rri.pdf'
        #         plt.savefig(name, bbox_inches='tight')
        #         plt.close('all')

        # output
        args = (ts, filtered, peaks, templates_ts,templates)
        names = ('ts', 'filtered', 'peaks', 'templates_ts','templates')    

    return utils.ReturnTuple(args, names)



def find_onsets_elgendi2013(signal=None, sampling_rate=1000., peakwindow=0.111, beatwindow=0.667, beatoffset=0.02,
                            mindelay=0.3):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peakwindow : float
        Parameter W1 on referenced article
        Optimized at 0.111
    beatwindow : float
        Parameter W2 on referenced article
        Optimized at 0.667
    beatoffset : float
        Parameter beta on referenced article
        Optimized at 0.2
    mindelay : float
        Minimum delay between peaks.
        Avoids false positives

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    params : dict
        Input parameters of the function


    References
    ----------
    - Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions.
    PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585.
    
    Notes
    ---------------------
    Optimal ranges for signal filtering (from Elgendi et al. 2013):
    "Optimization of the beat detector’s spectral window for the lower frequency resulted in a 
    value within 0.5– 1 Hz with the higher frequency within 7–15 Hz"
    
    All the number references below between curly brackets {...} by the code refer to the line numbers of
    code in "Table 2 Algorithm IV: DETECTOR (PPG signal, F1, F2, W1, W2, b)" from Elgendi et al. 2013 for a
    better comparison of the algorithm
    
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # Create copy of signal (not to modify the original object)
    signal_copy = np.copy(signal)

    # Truncate to zero and square
    # {3, 4}
    signal_copy[signal_copy < 0] = 0
    squared_signal = signal_copy ** 2

    # Calculate peak detection threshold
    # {5}
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_peak_kernel)

    # {6}
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_beat_kernel)

    # Calculate threshold value
    # {7, 8, 9}
    thr1 = ma_beat + beatoffset * np.mean(squared_signal)

    # Identify start and end of PPG waves.
    # {10-16}
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    # {18}
    min_len = int(np.rint(peakwindow * sampling_rate))
    min_delay = int(np.rint(mindelay * sampling_rate))
    onsets = [0]

    # {19}
    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        # {20, 22, 23}
        if len_wave < min_len:
            continue

        # Find local maxima and their prominence within wave span.
        # {21}
        data = signal_copy[beg:end]
        locmax, props = ss.find_peaks(data, prominence=(None, None))

        # If more than one peak
        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between onsets.
            if peak - onsets[-1] > min_delay:
                onsets.append(peak)

    onsets.pop(0)
    onsets = np.array(onsets, dtype='int')

    # output
    params = {'signal': signal, 'sampling_rate': sampling_rate, 'peakwindow': peakwindow, 'beatwindow': beatwindow, 'beatoffset': beatoffset, 'mindelay': mindelay}

    args = (onsets, params)
    names = ('onsets', 'params')

    return utils.ReturnTuple(args, names)



def find_onsets_kavsaoglu2016(
    signal=None,
    sampling_rate=1000.0,
    alpha=0.2,
    k=4,
    init_bpm=90,
    min_delay=0.6,
    max_BPM=150,
):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    alpha : float, optional
        Low-pass filter factor.
        Avoids abrupt changes of BPM.
    k : int, float, optional
        Number of segments by pulse.
        Width of each segment = Period of pulse according to current BPM / k
    init_bpm : int, float, optional
        Initial BPM.
        Higher value results in a smaller segment width.
    min_delay : float
        Minimum delay between peaks as percentage of current BPM pulse period.
        Avoids false positives
    max_bpm : int, float, optional
        Maximum BPM.
        Maximum value accepted as valid BPM.

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    window_marks : array
        Indices of segments window boundaries.
    params : dict
        Input parameters of the function


    References
    ----------
    - Kavsaoğlu, Ahmet & Polat, Kemal & Bozkurt, Mehmet. (2016). An innovative peak detection algorithm for
    photoplethysmography signals: An adaptive segmentation method. TURKISH JOURNAL OF ELECTRICAL ENGINEERING
    & COMPUTER SCIENCES. 24. 1782-1796. 10.3906/elk-1310-177.

    Notes
    ---------------------
    This algorithm is an adaption of the one described on Kavsaoğlu et al. (2016).
    This version takes into account a minimum delay between peaks and builds upon the adaptive segmentation
    by using a low-pass filter for BPM changes. This way, even if the algorithm wrongly detects a peak, the
    BPM value will stay relatively constant so the next pulse can be correctly segmented.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if alpha <= 0 or alpha > 1:
        raise TypeError("The value of alpha must be in the range: ]0, 1].")

    if k <= 0:
        raise TypeError("The number of divisions by pulse should be greater than 0.")

    if init_bpm <= 0:
        raise TypeError("Provide a valid BPM value for initial estimation.")

    if min_delay < 0 or min_delay > 1:
        raise TypeError(
            "The minimum delay percentage between peaks must be between 0 and 1"
        )

    if max_BPM >= 248:
        raise TypeError("The maximum BPM must assure the person is alive")

    # current bpm
    bpm = init_bpm

    # current segment window width
    window = int(sampling_rate * (60 / bpm) / k)

    # onsets array
    onsets = []

    # window marks array - stores the boundaries of each segment
    window_marks = []

    # buffer for peak indices
    idx_buffer = [-1, -1, -1]

    # buffer to store the previous 3 values for onset detection
    min_buffer = [0, 0, 0]

    # signal pointer
    i = 0
    while i + window < len(signal):
        # remove oldest values
        idx_buffer.pop(0)
        min_buffer.pop(0)

        # add the index of the minimum value of the current segment to buffer
        idx_buffer.append(int(i + np.argmin(signal[i : i + window])))

        # add the minimum value of the current segment to buffer
        min_buffer.append(signal[idx_buffer[-1]])

        if (
            # the buffer has to be filled with valid values
            idx_buffer[0] > -1
            # the center value of the buffer must be smaller than its neighbours
            and (min_buffer[1] < min_buffer[0] and min_buffer[1] <= min_buffer[2])
            # if an onset was previously detected, guarantee that the new onset respects the minimum delay, minimum BPM and maximum BPM
            and (
                len(onsets) == 0
                or (
                    (idx_buffer[1] - onsets[-1]) / sampling_rate >= min_delay * 60 / bpm
                    and (idx_buffer[1] - onsets[-1]) / sampling_rate > 60 / max_BPM
                )
            )
        ):
            # store the onset
            onsets.append(idx_buffer[1])

            # if more than one onset was detected, update the bpm and the segment width
            if len(onsets) > 1:
                # calculate new bpm from the latest two onsets
                new_bpm = int(60 * sampling_rate / (onsets[-1] - onsets[-2]))

                # update the bpm value
                bpm = alpha * new_bpm + (1 - alpha) * bpm

                # update the segment window width
                window = int(sampling_rate * (60 / bpm) / k)

        # update the signal pointer
        i += window

        # store window segment boundaries index
        window_marks.append(i)

    onsets = np.array(onsets, dtype="int")
    window_marks = np.array(window_marks, dtype="int")

    # output
    params = {
        "signal": signal,
        "sampling_rate": sampling_rate,
        "alpha": alpha,
        "k": k,
        "init_bpm": init_bpm,
        "min_delay": min_delay,
        "max_bpm": max_BPM,
    }

    args = (onsets, window_marks, params)
    names = ("onsets", "window_marks", "params")

    return utils.ReturnTuple(args, names)



def ppg_segmentation(signal=None,
                     sampling_rate=1000.,
                     peaks=None,
                     selection=False,
                     peak_threshold=None):
    """Segments a filtered PPG signal. Segmentation filtering is achieved by
    taking into account segments selected by peak height and pulse morphology.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array
        List of PPG systolic peaks.
    selection : bool, optional
        If True, performs selection with peak height and pulse morphology.
    peak_threshold : int, float, optional
        If `selection` is True, selects peaks with height greater than defined
        threshold.

    Returns
    -------
    onsets : array
        Indices of PPG pulse onsets (i.e., start of beats) of the selected
        segments.
    peaks : array
        List of PPG systolic peaks of the selected segments.
    segments_loc : array
        Start and end indices for each selected pulse segment.

    """

    # check inputs
    if signal is None or peaks is None:
        raise TypeError("Please check inputs.")

    # ensure input format
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)

    # find onsets
    onsets = []
    minima = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0]
    
    missed_onsets = []
    for ind in peaks:
        relevant_minima = minima[minima < ind]
        if relevant_minima.size > 0:
            onsets.append(relevant_minima.max())
        else: missed_onsets.append(ind)    
    onsets = np.array(onsets, dtype='int')

    # raise error if onset detection failed
    if len(onsets) == 0:
        raise TypeError("No onsets detected.")

    # assign start and end of each segment
    segments_loc = np.vstack((onsets[:-1], onsets[1:])).T

    # segment selection by morphology
    if selection:
        segments_sel = []
        for ind in range(segments_loc.shape[0]):
            # search segments with at least 4 max+min (standard waveform)
            segment = signal[segments_loc[ind, 0]: segments_loc[ind, 1]]
            if len(np.where(np.diff(np.sign(np.diff(segment))))[0]) >= 4:
                segments_sel.append(ind)

        segments_loc = segments_loc[segments_sel]
        onsets = onsets[segments_sel]
        peaks = peaks[segments_sel]

    # segment selection by height
    if peak_threshold is not None:
        segments_sel = []
        for ind in range(segments_loc.shape[0]):
            # search segments with peak higher than threshold
            segment = signal[segments_loc[ind, 0]: segments_loc[ind, 1]]
            if max(segment) > peak_threshold:
                segments_sel.append(ind)

        segments_loc = segments_loc[segments_sel]
        onsets = onsets[segments_sel]
        peaks = peaks[segments_sel]
    if not len(peaks) == len(onsets):
        if len(missed_onsets)>0:
            
            for i in missed_onsets:
                peaks.tolist().remove(i)
    # output
    args = (onsets, peaks, segments_loc)
    names = ('onsets', 'peaks', 'segments_loc')

    return utils.ReturnTuple(args, names)



def _extract_templates(signal=None,
                      sampling_rate=1000.,
                      onsets=None,
                      peaks=None,
                      segments_loc=None):
    """Extracts the templates from the PPG signal, which are aligned with their
    systolic peaks. To achieve this, the segments are padded with NaNs. Should
    be used in combination with signals.ppg.ppg_segmentation.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    onsets : array
        List of onsets (i.e., start of beats) of the PPG waves.
    peaks : array
        List of PPG systolic peaks.
    segments_loc : array
        Start and end indices for each selected pulse segment.

    Returns
    -------
    templates_ts : array
        Time axis common to all templates.
    templates : array
        List of templates aligned with the systolic peaks.

    """
    # initialize output
    templates = []

    # find the longest onset-peak duration
    try:
        shifts = peaks - onsets
        max_shift = np.max(peaks - onsets)
    except ValueError:
        peaks = peaks[-len(onsets):]
        shifts = peaks - onsets
        max_shift = np.max(peaks - onsets)
    
    # left padding
    max_len = 0
    for i in range(len(segments_loc)):
        segment = signal[segments_loc[i, 0]: segments_loc[i, 1]]
        segment = np.pad(segment, max_shift - shifts[i], mode='constant',
                         constant_values=(np.nan,))
        templates.append(segment)

        # find the largest segment
        if len(segment) > max_len:
            max_len = len(segment)

    # right padding
    for index, segment in enumerate(templates):
        templates[index] = np.pad(segment, (0, max_len - len(segment)),
                                  mode='constant', constant_values=(np.nan,))

    templates = np.asarray(templates).T

    # time vector
    templates_ts = np.arange(-max_shift, max_len - max_shift) / sampling_rate

    # output
    args = (templates_ts, templates)
    names = ('templates_ts', 'templates')

    return utils.ReturnTuple(args, names)