"""

This module contains the peak and onset detection functionality adapted from
the `pyPPG` library.

Only the peak–onset detection algorithm is used within the processing pipeline.
All additional functions included in this module are internal helper functions
that are called exclusively by the peak detection routine.

The original algorithmic content of the peak–onset method remains unchanged.
Adaptations were limited to structural and compatibility-related modifications
to allow integration into the existing pipeline, as the original `pyPPG`
library was incompatible with the Python version used in this project.


"""

import copy
import pandas as pd
import numpy as np

from scipy.signal import kaiserord, firwin, filtfilt, detrend, periodogram, lfilter, find_peaks, firls, resample
from scipy import signal
import dotmap
from dotmap import DotMap
def get_peak_onset( peak_detector='PPGdet', ppg:pd.Series=None, fs:int=64):
        '''PPGdet detects beats in a photoplethysmogram (PPG) signal
        using the improved 'Automatic Beat Detection' of Aboy M et al.

        :param peak_detector: type of peak detector
        :type peak_detector: str

        :return:
            - peaks, 1-d array: indices of detected systolic peaks
            - onsets, 1-d array: indices of detected pulse onsets

        Reference
        ---------
        Aboy M et al., An automatic beat detection algorithm for pressure signals.
        IEEE Trans Biomed Eng 2005; 52: 1662 - 70. <https://doi.org/10.1109/TBME.2005.855725>

        Author:
        Marton A. Goda – Faculty of Biomedical Engineering,
        Technion – Israel Institute of Technology, Haifa, Israel (August 2022)

        Original Matlab implementation:
        Peter H. Charlton – King's College London (August 2017) – University of Cambridge (February 2022)
        <https://github.com/peterhcharlton/ppg-beats>

        Changes from Charlton's implementation:
            1) Detect Maxima:
                *  Systolic peak-to-peak distance is predicted by the heart rate estimate over the preceding 10 sec window.
                *  The peak location is estimated by distances and prominences of the previous peaks.
            2) Find Onsets:
                *  The onset is a local minimum, which is always calculated from the peak that follows it within a given time window
            3) Tidy of Peaks and Onsets:
                *  There is a one-to-one correspondence between onsets and peaks
                *  There are only onset and peak pairs
                *  The distance between the onset and peak pairs can't be smaller than 30 ms
            4) Correct Peaks and Onsets:
                * The Peaks must be the highest amplitude between two consecutive pulse onsets, if not, then these are corrected
                * After the correction of Peaks, the Onsets are recalculated
        '''

        # inputs
        x = copy.deepcopy(ppg)                    #signal
        fso=fs
        fs = 75
        x = resample(x, int(len(ppg)*(fs/fso)))
        up = set_beat_detection()                 #settings
        win_sec=10
        w = fs * win_sec                                    #window length(number of samples)
        win_starts = np.array(list(range(0,len(x),round(0.8*w))))
        win_starts = win_starts[0:min(np.where([win_starts >= len(x) - w])[1])]
        win_starts = np.insert(win_starts,len(win_starts), len(x) + 1 - w)

        # before pre-processing
        hr_win=0  #the estimated systolic peak-to-peak distance, initially it is 0
        hr_win_v=[]
        px = detect_maxima(x, 0, hr_win, peak_detector) # detect all maxima
        if len(px)==0:
            peaks = []
            onsets = []
            return peaks, onsets

        # detect peaks in windows
        all_p4 = []
        all_hr = np.empty(len(win_starts)-1)
        all_hr [:] = np.nan
        hr_past = 0 # the actual heart rate
        hrvi = 0    # heart rate variability index

        for win_no in range(0,len(win_starts) - 1):
            curr_els = range(win_starts[win_no],win_starts[win_no] + w)
            curr_x = x[curr_els]

            y1 = def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 3 * up.fh_hz)   # Filter no.1
            hr = estimate_HR(y1, fs, up, hr_past)               # Estimate HR from weakly filtered signal
            hr_past=hr
            all_hr[win_no] = hr

            if (peak_detector=='PPGdet') and (hr>40):
                if win_no==0:
                    p1 = detect_maxima(y1, 0, hr_win, peak_detector)
                    tr = np.percentile(np.diff(p1), 50)
                    pks_diff = np.diff(p1)
                    pks_diff = pks_diff[pks_diff>=tr]
                    hrvi = np.std(pks_diff) / np.mean(pks_diff) * 5

                hr_win = fs / ((1 + hrvi) * 3)
                hr_win_v.append(hr_win)
            else:
                hr_win=0

            y2 = def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 2.5 * hr / 60)           # Filter no. 2
            y2_deriv = estimate_deriv(y2)                                          # Estimate derivative from highly filtered signal
            p2 = detect_maxima(y2_deriv, up.deriv_threshold,hr_win, peak_detector) # Detect maxima in derivative
            y3 = def_bandpass(curr_x, fs, 0.9 * up.fl_hz, 10 * hr / 60)
            p3 = detect_maxima(y3, 50, hr_win, peak_detector)                      # Detect maxima in moderately filtered signal
            p4 = find_pulse_peaks(p2, p3)
            p4 = np.unique(p4)

            if peak_detector=='PPGdet':
                if len(p4)>round(win_sec/2):
                    pks_diff = np.diff(p4)
                    tr = np.percentile(pks_diff, 30)
                    pks_diff = pks_diff[pks_diff >= tr]

                    med_hr=np.median(all_hr[np.where(all_hr>0)])
                    if ((med_hr*0.5<np.mean(pks_diff)) and (med_hr*1.5<np.mean(pks_diff))):
                        hrvi = np.std(pks_diff) / np.mean(pks_diff)*10

            all_p4 = np.concatenate((all_p4, win_starts[win_no] + p4), axis=None)

        all_p4=all_p4.astype(int)
        all_p4 = np.unique(all_p4)

        peaks, fn = correct_IBI(all_p4, px, np.median(all_hr), fs, up)

        peaks = (all_p4/fs*fso).astype(int)
        onsets, peaks = find_onsets(ppg, fso, up, peaks,60/np.median(all_hr)*fs)

        # Correct Peaks
        for i in range(0, len(peaks) - 1):
            max_loc = np.argmax(ppg[onsets[i]:onsets[i + 1]]) + onsets[i]
            if peaks[i] != max_loc:
                peaks[i] = max_loc

        # Correct onsets
        onsets, peaks = find_onsets(ppg, fso, up, peaks, 60 / np.median(all_hr) * fs)

        temp_i = np.where(np.diff(onsets) == 0)[0]
        if len(temp_i) > 0:
            peaks = np.delete(peaks, temp_i)
            onsets = np.delete(onsets, temp_i)

        temp_i = np.where((peaks - onsets) < fso / 30)[0]
        if len(temp_i) > 0:
            peaks = np.delete(peaks, temp_i)
            onsets = np.delete(onsets, temp_i)

        return peaks, onsets

    ###########################################################################
    ############################# Maximum detector ############################
    ###########################################################################
def detect_maxima( sig: np.array, percentile: int ,hr_win: int, peak_detector: str):
        #Table VI pseudocode
        """
        Detect Maxima function detects all peaks in the raw and also in the filtered signal to find.

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param percentile: in each signal partition, a rank filter detects the peaks above a given percentile
        :type percentile: int
        :param hr_win: window for adaptive the heart rate estimate
        :type hr_win: int
        :param peak_detector: type of peak detector
        :type peak_detector: str

        :return: maximum peaks of signal, 1-d array.

        """

        tr = np.percentile(sig, percentile)

        if peak_detector=='ABD':

            s1,s2,s3 = sig[2:], sig[1:-1],sig[0:-2]
            m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
            max_pks = m[sig[m] > tr]

        if peak_detector=='PPGdet':
            s1,s2,s3 = sig[2:], sig[1:-1],sig[0:-2]

            max_loc = []
            min_loc = []
            max_pks=[]
            intensity_v = []
            if hr_win == 0:
                m = 1 + np.array(np.where((s1 < s2) & (s3 < s2)))
                max_pks = m[sig[m] > tr]
            else:
                max_loc = find_peaks(sig, distance=hr_win)[0]
                min_loc = find_peaks(-sig, distance=hr_win)[0]

                for i in range(0,len(max_loc)):
                    values = abs(max_loc[i] - min_loc)
                    min_v = min(values)
                    min_i = np.where(min_v==values)[0][0]
                    intensity_v.append(sig[max_loc[i]] - sig[min_loc[min_i]])

                # possible improvements:
                #   - using adaptive thresholding for the maximum
                #   - estimate probability density of the maximum

                tr2 = np.mean(intensity_v)*0.25
                max_pks = find_peaks(sig+min(sig),prominence=tr2,distance=hr_win)[0]

        return max_pks

    ###########################################################################
    ############################ Bandpass filtering ###########################
    ###########################################################################
def def_bandpass( sig: np.array, fs: int, lower_cutoff: float, upper_cutoff: float):
        """
        def_bandpass filter function detects all peaks in the raw and also in the filtered signal to find.

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param fs: sampling frequency
        :type fs: int
        :param lower_cutoff: lower cutoff frequency
        :type lower_cutoff: float
        :param upper_cutoff: upper cutoff frequency
        :type upper_cutoff: float

        :return: bandpass filtered signal, 1-d array.

        """

        # Filter characteristics: Eliminate VLFs (below resp freqs): For 4bpm cutoff
        up = DotMap()
        up.paramSet.elim_vlf.Fpass = 1.3*lower_cutoff   #in Hz
        up.paramSet.elim_vlf.Fstop = 0.8*lower_cutoff   #in Hz
        up.paramSet.elim_vlf.Dpass = 0.05
        up.paramSet.elim_vlf.Dstop = 0.01

        # Filter characteristics: Eliminate VHFs (above frequency content of signals)
        up.paramSet.elim_vhf.Fpass = 1.2*upper_cutoff   #in Hz
        up.paramSet.elim_vhf.Fstop = 0.8*upper_cutoff   #in Hz
        up.paramSet.elim_vhf.Dpass = 0.05
        up.paramSet.elim_vhf.Dstop = 0.03

        # perform BPF
        s = DotMap()
        s.v = sig
        s.fs = fs

        b, a = signal.iirfilter(5, [2 * np.pi * lower_cutoff, 2 * np.pi * upper_cutoff], rs=60,
                                btype='band', analog=True, ftype='cheby2')

        bpf_sig = filtfilt(b, 1, sig)

        return bpf_sig


def estimate_deriv( sig: np.array, deriv_no = 1, win_size = 5):
        """
        Derivative Estimation function estimate derivative from highly filtered signal based on the
        General least-squares smoothing and differentiation by the convolution (Savitzky Golay) method

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array

        :return: derivative, 1-d array.

        """

        #Savitzky Golay
        deriv = savitzky_golay(sig, deriv_no, win_size)

        return deriv


def savitzky_golay( sig: np.array, deriv_no: int, win_size: int):
        """
        This function estimate the Savitzky Golay derivative from highly filtered signal

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param deriv_no: number of derivative
        :type deriv_no: int
        :param win_size: size of window
        :type win_size: int

        :return: Savitzky Golay derivative, 1-d array.

        """

        ##assign coefficients
        # From: https: // en.wikipedia.org / wiki / Savitzky % E2 % 80 % 93 Golay_filter  # Tables_of_selected_convolution_coefficients
        # which are calculated from: A., Gorry(1990). "General least-squares smoothing and differentiation by the convolution (Savitzky?Golay) method".Analytical Chemistry. 62(6): 570?3. doi: 10.1021 / ac00205a007.

        if deriv_no==0:
            #smoothing
            if win_size == 5:
                coeffs = [-3, 12, 17, 12, -3]
                norm_factor = 35
            elif win_size == 7:
                coeffs = [-2, 3, 6, 7, 6, 3, -2]
                norm_factor = 21
            elif win_size == 9:
                coeffs = [-21, 14, 39, 54, 59, 54, 39, 14, -21]
                norm_factor = 231
            else:
                print('Can''t do this window size')
        elif deriv_no==1:
            # first derivative
            if win_size == 5:
                coeffs = range(-2,3)
                norm_factor = 10
            elif win_size == 7:
                coeffs = range(-3,4)
                norm_factor = 28
            elif win_size == 9:
                coeffs = range(-4,5)
                norm_factor = 60
            else:
                print('Can''t do this window size')
        elif deriv_no == 2:
            # second derivative
            if win_size == 5:
                coeffs = [2, -1, -2, -1, 2]
                norm_factor = 7
            elif win_size == 7:
                coeffs = [5, 0, -3, -4, -3, 0, 5]
                norm_factor = 42
            elif win_size == 9:
                coeffs = [28, 7, -8, -17, -20, -17, -8, 7, 28]
                norm_factor = 462
            else:
                print('Can''t do this window size')
        elif deriv_no == 3:
            # third derivative
            if win_size == 5:
                coeffs = [-1, 2, 0, -2, 1]
                norm_factor = 2
            elif win_size == 7:
                coeffs = [-1, 1, 1, 0, -1, -1, 1]
                norm_factor = 6
            elif win_size == 9:
                coeffs = [-14, 7, 13, 9, 0, -9, -13, -7, 14]
                norm_factor = 198
            else:
                print('Can''t do this window size')
        elif deriv_no == 4:
            # fourth derivative
            if win_size == 7:
                coeffs = [3, -7, 1, 6, 1, -7, 3]
                norm_factor = 11
            elif win_size == 9:
                coeffs = [14, -21, -11, 9, 18, 9, -11, -21, 14]
                norm_factor = 143
            else:
                print('Can''t do this window size')
        else:
            print('Can''t do this order of derivative')


        if deriv_no % 2 == 1:
            coeffs = -np.array(coeffs)

        A = [1, 0]
        filtered_sig = lfilter(coeffs, A, sig)
        # filtered_sig = filtfilt(coeffs, A, sig)
        s = len(sig)
        half_win_size = np.floor(win_size * 0.5)
        zero_pad=filtered_sig[win_size] * np.ones(int(half_win_size))
        sig_in=filtered_sig[win_size-1:s]
        sig_end=filtered_sig[s-1] * np.ones(int(half_win_size))
        deriv = [*zero_pad,*sig_in,*sig_end]
        deriv = deriv / np.array(norm_factor)

        return deriv

    ###########################################################################
    ############################# Pulse detection #############################
    ###########################################################################
def find_pulse_peaks( p2: np.array, p3: np.array):
        """
        Pulse detection function detect the pulse peaks according to the peaks of 1st and 2nd derivatives
        General least-squares smoothing and differentiation by the convolution (Savitzky Golay) method

        :param p2: peaks of the 1st derivatives
        :type p2: 1-d array
        :param p3: peaks of the 2nd derivatives
        :type p2: 1-d array

        :return: pulse peaks, 1-d array.

        """

        p4 = np.empty(len(p2))
        p4[:] = np.nan
        for k in range(0,len(p2)):
            rel_el = np.where(p3>p2[k])
            if np.any(rel_el) and ~np.isnan(rel_el[0][0]):
                p4[k] = p3[rel_el[0][0]]

        p4 = p4[np.where(~np.isnan(p4))]
        p4 = p4.astype(int)
        return p4

    ###########################################################################
    ####################### Correct peaks' location error #####################
    ###########################################################################
def  correct_IBI( p: np.array, m: np.array, hr: float, fs: int, up: DotMap):
        """
        This function corrects the peaks' location (interbeat intervals) error

        :param p: systolic peaks of the PPG signal
        :type p: 1-d array
        :param m: all maxima of the PPG signal
        :type m: 1-d array
        :param hr: heart rate
        :type hr: float
        :param fs: sampling frequency
        :type fs: int
        :param up: setup up parameters of the algorithm
        :type up: DotMap

        :return: onsets, 1-d array.

        """

        #Correct peaks' location error due to pre-processing
        pc = np.empty(len(p))
        pc[:] = np.nan
        pc1=[]
        for k in range(0,len(p)):
            temp_pk=abs(m - p[k])
            rel_el = np.where(temp_pk==min(temp_pk))
            pc1=[*pc1,*m[rel_el]]

        # Correct false positives
        # identify FPs
        d = np.diff(pc1)/fs    # interbeat intervals in secs
        fp = find_reduced_IBIs(d, hr, up)

        # remove FPs
        pc = np.array(pc1)[fp]

        # Correct false negatives
        d = np.diff(pc)/fs    # interbeat intervals in secs
        fn = find_prolonged_IBIs(d, hr, up)

        pc = pc1

        return pc, fn

def find_reduced_IBIs(IBIs: np.array, med_hr: float, up: DotMap):
        """
        This function finds the reduced interbeat intervals

        :param IBIs: interbeat intervals in secs
        :type IBIs: 1-d array
        :param med_hr: median heart rate
        :type med_hr: float
        :param up: setup up parameters of the algorithm
        :type up: DotMap

        :return: fp, the false positive case

        """

        IBI_thresh = up.lower_hr_thresh_prop*60/med_hr
        fp = IBIs < IBI_thresh
        fp = [*np.where(fp == 0)[0].astype(int)]
        return fp

def find_prolonged_IBIs( IBIs: np.array, med_hr: float, up: DotMap):
        """
        This function finds the prolonged interbeat intervals

        :param IBIs: interbeat intervals in secs
        :type IBIs: 1-d array
        :param med_hr: median heart rate
        :type med_hr: float
        :param up: setup up parameters of the algorithm
        :type up: DotMap

        :return: fn, the false negative case

        """
        IBI_thresh = up.upper_hr_thresh_prop*60/med_hr
        fn = IBIs > IBI_thresh

        return fn

  
def find_onsets(sig: np.array, fs: int, up: DotMap, peaks: np.array, med_hr: float):
        """
        This function finds the onsets of PPG sigal

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :param fs: sampling frequency
        :type fs: int
        :param up: setup up parameters of the algorithm
        :type up: DotMap
        :param peaks: peaks of the signal
        :type peaks: 1-d array
        :param med_hr: median heart rate
        :type med_hr: float

        :return: onsets, 1-d array

        """

        Y1=def_bandpass(sig, fs, 0.9*up.fl_hz, 3*up.fh_hz)
        temp_oi0=find_peaks(-Y1,distance=med_hr*0.3)[0]

        null_indexes = np.where(temp_oi0<peaks[0])
        if len(null_indexes[0])!=0:
            if len(null_indexes[0])==1:
                onsets = [null_indexes[0][0]]
            else:
                onsets = [null_indexes[0][-1]]
        else:
            onsets = [peaks[0]-round(fs/50)]

        i=1
        while i < len(peaks):
            min_SUT=fs*0.12     # minimum Systolic Upslope Time 120 ms
            min_DT=fs*0.3       # minimum Diastolic Time 300 ms

            before_peak=temp_oi0 <peaks[i]
            after_last_onset=temp_oi0 > onsets[i - 1]
            SUT_time=peaks[i]-temp_oi0>min_SUT
            DT_time = temp_oi0-peaks[i-1]  > min_DT
            temp_oi1 = temp_oi0[np.where(before_peak * after_last_onset*SUT_time*DT_time)]
            if len(temp_oi1)>0:
                if len(temp_oi1) == 1:
                    onsets.append(temp_oi1[0])
                else:
                    onsets.append(temp_oi1[-1])
                i=i+1
            else:
                peaks = np.delete(peaks, i)

        return onsets,peaks


def set_beat_detection():
        """
        This function setups the filter parameters of the algorithm

        :return: filter parameters of the algorithm, DotMap

        """
        # plausible HR limits
        up=DotMap()
        up.fl = 30               #lower bound for HR
        up.fh = 200              #upper bound for HR
        up.fl_hz = up.fl/60
        up.fh_hz = up.fh/60

        # Thresholds
        up.deriv_threshold = 75          #originally 90
        up.upper_hr_thresh_prop = 2.25   #originally 1.75
        up.lower_hr_thresh_prop = 0.5    #originally 0.75

        # Other parameters
        up.win_size = 10    #in secs

        return up




def estimate_HR( sig: np.array, fs: int, up: DotMap, hr_past: int):
        """
        Heart Rate Estimation function estimate the heart rate according to the previous heart rate in given time window

        :param sig: array of signal with shape (N,) where N is the length of the signal
        :type sig: 1-d array
        :type fs: int
        :param up: setup up parameters of the algorithm
        :type up: DotMap
        :param hr_past: the average heart rate in the past in given time window
        :type hr_past: int

        :return: estimated heart rate, 1-d array.

        """

        # Estimate PSD
        blackman_window = np.blackman(len(sig))
        f, pxx = periodogram(sig,fs, blackman_window)
        ph = pxx
        fh = f

        # Extract HR
        if (hr_past / 60 < up.fl_hz) | (hr_past / 60 > up.fh_hz):
            rel_els = np.where((fh >= up.fl_hz) & (fh <= up.fh_hz))
        else:
            rel_els = np.where((fh >= hr_past / 60 * 0.5) & (fh <= hr_past / 60 * 1.4))

        rel_p = ph[rel_els]
        rel_f = fh[rel_els]
        max_el = np.where(rel_p==max(rel_p))
        hr = rel_f[max_el]*60
        hr = int(hr[0])

        return hr