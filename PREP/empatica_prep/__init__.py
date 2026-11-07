# -*- coding: utf-8 -*-
"""
empatica_prep
---------------

This package provides methods to process common
physiological signals (biosignals):
    * Photoplethysmogram (PPG)
    * Electrodermal Activity (EDA)
    * Acceleration (ACC)
    * Skin temperature (TEMP)


"""

# compat
from __future__ import absolute_import, division, print_function

# allow lazy loading
from . import acc, ppg, eda, temp