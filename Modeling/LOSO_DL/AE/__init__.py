# -*- coding: utf-8 -*-
"""
AE
---------------

This package provides various autoencoder written in pytorch with different layers  for the feature dimensions reduction:
    * Temporal Kolomogrov-Arnold-Network (TKAN) based autoencoder
    * Long-Short-Term-Memory (LSTM) based autoencoder
    * Temporal Convolutional Network (TCN) based autoencoder
   

"""

# compat
from __future__ import absolute_import, division, print_function
from . import lstm, tkan, tcn