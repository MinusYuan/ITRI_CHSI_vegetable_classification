import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import pandas as pd
import math
import time
from scipy.interpolate import interp1d

def number2nm(number, original_degree):
    return (1/number)*pow(10,9)/pow(10,original_degree)
    
def load_spectral_data(fileName):
    with open(fileName, 'rb') as f:
        spectral_data = pickle.load(f)
    return spectral_data

def load_RSR_data(fileName, wavenum_degree):
    RSR_pd = pd.read_csv(fileName)
    RSR_np_array = np.array(RSR_pd)
    RSR_wave = RSR_np_array[:, 0]
    with np.nditer(RSR_wave, op_flags=['readwrite']) as wavenums:
        for wavenum in wavenums:
            wavenum[...] = number2nm(wavenum, wavenum_degree)
    RSR = RSR_np_array[:, 1]
    return RSR_wave, RSR

def NDVI(spectral, RSR_red, RSR_IR):
    red = np.inner(spectral, RSR_red)
    IR = np.inner(spectral, RSR_IR)
    NDVI_value = (IR - red)/(IR + red)
    return NDVI_value