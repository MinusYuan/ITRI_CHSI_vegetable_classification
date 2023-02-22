#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from spectral import *
from .VI_module import *
from scipy.signal import savgol_filter
import pickle


def hyper(path_hdr,path_bsq):

    def open_file_as_array(path_hdr,path_bsq):
        hdr1 = envi.open(path_hdr,path_bsq)
        #view1 = imshow(hdr1, (114, 69, 20), aspect='auto')
        hdr1_np = np.array(hdr1.load())
        return hdr1_np

    plt.rcParams["figure.figsize"] = (8,6)
    c0 = open_file_as_array(path_hdr,path_bsq)
    df4 = c0.astype(np.float16)

    def reshape_wave(df0):
        wave=[]
        for x in range(0,df0.shape[0],1):
            for y in range(0,df0.shape[1],1):
                    wave_length = df0[x,y,:]
                    wave.append(wave_length)
        return wave
    def raw_to_deriv(x, deriv):
        derivative = savgol_filter(x, 3, 2,deriv,mode='nearest')
        return derivative

    wave_all = np.array(reshape_wave(df4))
    wavelength_wave = np.array(pd.DataFrame(pd.read_excel("welcome/HyperSpectralAnalysis_VegationIndex/band_name.xlsx")).columns.str.replace('nm', '').astype('float64'))

##################################################NDVI####################################################################

    RSR_red_fileName = 'welcome/HyperSpectralAnalysis_VegationIndex/ReflectanceSpectralResponse_I1_red_SNPP_VIIRS.csv'
    RSR_red_wave, RSR_red = load_RSR_data(RSR_red_fileName, 2)
    RSR_red_interp = interp1d(RSR_red_wave, RSR_red, kind = 'cubic', bounds_error=False, fill_value=0)

    RSR_IR_fileName = 'welcome/HyperSpectralAnalysis_VegationIndex/ReflectanceSpectralResponse_I2_IR_SNPP_VIIRS.csv'
    RSR_IR_wave, RSR_IR = load_RSR_data(RSR_IR_fileName, 2)
    RSR_IR_interp = interp1d(RSR_IR_wave, RSR_IR, kind = 'cubic', bounds_error=False, fill_value=0)


    NDVI_value = np.apply_along_axis(NDVI, 1, wave_all[:, :], RSR_red_interp(wavelength_wave), RSR_IR_interp(wavelength_wave))
    #remove the value with nan (if IR == red, the calculated NDVI value would be nan)
    NDVI_value[np.argwhere(np.isnan(NDVI_value))] = 0
    #############################################SG_Filter####################################################################

    wave_all_zero_deriv = raw_to_deriv(wave_all, deriv=0)
    wave_all_first_deriv = raw_to_deriv(wave_all, deriv=1)
    wave_all_first_deriv_concat = np.column_stack([wave_all_first_deriv,NDVI_value])

    band_name = pd.DataFrame(pd.read_excel("welcome/HyperSpectralAnalysis_VegationIndex/band_name_NDVI.xlsx")).columns
    df_test = pd.DataFrame(wave_all_first_deriv_concat,columns=band_name)
    X = df_test.iloc[:,:].values
################################################Call_Models###############################################################
    with open('welcome/RF_20210713.pickle', 'rb') as file0:
        classifier_rf =pickle.load(file0,  encoding='iso-8859-1')
#with open('XGB_20210713.pickle', 'rb') as file1:
#    classifier_xgb =pickle.load(file1,  encoding='iso-8859-1')
##############################################prediction##################################################################
    y_pred_proba_RF = classifier_rf.predict_proba(X)
    y_pred_RF = classifier_rf.predict(X)

    pred_RF = np.reshape(y_pred_RF, (df4.shape[0],df4.shape[1]))

    proba_RF0 = np.reshape(y_pred_proba_RF[:,0], (df4.shape[0],df4.shape[1]))


    proba_RF1 = np.reshape(y_pred_proba_RF[:,1], (df4.shape[0],df4.shape[1]))

    proba_RF2 = np.reshape(y_pred_proba_RF[:,2], (df4.shape[0],df4.shape[1]))
##########################################################################################################################
    proba_RF2_dust = np.interp(proba_RF2, (proba_RF2.min(), proba_RF2.max()), (0, 1))
    proba_RF1_healthy = np.interp(proba_RF1, (proba_RF1.min(), proba_RF1.max()), (0, 1))
    proba_RF0_unhealthy = np.interp(proba_RF0, (proba_RF0.min(), proba_RF0.max()), (0, 1))
##########################################################################################################################
#parameters
    a0 = (proba_RF1_healthy+proba_RF0_unhealthy)-2*proba_RF2_dust
    a1 = (proba_RF1_healthy*proba_RF0_unhealthy)-proba_RF2_dust*proba_RF2_dust
##################################################################################################
    proba_array_c=[]
    for x in range(0,proba_RF1.shape[0],1):
        for y in range(0,proba_RF1.shape[1],1):
            if (proba_RF1_healthy[x,y] >0.4) & (proba_RF1_healthy[x,y] <0.8) & (proba_RF2_dust[x,y] <0.03) & (proba_RF0_unhealthy[x,y] <0.35) & (proba_RF0_unhealthy[x,y] >0.2):
                proba_array_c.append(255)          
            else:
                proba_array_c.append(0)
            
    proba_array_new = np.reshape(proba_array_c, (df4.shape[0],df4.shape[1]))   
##########################################################################################################
    proba_healthy_without_dust0 = np.interp(a0, (a0.min(), a0.max()), (0, 1))
    proba_healthy_without_dust1 = np.interp(a1, (a1.min(), a1.max()), (0, 1))
    proba_healthy_without_dust0 = np.round(proba_healthy_without_dust0,2)
    proba_healthy_without_dust1 = np.round(proba_healthy_without_dust1,2)
    pre1_sum = (((0 < proba_healthy_without_dust0) & (proba_healthy_without_dust0 < 0.2)).sum() / (((0 < proba_healthy_without_dust0) & (proba_healthy_without_dust0 < 0.2)).sum() + (((0.2 < proba_healthy_without_dust0) & (proba_healthy_without_dust0 < 1)).sum())/2))*100
    pre2_sum = (((0 < proba_healthy_without_dust1) & (proba_healthy_without_dust1 < 0.2)).sum() / (((0 < proba_healthy_without_dust1) & (proba_healthy_without_dust1 < 0.2)).sum() + (((0.2 < proba_healthy_without_dust1) & (proba_healthy_without_dust1 < 1)).sum())/2))*100
    ndvi_sum = (((0 < NDVI_value) & (NDVI_value < 0.2)).sum() / (((0 < NDVI_value) & (NDVI_value < 0.2)).sum() + (((0.2 < NDVI_value) & (NDVI_value < 1)).sum())/2))*100
    pre1_sum = round(pre1_sum,2)
    pre2_sum = round(pre2_sum,2)
    ndvi_sum = round(ndvi_sum,2)
############################################################################################################
############################################################################################################
    fig =plt.subplots(figsize=(24,16))
    plt.title('healthy without dust I', fontsize=20)
    plt.imshow(proba_healthy_without_dust0,cmap = plt.cm.nipy_spectral_r)
    plt.colorbar()
    #plt.grid()
    plt.savefig('welcome/static/images/healthy_without_dust_I.png')
#############################################################################################################
    fig =plt.subplots(figsize=(24,16))
    plt.title('healthy without dust II', fontsize=20)
    plt.imshow(proba_healthy_without_dust1,cmap = plt.cm.nipy_spectral_r)
    plt.colorbar()
    plt.savefig('welcome/static/images/healthy_without_dust_II.png')
###############################show_NDVI#####################################################################
    fig =plt.subplots(figsize=(24,16))
    plt.title('NDVI', fontsize=20)
    plt.imshow(NDVI_value.reshape(df4.shape[0],df4.shape[1]),cmap = plt.cm.nipy_spectral_r)
    plt.colorbar()
    plt.savefig('welcome/static/images/NDVI.png')

    return pre1_sum,pre2_sum,ndvi_sum
