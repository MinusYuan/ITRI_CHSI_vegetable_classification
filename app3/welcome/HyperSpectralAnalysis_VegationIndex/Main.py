import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import pandas as pd
from scipy.interpolate import interp1d

def main(flag_showSpectrum_RSR):
    Spectral_data_fileName = '0415_hyperSpectral_100_600_250_750_10312019_withWave.pickle'
    spectral_data = load_spectral_data(Spectral_data_fileName)
    spectral_data_nparray = spectral_data['data']
    wave_spectral = spectral_data['wave_length']
    spectral_data_line = spectral_data_nparray.reshape(250000, 274)

    RSR_red_fileName = 'ReflectanceSpectralResponse_I1_red_SNPP_VIIRS.csv'
    RSR_red_wave, RSR_red = load_RSR_data(RSR_red_fileName, 2)
    RSR_red_interp = interp1d(RSR_red_wave, RSR_red, kind = 'cubic', bounds_error=False, fill_value=0)

    RSR_IR_fileName = 'ReflectanceSpectralResponse_I2_IR_SNPP_VIIRS.csv'
    RSR_IR_wave, RSR_IR = load_RSR_data(RSR_IR_fileName, 2)
    RSR_IR_interp = interp1d(RSR_IR_wave, RSR_IR, kind = 'cubic', bounds_error=False, fill_value=0)

    ### Show spectrum of one pixel and reflectance spectral responses of red and IR 
    if flag_showSpectrum_RSR:
        fig, ax = plt.subplots(1, 3, figsize=[20,5])
        ax[0].plot(wave_spectral, spectral_data_line[0, 2:])
        ax[0].set_title('Spectrum measured at (%i,%i)' % (spectral_data_line[0, 0], spectral_data_line[0, 1]))
        ax[0].set_xlabel('wavelength (nm)')
        ax[0].set_ylabel('Relative spectrum intensity')
        ax[1].plot(wave_spectral, RSR_red_interp(wave_spectral))
        ax[1].set_title('Reflectance spectrum responses (RSR)  \n of red region (used for SNPP VIIRS)')
        ax[1].set_xlabel('wavelength (nm)')
        ax[1].set_ylabel('Spectrum responses')
        ax[2].plot(wave_spectral, RSR_IR_interp(wave_spectral))
        ax[2].set_title('Reflectance spectrum responses (RSR)  \n of IR region (used for SNPP VIIRS)')
        ax[2].set_xlabel('wavelength (nm)')
        ax[2].set_ylabel('Spectrum responses')
        plt.show()
    
    ### Calculate the NDVI calue
    NDVI_value = np.apply_along_axis(NDVI, 1, spectral_data_line[:, 2:], RSR_red_interp(wave_spectral), RSR_IR_interp(wave_spectral))
    fig, ax = plt.subplots(1, 2, figsize=[10,5])
    ax[0].imshow(spectral_data_line[:, 2].reshape(500,500))
    ax[0].axis('off')
    ax[1].imshow(NDVI_value.reshape(500, 500))
    ax[1].axis('off')
    plt.show()

main(flag_showSpectrum_RSR = False)