import numpy as np
import scipy
import pyfits
from sklearn import neural_network
#import cv2
from matplotlib import pyplot as plt

#input
pair_number = 20
url = 'rd_imdata/' #name of the folder
stat_num = 7 #std, var, max, skew, median(fft), mean(fft), var(fft)
#data
rw_data = np.array.empty((pair_number))
rd_data = np.array.empty((pair_number))
rw_stat = np.array.empty((pair_number,stat_num))
rd_stat = np.array.empty((pair_number,stat_num))
rw_stat_8_split = np.array.empty((pair_number,stat_num,8))
rd_stat_8_split = np.array.empty((pair_number,stat_num,8))

#read data
for i in range(pair_number):
    rawscienceframe = pyfits.open(url+'rw'+str(i+1)+'.fits')
    reducedscienceframe = pyfits.open(url+'rd'+str(i+1)+'.fits')
    rw_data[i] = rawscienceframe[0].data
    rd_data[i] = reducescienceframe[0].data
    rawscienceframe.close()
    reducedscienceframe.close()

#get statistics
for i in range(pair_number):
    rw_stat[i][0] = np.nanstd(rw_data[i])
    rw_stat[i][1] = np.nanvar(rw_data[i])
    rw_stat[i][2] = np.nanmax(rw_data[i])
    rw_stat[i][3] = scipy.stats.skew(rw_data[i].ravel())
    fft = np.abs(np.fft.fft2(rw_data[i]))
    rw_stat[i][4] = np.median(fft)
    rw_stat[i][5] = np.mean(fft)
    rw_stat[i][6] = np.nanvar(fft)

    
