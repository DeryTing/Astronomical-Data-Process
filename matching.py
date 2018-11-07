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
rw_block = np.array.empty((8))
rd_block = np.array.empty((8))

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

    rw_row = len(rw_data[i])
    rw_column = len(rw_data[i][0]) 
    rw_block[0] = rw_data[i][0:rw_row/4][0:rw_column/2]
    rw_block[1] = rw_data[i][0:rw_row/4][rw_column/2:rw_column]
    rw_block[2] = rw_data[i][rw_row/4:rw_row/2][0:rw_column/2]
    rw_block[3] = rw_data[i][rw_row/4:rw_row/2][rw_column/2:rw_column]
    rw_block[4] = rw_data[i][rw_row/2:(3*rw_row)/4][0:rw_column/2]
    rw_block[5] = rw_data[i][rw_row/2:(3*rw_row)/4][rw_column/2:rw_column]
    rw_block[6] = rw_data[i][(3*rw_row)/4:rw_row][0:rw_column/2]
    rw_block[7] = rw_data[i][(3*rw_row)/4:rw_row][rw_column/2:rw_column]
                
    for j in range(8):
        rw_stat_8_split[i][j][0] = np.nanstd(rw_block[j])
        rw_stat_8_split[i][j][1] = np.nanvar(rw_block[j])
        rw_stat_8_split[i][j][2] = np.nanmax(rw_block[j])
        rw_stat_8_split[i][j][3] = scipy.stats.skew(rw_block[j].ravel())
        fft = np.abs(np.fft.fft2(rw_block[j]))
        rw_stat_8_split[i][j][4] = np.median(fft)
        rw_stat_8_split[i][j][5] = np.mean(fft)
        rw_stat_8_split[i][j][6] = np.nanvar(fft)
#plot statistics    
