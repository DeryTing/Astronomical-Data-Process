import numpy as np
import scipy
import pyfits
from sklearn import neural_network
#import cv2
from matplotlib import pyplot as plt

#input
pair_number = 20
stat_num = 7 #std, var, max, skew, median(fft), mean(fft), var(fft)
#data
rw_data = np.empty((pair_number))
rd_data = np.empty((pair_number))
rw_stat = np.empty((pair_number,stat_num))
rd_stat = np.empty((pair_number,stat_num))
rw_stat_8_split = np.empty((pair_number,stat_num,8))
rd_stat_8_split = np.empty((pair_number,stat_num,8))
rw_block = np.empty((8))
rd_block = np.empty((8))

#read data
for i in range(pair_number):
    rawscienceframe = pyfits.open('rw_imdata/rw'+str(i+1)+'.fits')
    reducedscienceframe = pyfits.open('rd_imdata/rd'+str(i+1)+'.fits')
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

    rd_stat[i][0] = np.nanstd(rd_data[i])
    rd_stat[i][1] = np.nanvar(rd_data[i])
    rd_stat[i][2] = np.nanmax(rd_data[i])
    rd_stat[i][3] = scipy.stats.skew(rd_data[i].ravel())
    fft = np.abs(np.fft.fft2(rd_data[i]))
    rd_stat[i][4] = np.median(fft)
    rd_stat[i][5] = np.mean(fft)
    rd_stat[i][6] = np.nanvar(fft)

    rd_row = len(rd_data[i])
    rd_column = len(rd_data[i][0])
    rd_block[0] = rd_data[i][0:rd_row/4][0:rd_column/2]
    rd_block[1] = rd_data[i][0:rd_row/4][rd_column/2:rd_column]
    rd_block[2] = rd_data[i][rd_row/4:rd_row/2][0:rd_column/2]
    rd_block[3] = rd_data[i][rd_row/4:rd_row/2][rd_column/2:rd_column]
    rd_block[4] = rd_data[i][rd_row/2:(3*rd_row)/4][0:rd_column/2]
    rd_block[5] = rd_data[i][rd_row/2:(3*rd_row)/4][rd_column/2:rd_column]
    rd_block[6] = rd_data[i][(3*rd_row)/4:rd_row][0:rd_column/2]
    rd_block[7] = rd_data[i][(3*rd_row)/4:rd_row][rd_column/2:rd_column]

    for j in range(8):
        rd_stat_8_split[i][j][0] = np.nanstd(rd_block[j])
        rd_stat_8_split[i][j][1] = np.nanvar(rd_block[j])
        rd_stat_8_split[i][j][2] = np.nanmax(rd_block[j])
        rd_stat_8_split[i][j][3] = scipy.stats.skew(rd_block[j].ravel())
        fft = np.abs(np.fft.fft2(rd_block[j]))
        rd_stat_8_split[i][j][4] = np.median(fft)
        rd_stat_8_split[i][j][5] = np.mean(fft)
        rd_stat_8_split[i][j][6] = np.nanvar(fft)
#plot statistics


#apply machinelearning
#classify = neural_network.MLPClassifier(solver = 'lbfgs', alpha = 1e-10, hidden_layer_sizes = (100,100), random_state=1)
#for p in range(20):#per product
#    output[p] = 1
#    for i in range(20):#per source
#        static_train_data[i] = [(s_static[i][j]-p_static[p][j])/(s_static[i][j]) for j in range(8)]
#    classify.fit(static_train_data,output)
#  
