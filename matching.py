import numpy as np
import scipy
import pyfits
from sklearn import neural_network
#import cv2
from matplotlib import pyplot as plt

#input
pair_number = 20
stat_num = 7 #std, var, max, skew, median(fft), mean(fft), var(fft)
par_num = 3
#data
rw_data = [[] for i in range(pair_number)]
rd_data = [[] for i in range(pair_number)]
rw_stat = [[[] for i in range(stat_num)] for j in range(pair_number)]
rd_stat = [[[] for i in range(stat_num)] for j in range(pair_number)]
rw_stat_8_split = [[[[] for i in range(stat_num)] for j in range(8)] for k in range(pair_number)]
rd_stat_8_split = [[[[] for i in range(stat_num)] for j in range(8)] for k in range(pair_number)]
rw_block = [[] for i in range(8)]
rd_block = [[] for i in range(8)]
corr_stats = [[] for i in range(stat_num)]
fit_par = [[[] for i in range(par_num)] for j in range(stat_num)]

#read data
for i in range(pair_number):
    rawscienceframe = pyfits.open('rw_imdata/rw'+str(i+1)+'.fits')
    reducedscienceframe = pyfits.open('rd_imdata/rd'+str(i+1)+'.fits')
    rw_data[i] = np.array(rawscienceframe[0].data)
    rd_data[i] = np.array(reducedscienceframe[0].data)
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

    rw_row = len(rw_data[i][0])
    rw_column = len(rw_data[i])
    rw_block[0] = rw_data[i][0:rw_column/2][0:rw_row/4]
    rw_block[1] = rw_data[i][rw_column/2:rw_column][0:rw_row/4]
    rw_block[2] = rw_data[i][0:rw_column/2][rw_row/4:rw_row/2]
    rw_block[3] = rw_data[i][rw_column/2:rw_column][rw_row/4:rw_row/2]
    rw_block[4] = rw_data[i][0:rw_column/2][rw_row/2:(3*rw_row)/4]
    rw_block[5] = rw_data[i][rw_column/2:rw_column][rw_row/2:(3*rw_row)/4]
    rw_block[6] = rw_data[i][0:rw_column/2][(3*rw_row)/4:rw_row]
    rw_block[7] = rw_data[i][rw_column/2:rw_column][(3*rw_row)/4:rw_row]

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

    rd_row = len(rd_data[i][0])
    rd_column = len(rd_data[i])
    rd_block[0] = rd_data[i][0:rd_column/2][0:rd_row/4]
    rd_block[1] = rd_data[i][rd_column/2:rd_column][0:rd_row/4]
    rd_block[2] = rd_data[i][0:rd_column/2][rd_row/4:rd_row/2]
    rd_block[3] = rd_data[i][rd_column/2:rd_column][rd_row/4:rd_row/2]
    rd_block[4] = rd_data[i][0:rd_column/2][rd_row/2:(3*rd_row)/4]
    rd_block[5] = rd_data[i][rd_column/2:rd_column][rd_row/2:(3*rd_row)/4]
    rd_block[6] = rd_data[i][0:rd_column/2][(3*rd_row)/4:rd_row]
    rd_block[7] = rd_data[i][rd_column/2:rd_column][(3*rd_row)/4:rd_row]

    for j in range(8):
        rd_stat_8_split[i][j][0] = np.nanstd(rd_block[j])
        rd_stat_8_split[i][j][1] = np.nanvar(rd_block[j])
        rd_stat_8_split[i][j][2] = np.nanmax(rd_block[j])
        rd_stat_8_split[i][j][3] = scipy.stats.skew(rd_block[j].ravel())
        fft = np.abs(np.fft.fft2(rd_block[j]))
        rd_stat_8_split[i][j][4] = np.median(fft)
        rd_stat_8_split[i][j][5] = np.mean(fft)
        rd_stat_8_split[i][j][6] = np.nanvar(fft)
    print(i)
#calculate correction
##for unsplit images
for i in range(stat_num):
	corr_stats[i] = (scipy.stats.pearsonr([rw_stat[j][i] for j in range(20)], [rd_stat[j][i] for j in range(20)]))[0]
#plot figure
def func(x, a, b, c):
    return a * np.exp(b * x) + c
for i in range(stat_num):
    fit_par[i] = scipy.optimize.curve_fit(func, [rw_stat[j][i] for j in range(20)], [rd_stat[j][i] for j in range(20)], p0=np.array([2, 0.5, 1]))
    print(str(fit_par[i][0][0])+'*exp('+str(fit_par[i][0][1])+'*x)+'+str(fit_par[i][0][2]))

plt.figure(1)
xlabel = [600, 350000, 70000, 300, 250000, 500000, 10000000000000]
for i in range(stat_num):
    plt.subplot(4,2,i+1)
    plt.scatter([rw_stat[j][i] for j in range(20)], [rd_stat[j][i] for j in range(20)])
#    plt.plot(np.linspace(0, xlabel[i], num =20), func(np.linspace(0, xlabel[i], num =20), fit_par[i][0][0],fit_par[i][0][1],fit_par[i][0][2]), label = 'Fitted Curve')
    plt.legend()
plt.show()

plt.figure(2)
plt.bar(np.arange(7), corr_stats)
plt.xticks(np.arange(7), ('std', 'var', 'max', 'skew', 'median(fft)', 'mean(fft)', 'var(fft)'))
plt.show()

plt.figure(3)
for i in range(8):
    for j in range(stat_num):
        plt.subplot(8,stat_num,j+1+i*stat_num)
        plt.scatter([rw_stat_8_split[k][i][j] for k in range(20)],[rd_stat_8_split[k][i][j] for k in range(20)])
plt.show()
