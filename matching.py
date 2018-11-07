import numpy as np
import scipy
import pyfits
from sklearn import neural_network
#import cv2
from matplotlib import pyplot as plt

#data
rw_data = np.array.empty((20))
rd_data = np.array.empty((20))

#input
pair_number = 20
url = 'rd_imdata/' #name of the folder

#read data

for i in range(pair_number):
    rawscienceframe = pyfits.open(url+'rw'+str(i+1)+'.fits')
    reducedscienceframe = pyfits.open(url+'rd'+str(i+1)+'.fits')
    rw_data = rawscienceframe[0].data
    rd_data = reducescienceframe[0].data
    rawscienceframe.close()
    reducedscienceframe.close()

#get statistics
