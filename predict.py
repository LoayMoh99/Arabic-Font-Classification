from __future__ import division
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.filters import threshold_otsu
import numpy as np
from scipy.signal import convolve2d
from sklearn import svm ,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.utils import shuffle
import pickle
import os
import sys
from contextlib import redirect_stdout
import time


def result_file(prediction,calc_time,Out):

    with  open(Out +'times.txt','w') as outfile:
        with redirect_stdout(outfile):
            for c in calc_time:
                print(c)

    with  open(Out +'results.txt','w') as outfile:
        with redirect_stdout(outfile):
            for p in prediction:
                print(p[0]+1)
                
            


#preprocessing
def preprocess(img):
    # Preprocess the given image img.
    #-----------------------------------------------------------------------------------------------
    # Convert the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #-----------------------------------------------------------------------------------------------
    #Convert the grayscale image to a binary image. Apply a threshold using Otsu's method on the blurred image.
    # get the threshold of the image using Otsu's method
    thresh = threshold_otsu(gray)
    thresholded_img = gray > thresh
    #_, thresholded_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    #-----------------------------------------------------------------------------------------------
    return thresholded_img
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------

# LPQ "Local Phase Quantization" Feature Extraction:
# Blurring insensitive
def lpq(img,winSize=5):

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    #  STFT uniform window Basic STFT filters
    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Histogram if needed
    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    LPQdesc=LPQdesc/LPQdesc.sum()

    #print(LPQdesc)
    return LPQdesc

#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    #-----------------------------------------------------------------------------------------------

# Read file from

TestDetDir = sys.argv[1]
OutDir = sys.argv[2]

if os.path.exists(TestDetDir):
    AC_DIR = os.path.basename(TestDetDir)
    OUT_DIR = os.path.basename(OutDir)
    print (AC_DIR,'-',OUT_DIR,'  ',TestDetDir,'-',OutDir)

print('----------------------------------------------------------------')
# file exists
# Reading Dataset
x_test=[]

for i in range(1,len(sorted(glob.glob(TestDetDir+'*.png')))+1):
    filename = glob.glob(TestDetDir+str(i)+'.png')[0]
    img = cv2.imread(filename) ## cv2.imread reads images in RGB format
    x_test.append(img)
    print(filename+' index: '+str(i))
#estelam el dump

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
x_test = np.array(x_test)
preprocessed_x = []

## Getitng time

prediction = []
calc_time = []
for i in range(x_test.shape[0]):
    ## starting time
    start = time.time()
    try:
        preprocessed_img = preprocess(x_test[i])
        features_x = lpq(preprocessed_img)
        prediction.append(loaded_model.predict(np.array(features_x).reshape(1,-1))) 
        # end time
        end = time.time()
        calc_t = round(end-start,2)
        if(calc_t != 0):
            calc_time.append(calc_t)
        else:
            calc_time.append(0.01)
    except:
        end = time.time()
        calc_t = round(end-start,2)
        if(calc_t != 0):
            calc_time.append(calc_t)
        else:
            calc_time.append(0.01)
        prediction.append([-2])




# result = loaded_model.score(test_y.reshape(-1,1), prediction)
print('-----------------',prediction,'\n',calc_time)
result_file(prediction,calc_time,OutDir)

