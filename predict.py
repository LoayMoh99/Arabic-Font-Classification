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

# Writout function

def result_file(prediction,Out):
    with  open(Out +'.txt','w') as outfile:
        with redirect_stdout(outfile):
            for p in prediction:
                print(p+1)
                
            


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

# Feature Extraction LPQ
# LPQ "Local Phase Quantization" Feature Extraction:
def lpq(img,winSize=5,freqestim=1,mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
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

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
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

for filename in sorted(glob.glob(TestDetDir+'/*.png')):
    img = cv2.imread(filename) ## cv2.imread reads images in RGB format
    x_test.append(img)
    print(filename+'heheeeeeeeeeeeeeeeeeeeeeeee')
#estelam el dump

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
x_test = np.array(x_test)
preprocessed_x = []
for i in range(x_test.shape[0]):
    preprocessed_img = preprocess(x_test[i])
    preprocessed_x.append(preprocessed_img)

preprocessed_x=np.asarray(preprocessed_x)


features_x = [lpq(x) for x in preprocessed_x]
features_x=np.array(features_x)
prediction = loaded_model.predict(features_x)
# result = loaded_model.score(test_y.reshape(-1,1), prediction)
print('-----------------',prediction,)
result_file(prediction,OutDir)

