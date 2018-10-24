# Import packages - matplotlibin three instances. Matplotlib.use(agg) makes sure that plt does not try to use a X window (which gives errors on the DGX)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os,sys
from pathlib import Path
from skimage.transform import resize

# Set global variables
path = Path('/flashblade/lars_data/2018_panorama_amsterdam/')
labelPath= path / 'labeled_by_model//'
picturePath = path /'Target/'
groundTruthPath = path / 'labeled'

xsize = 1024
ysize = 512

# list labels and pictures
labelList = ['Foto_596180_4.9291121678783_52.4111631089982.txt']
labelList = os.listdir(str(groundTruthPath))
pictureList = [l.replace('.txt', '.jpg') for l in labelList]

print(np.shape(labelList))

# Load labels and image
for label,picture in zip(labelList,pictureList):
    labelArr=np.genfromtxt(str(labelPath / label),delimiter=',')
    pictureArr = plt.imread(str(picturePath / picture))
#    pictureArr = resize(pictureArr,(ysize,xsize))
    truthArr = np.genfromtxt(str(groundTruthPath / label),delimiter=',')
    
    # Plot labels and image as subplots next to each other
    # -- Set font settings

    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams.update({'font.size': 20})

    # -- Create figure
    fig = plt.figure(figsize=[20,25])
    gs = GridSpec(3,1,width_ratios=[1],height_ratios=[1,1,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(pictureArr)
    
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(xsize)
    y = np.arange(ysize)
    ax2 = plt.pcolormesh(x,np.flip(y,axis=0),labelArr, cmap='rainbow',vmin=0,vmax=4)
    fig.colorbar(ax2)
#     ax2.imshow(labelArr)

    ax3 = fig.add_subplot(gs[2])
    x = np.arange(2000)
    y = np.arange(1000)
    ax3 = plt.pcolormesh(x,np.flip(y,axis=0),truthArr, cmap='rainbow',vmin=0,vmax=4)
    fig.colorbar(ax3)
    
    savePath=str(path / 'labelplots' / label.replace('.txt','_200_HD.png'))
    print(savePath)
#     plt.show()
    plt.savefig(savePath,bbox_inches='tight')
    plt.close()
