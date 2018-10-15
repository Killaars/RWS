import keras
from keras.utils import np_utils
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pathlib import Path

laptop_path = Path('/Users/datalab1/Lars/Panorama_Amsterdam/')
dgx_path=Path('/flashblade/lars_data/2018_panorama_amsterdam')

path=dgx_path
labeldir = 'labeled'
files_labels = os.listdir(str(path/labeldir ))
#files_labels = ['Foto_596180_4.9291121678783_52.4111631089982.txt']
outputdir = path/'prepared_input_256_512'

xsize = 256
ysize = 512
classes = 5

for labelname in files_labels:
	labelPath = str(path/labeldir/labelname)
	label = np.genfromtxt(labelPath,delimiter = ',')
	label = resize(label, (xsize,ysize), order=0)
	label = np_utils.to_categorical(label, classes)
	label = label.astype(np.int32)
	label = np.expand_dims(label, axis = 0)
	print(np.shape(label))
	np.save(outputdir/labelname.replace('.txt', '.npy'),label)
	
	imagePath = str(path/'Target'/labelname.replace('.txt', '.jpg'))
	image = plt.imread(imagePath)
	image = resize(image, (xsize,ysize))
	print(np.shape(image))
	np.save(outputdir/labelname.replace('.txt', '_figure.npy'),image)
