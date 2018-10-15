# Import packages
from keras.models import load_model
import random
import cv2
from skimage.transform import resize
import numpy as np
import os,sys
sys.path.insert(0, '/flashblade/lars_data/2018_panorama_amsterdam/python_scripts')
from loss_with_weights import weighted_categorical_crossentropy
import glob

# Set global variables
globalPath='/flashblade/lars_data/2018_panorama_amsterdam/'
fileDir = os.path.join(globalPath,'Target')
labelDir = os.path.join(globalPath,'testmap')
outputDir = os.path.join(globalPath,'testmap')
modelName = '50epochs_drop01_filter_64_weights_lr1-3_256_512_epoch'+'*'
modelDir = os.path.join(globalPath,'model')

xsize=256
ysize=512

# list models
modelNames = sorted(glob.glob(os.path.join(modelDir,modelName)))
print(modelNames)

# loop over models
for saved_model in modelNames:
	epoch = saved_model[-4:]
	print(epoch)
	# Load image
	ims=cv2.imread(os.path.join(fileDir,'Foto_596180_4.9291121678783_52.4111631089982.jpg'))
	ims=resize(ims,(xsize,ysize))
	ims = np.expand_dims(ims, axis=0)
	
	# Let model predict the labels
	model=load_model(saved_model, custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
	output = model.predict(ims)
	
	# Transfer the output to correct dimensions and write to file
	output=output[0,:,:,:]
	output=np.argmax(output,axis=-1)
	print(np.unique(output,return_counts=True))
	#output=resize(output,(1000,2000))
	np.savetxt(os.path.join(outputDir,'output'+epoch+'.txt'),output,fmt='%.1d',delimiter=',')	
