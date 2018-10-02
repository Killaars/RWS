# Import packages
from keras.models import load_model
import random
import cv2
from skimage.transform import resize
import numpy as np
import os,sys

# Set global variables
globalPath='/flashblade/lars_data/2018_panorama_amsterdam/'
fileDir = os.path.join(globalPath,'Target')
labelDir = os.path.join(globalPath,'labeled')
outputDir = os.path.join(globalPath,'labeled_by_model')
modelFile = os.path.join(globalPath,'model','model_reflect_75images_softmax0')

xsize=512
ysize=1024

labelNames = []

# List all pictures
fileNames=os.listdir(fileDir)
#fileNames=['Foto_598311_4.84414291467923_52.3777060218316.jpg','Foto_315126_4.9673188447722_52.3610755119837.jpg']

# Check if picture is already labeled
for fileName in fileNames:
	labelName=fileName.replace('jpg','txt')
	if labelName not in os.listdir(labelDir):
		if labelName not in os.listdir(outputDir):
			labelNames=np.append(labelNames,labelName)

# Pick n pictures to label
n=10
list_of_random_items = random.sample(list(labelNames), n)
print(list_of_random_items)

for file in list_of_random_items:
	print('file',file)
	filePath=os.path.join(fileDir,file)

	# Load image
	ims=cv2.imread(filePath.replace('.txt','.jpg'))
	ims=resize(ims,(xsize,ysize))
	ims = np.expand_dims(ims, axis=0)

	# Let model predict the labels
	model=load_model(modelFile)
	output = model.predict(ims)
	
	# Transfer the output to correct dimensions and write to file
	output=output[0,:,:,:]
	output=np.argmax(output,axis=2)
	np.savetxt(os.path.join(outputDir,file),output,fmt='%.1d',delimiter=',')
