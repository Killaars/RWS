# Import packages
from keras.models import load_model
import random
import cv2
from skimage.transform import resize
import numpy as np
import os,sys
sys.path.insert(0, '/flashblade/lars_data/2018_panorama_amsterdam/python_scripts')
from loss_with_weights import weighted_categorical_crossentropy

# Set global variables
globalPath='/flashblade/lars_data/2018_panorama_amsterdam/'
fileDir = os.path.join(globalPath,'Target')
labelDir = os.path.join(globalPath,'labeled')
outputDir = os.path.join(globalPath,'labeled_by_model')
modelFile = os.path.join(globalPath,'model','75pics_drop01_filter_64_lr0005_256_512_epoch0100')

xsize=256
ysize=512

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
n=20
list_of_random_items = random.sample(list(labelNames), n)
print(list_of_random_items)
list_of_random_items=['Foto_596180_4.9291121678783_52.4111631089982.txt']
#list_of_random_items=['Foto_412949_4.84737765922196_52.3973588044208.txt']

for file in list_of_random_items:
	print('file',file)
	filePath=os.path.join(fileDir,file)

	# Load image
	ims=cv2.imread(filePath.replace('.txt','.jpg'))
	ims=resize(ims,(xsize,ysize))
	ims = np.expand_dims(ims, axis=0)

	# Let model predict the labels
	model=load_model(modelFile, custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
#	model=load_model(modelFile)
	output = model.predict(ims)
	print(output)	
	# Transfer the output to correct dimensions and write to file
	output=output[0,:,:,:]
#	print(np.shape(output))
	test=output[0,0,:]
	np.savetxt(os.path.join(outputDir,'dim0.txt' ),output[:,:,0],delimiter=',')	
	np.savetxt(os.path.join(outputDir,'dim1.txt' ),output[:,:,1],delimiter=',')	
	np.savetxt(os.path.join(outputDir,'dim2.txt' ),output[:,:,2],delimiter=',')	
	np.savetxt(os.path.join(outputDir,'dim3.txt' ),output[:,:,3],delimiter=',')	
	np.savetxt(os.path.join(outputDir,'dim4.txt' ),output[:,:,4],delimiter=',')	
#	print(test,np.argmax(test))
	output=np.argmax(output,axis=-1)
#	print(output)
	print(np.unique(output,return_counts=True))
	#output=resize(output,(1000,2000))
#	print(output)
	np.savetxt(os.path.join(outputDir,file),output,fmt='%.1d',delimiter=',')	

