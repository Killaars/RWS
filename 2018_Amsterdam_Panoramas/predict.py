# Import packages
from keras.models import load_model
import random
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import os,sys
sys.path.insert(0, '/flashblade/lars_data/2018_panorama_amsterdam/python_scripts')
from loss_with_weights import weighted_categorical_crossentropy
from pathlib import Path

# Set global variables
globalPath = Path('/flashblade/lars_data/2018_panorama_amsterdam/')
fileDir = globalPath / 'Target'
labelDir = globalPath / 'labeled'
outputDir = globalPath / 'labeled_by_model'
modelFile = globalPath / 'model' / '75pics_drop01_filter_64_lr0005_512_1024_epoch2000'
# best model: '75pics_drop01_filter_64_lr0005_512_1024_epoch0200'

xsize=512
ysize=1024

labelNames = []

# List all pictures
fileNames = os.listdir(str(fileDir))
#fileNames=['Foto_598311_4.84414291467923_52.3777060218316.jpg','Foto_315126_4.9673188447722_52.3610755119837.jpg']

# Check if picture is already labeled
for fileName in fileNames:
	labelName=fileName.replace('jpg','txt')
	if labelName not in os.listdir(str(labelDir)):
		if labelName not in os.listdir(str(outputDir)):
			labelNames=np.append(labelNames,labelName)

# Pick n pictures to label
n=20
list_of_random_items = random.sample(list(labelNames), n)
print(list_of_random_items)
#list_of_random_items=['Foto_596180_4.9291121678783_52.4111631089982.txt']
list_of_random_items=['Foto_601556_4.8621183962168_52.3379946297222.txt','Foto_612658_4.92372569149918_52.4132159617551.txt','Foto_600530_4.86467829342951_52.4149249677368.txt','Foto_599917_4.93976115995873_52.4072977814027.txt',
			'Foto_597973_4.84093157229232_52.3411731318156.txt','Foto_597898_4.8417100483038_52.3407641437206.txt','Foto_597401_4.84392989349633_52.3927025378047.txt','Foto_596864_4.89493362315539_52.4247823784035.txt',
			'Foto_15184_4.84510695130622_52.3795943629646.txt','Foto_316025_4.97267131753808_52.3718801224243.txt',]
list_of_random_items = os.listdir(str(labelDir))
# Let model predict the labels
model=load_model(str(modelFile), custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy})

for file in list_of_random_items:
	print('file',file)
	filePath=fileDir / file

	# Load image
	ims=plt.imread(str(filePath).replace('.txt','.jpg'))
	ims=resize(ims,(xsize,ysize))
	ims = np.expand_dims(ims, axis=0)

	# Predict image
	output = model.predict(ims)
#	print(output)	
	# Transfer the output to correct dimensions and write to file
	output=output[0,:,:,:]
#	print(np.shape(output))
	test=output[0,0,:]
#	np.savetxt(os.path.join(outputDir,'dim0.txt' ),output[:,:,0],delimiter=',')	
#	np.savetxt(os.path.join(outputDir,'dim1.txt' ),output[:,:,1],delimiter=',')	
#	np.savetxt(os.path.join(outputDir,'dim2.txt' ),output[:,:,2],delimiter=',')	
#	np.savetxt(os.path.join(outputDir,'dim3.txt' ),output[:,:,3],delimiter=',')	
#	np.savetxt(os.path.join(outputDir,'dim4.txt' ),output[:,:,4],delimiter=',')	
#	print(test,np.argmax(test))
	output=np.argmax(output,axis=-1)
#	print(output)
	print(np.unique(output,return_counts=True))
	#output=resize(output,(1000,2000))
#	print(output)
	np.savetxt(outputDir / file,output,fmt='%.1d',delimiter=',')	

