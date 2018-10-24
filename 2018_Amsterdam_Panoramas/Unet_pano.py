import keras
from keras.utils import np_utils
import rasterio
import os,sys
import random
import numpy as np
import math
from pathlib import Path
import cv2
from skimage.transform import resize
from itertools import compress
import matplotlib.pyplot as plt
sys.path.insert(0, '/flashblade/lars_data/2018_panorama_amsterdam/python_scripts')
from loss_with_weights import weighted_categorical_crossentropy
import datetime
import glob

#os.chdir('/home/daniel/R/landuse/Sentinel_models')
#random.seed(65)
epochs = 2000
classes = 5 # Remember: 'not labeled' is also a class!
batch_size = 1
xsize=512
ysize=1024
model_name = 'model/75pics_drop01_filter_64_lr0005_'+str(xsize)+'_'+str(ysize)+'_epoch'

laptop_path = Path('/Users/datalab1/Lars/Panorama_Amsterdam/')
dgx_path=Path('/flashblade/lars_data/2018_panorama_amsterdam')

path=dgx_path
labeldir='10label'
labeldir = 'prepared_input_'+str(xsize)+'_'+str(ysize)
files_labels = [os.path.basename(x) for x in glob.glob(str(path/labeldir)+'/Foto_*_txt.npy')]
#files_labels = ['Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy']
print(files_labels)
#sys.exit()


def generator():
    while(True):
        #get all labels and shuffle them
        files_labels = [os.path.basename(x) for x in glob.glob(str(path/labeldir)+'/Foto_*_txt.npy')]
        random.shuffle(files_labels)
#        files_labels = ['Foto_596180_4.9291121678783_52.4111631089982_txt.npy']
#        files_labels = ['Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy','Foto_596180_4.9291121678783_52.4111631089982_txt.npy']
        #construct filenames of raw data       
        files_raws = list()
#        for label in files_labels:
#            files_raws.append(label.replace('.txt', '_figure.npy'))  

        #calculate how long it takes to loop through all files            
        n = int(math.floor(len(files_labels)/batch_size))      
        
        #construct number of rotations and reflections per file
#         num_rotations =  np.random.choice([0,1,2,3], size = n , replace = True)
        reflect = np.random.choice([False, True], size = n, replace = True)
        
        for i in np.arange(n):        
            
            labels = list() # Label is empty list every loop because a single label and image are passed to the model at the end
            for j in np.arange(batch_size):
#                label = os.path.join(path,'labeled',files_labels[batch_size*i + j])
                label = np.load(path / labeldir / files_labels[batch_size*i + j])
#                 print('[label]',files_labels[batch_size*i + j])
#                label = np.genfromtxt(label, delimiter = ',')
#                 print(np.mean(label))
#                label = resize(label, (xsize,ysize),order=0)
#                print(np.unique(label,return_counts=True))
#                label = np_utils.to_categorical(label, classes)
#                label = label.astype(np.int32)
#                label = np.expand_dims(label, axis = 0)
                labels.append(label)
#                 print('end of label routine',label.shape)
            labels = np.concatenate(labels, axis = 0)            

            ims = list() # ims is empty list every loop because a single label and image are passed to the model at the end
            
            for j in np.arange(batch_size):
#                imagePath=os.path.join(path,'Target',files_raws[batch_size*i +j])
                image = np.load(path / labeldir / files_labels[batch_size*i + j].replace('_txt.npy', '_figure.npy'))
#                print(str(path / labeldir / files_labels[batch_size*i + j].replace('_txt.npy', '_figure.npy')))
#                image = cv2.imread(imagePath)
#                image = plt.imread(imagePath)
#                print('before image resize',np.shape(image))
#                image = resize(image, (xsize,ysize))
#                print(image[100,100,:])
#                 image = np.expand_dims(image, axis=0)
                ims.append(image)
            
##           Do a random mirror over x axis 
#            if reflect[batch_size*i]:
#                ims=np.flip(ims, axis = 1)
#                labels=np.flip(labels, axis = 1)
#            if batch_size == 1:
            ims = np.expand_dims(image, axis=0)
#            print(np.shape(ims),np.shape(labels))
#            sys.exit()
            yield( [ims, labels])

#built network - 4 pooling
dropout=0.1
filters=64*1
kernelsize = (3,3) # 3,3 voor 256x512, 5,5 voor 512x1024
input_im =keras.engine.Input( shape = [xsize,ysize,3], dtype = 'float32' )

l0 = keras.layers.convolutional.Conv2D( filters=filters*1, kernel_size = kernelsize,padding="same",     activation = 'relu' )(input_im)
l0 = keras.layers.convolutional.Conv2D( filters=filters*1, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0)

l1 = keras.layers.MaxPool2D(pool_size = (2,2))(l0)
l1 = keras.layers.Dropout(dropout)(l1)
l1 = keras.layers.convolutional.Conv2D( filters=filters*2, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1)
l1 = keras.layers.convolutional.Conv2D( filters=filters*2, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1)

l2 = keras.layers.MaxPool2D(pool_size = (2,2))(l1)
l2 = keras.layers.Dropout(dropout)(l2)
l2 = keras.layers.convolutional.Conv2D( filters=filters*4, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2)
l2 = keras.layers.convolutional.Conv2D( filters=filters*4, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2)

l3 = keras.layers.MaxPool2D(pool_size = (2,2))(l2)
l3 = keras.layers.Dropout(dropout)(l3)
l3 = keras.layers.convolutional.Conv2D( filters=filters*8, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3)
l3 = keras.layers.convolutional.Conv2D( filters=filters*8, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3)

l4 = keras.layers.MaxPool2D(pool_size = (2,2))(l3)
l4 = keras.layers.Dropout(dropout)(l4)
l4 = keras.layers.convolutional.Conv2D( filters=filters*16, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4)
l4 = keras.layers.convolutional.Conv2D( filters=filters*16, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4)


l3_up = keras.layers.convolutional.Conv2DTranspose(filters = filters*8 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l4)
l3_up = keras.layers.concatenate([l3,l3_up])
l3_up = keras.layers.Dropout(dropout)(l3_up)
l3_up = keras.layers.convolutional.Conv2D( filters=filters*8, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3_up)
l3_up = keras.layers.convolutional.Conv2D( filters=filters*8, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3_up)

l2_up = keras.layers.convolutional.Conv2DTranspose(filters = 256 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l3_up)
l2_up = keras.layers.concatenate([l2,l2_up])
l2_up = keras.layers.Dropout(dropout)(l2_up)
l2_up = keras.layers.convolutional.Conv2D( filters=filters*4, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2_up)
l2_up = keras.layers.convolutional.Conv2D( filters=filters*4, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2_up)

l1_up = keras.layers.convolutional.Conv2DTranspose(filters = 128 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l2_up)
l1_up = keras.layers.concatenate([l1,l1_up])
l1_up = keras.layers.Dropout(dropout)(l1_up)
l1_up = keras.layers.convolutional.Conv2D( filters=filters*2, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1_up)
l1_up = keras.layers.convolutional.Conv2D( filters=filters*2, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1_up)

l0_up = keras.layers.convolutional.Conv2DTranspose(filters = 64 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l1_up)
l0_up = keras.layers.concatenate([l0,l0_up])
l0_up = keras.layers.Dropout(dropout)(l0_up)
l0_up = keras.layers.convolutional.Conv2D( filters=filters*1, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0_up)
l0_up = keras.layers.convolutional.Conv2D( filters=filters*1, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0_up)

output = keras.layers.convolutional.Conv2D( filters=classes, kernel_size = kernelsize,padding="same",     activation = 'softmax' )(l0_up)

model = keras.models.Model(inputs = input_im, outputs = output)

opt = keras.optimizers.adam( lr= 0.0005 , decay = 0,  clipnorm = 0.3 )
model.compile(loss=weighted_categorical_crossentropy, optimizer=opt, metrics = ["accuracy"])
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ["accuracy"])

##built network - 5 pooling
#
#input_im =keras.engine.Input( shape = [xsize,ysize,3], dtype = 'float32' )
#
#l0 = keras.layers.convolutional.Conv2D( filters=64, kernel_size = kernelsize,padding="same",     activation = 'relu' )(input_im)
#l0 = keras.layers.convolutional.Conv2D( filters=64, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0)
#
#l1 = keras.layers.MaxPool2D(pool_size = (2,2))(l0)
#l1 = keras.layers.convolutional.Conv2D( filters=128, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1)
#l1 = keras.layers.convolutional.Conv2D( filters=128, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1)
#
#l2 = keras.layers.MaxPool2D(pool_size = (2,2))(l1)
#l2 = keras.layers.convolutional.Conv2D( filters=256, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2)
#l2 = keras.layers.convolutional.Conv2D( filters=256, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2)
#
#l3 = keras.layers.MaxPool2D(pool_size = (2,2))(l2)
#l3 = keras.layers.convolutional.Conv2D( filters=512, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3)
#l3 = keras.layers.convolutional.Conv2D( filters=512, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3)
#
#l4 = keras.layers.MaxPool2D(pool_size = (2,2))(l3)
#l4 = keras.layers.convolutional.Conv2D( filters=1024, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4)
#l4 = keras.layers.convolutional.Conv2D( filters=1024, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4)
#
#l5 = keras.layers.MaxPool2D(pool_size = (2,2))(l4)
#l5 = keras.layers.convolutional.Conv2D( filters=2048, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l5)
#l5 = keras.layers.convolutional.Conv2D( filters=2048, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l5)
#
#l4_up = keras.layers.convolutional.Conv2DTranspose(filters = 1024 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l5)
#l4_up = keras.layers.concatenate([l4,l4_up])
#l4_up = keras.layers.convolutional.Conv2D( filters=1024, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4_up)
#l4_up = keras.layers.convolutional.Conv2D( filters=1024, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l4_up)
#
#l3_up = keras.layers.convolutional.Conv2DTranspose(filters = 512 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l4_up)
#l3_up = keras.layers.concatenate([l3,l3_up])
#l3_up = keras.layers.convolutional.Conv2D( filters=512, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3_up)
#l3_up = keras.layers.convolutional.Conv2D( filters=512, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l3_up)
#
#l2_up = keras.layers.convolutional.Conv2DTranspose(filters = 256 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l3_up)
#l2_up = keras.layers.concatenate([l2,l2_up])
#l2_up = keras.layers.convolutional.Conv2D( filters=256, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2_up)
#l2_up = keras.layers.convolutional.Conv2D( filters=256, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l2_up)
#
#l1_up = keras.layers.convolutional.Conv2DTranspose(filters = 128 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l2_up)
#l1_up = keras.layers.concatenate([l1,l1_up])
#l1_up = keras.layers.convolutional.Conv2D( filters=128, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1_up)
#l1_up = keras.layers.convolutional.Conv2D( filters=128, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l1_up)
#
#l0_up = keras.layers.convolutional.Conv2DTranspose(filters = 64 , kernel_size = kernelsize ,strides = (2, 2), padding="same")(l1_up)
#l0_up = keras.layers.concatenate([l0,l0_up])
#l0_up = keras.layers.convolutional.Conv2D( filters=64, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0_up)
#l0_up = keras.layers.convolutional.Conv2D( filters=64, kernel_size = kernelsize,padding="same",     activation = 'relu' )(l0_up)
#
#output = keras.layers.convolutional.Conv2D( filters=classes, kernel_size = kernelsize,padding="same",     activation = 'softmax' )(l0_up)
#
#model = keras.models.Model(inputs = input_im, outputs = output)
#
#opt = keras.optimizers.adam( lr= 0.0001 , decay = 0,  clipnorm = 0.3 )
#model.compile(loss="categorical_crossentropy", optimizer=opt, metrics = ["accuracy"])
# #load a previous model
# model = keras.models.load_model( path +'model/model_rotations_3')

checkpointfilepath = str(path / model_name)+'{epoch:04d}'
callback = [keras.callbacks.EarlyStopping(monitor='loss', min_delta = 0.001, patience = 200,restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(checkpointfilepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=20)]

callback_no_checkpoint = [keras.callbacks.EarlyStopping(monitor='loss', min_delta = 0.001, patience = 20000,restore_best_weights=True)]
model.fit_generator(generator = generator(), steps_per_epoch = len(files_labels), epochs = epochs, callbacks=callback)


name = str(path / model_name)+"{:04d}".format(epochs)
model.save(name)
print('model name = ',model_name)

## train -- with saving only at the end
#model.fit_generator(generator = generator(), steps_per_epoch = len(files_labels), epochs = epochs )
#name = path / (str(model_name)+str(epochs))
#model.save(name)
