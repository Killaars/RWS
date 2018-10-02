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



#os.chdir('/home/daniel/R/landuse/Sentinel_models')

epochs = 1
classes = 5 # Remember: 'not labeled' is also a class!
batch_size = 1
model_name = 'model/test'
save_each = 1

laptop_path = Path('/Users/datalab1/Lars/Panorama_Amsterdam/')
dgx_path=Path('/flashblade/lars_data/2018_panorama_amsterdam')

path=dgx_path
files_labels = os.listdir(str(path/'labeled' ))
print(files_labels)

# np_utils.to_categorical is faster
# def get_one_hot(targets, nb_classes):
#     print('[get_one_hot]',np.shape(targets),np.shape(nb_classes))
# #     print(nb_classes)
#     res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
#     return res.reshape(list(targets.shape)+[nb_classes])


def generator():
    while(True):
        #get all labels and shuffle them
        files_labels = os.listdir(str(path/'labeled' ))
        random.shuffle(files_labels)
        
        #construct filenames of raw data       
        files_raws = list()
        for label in files_labels:
            files_raws.append(label.replace('.txt', '.jpg'))  

        #calculate how long it takes to loop through all files            
        n = int(math.floor(len(files_labels)/batch_size))      
        
        #construct number of rotations and reflections per file
#         num_rotations =  np.random.choice([0,1,2,3], size = n , replace = True)
        reflect = np.random.choice([False, True], size = n, replace = True)
                    
        for i in np.arange(n):        
            
            labels = list() # Label is empty list every loop because a single label and image are passed to the model at the end
            for j in np.arange(batch_size):
#                label = os.path.join(path,'labeled',files_labels[batch_size*i + j])
                label = path / 'labeled' / files_labels[batch_size*i + j]
#                 print('[label]',files_labels[batch_size*i + j])
                label = np.genfromtxt(label, delimiter = ',')
#                 print(np.mean(label))
                label = np.rint(resize(label, (512,1024))).astype(int)
#                 print(np.mean(label))
#                 print('label before one hot vector',np.shape(label))
                label = np_utils.to_categorical(label, classes)
                label = label.astype(np.int32)
                label = np.expand_dims(label, axis = 0)
                labels.append(label)
#                 print('end of label routine',label.shape)
            
            labels = np.concatenate(labels, axis = 0)            

            ims = list() # ims is empty list every loop because a single label and image are passed to the model at the end
            
            for j in np.arange(batch_size):
#                imagePath=os.path.join(path,'Target',files_raws[batch_size*i +j])
                imagePath = str(path / "Target" / files_raws[batch_size*i +j]) # cv2 wants a string as pathname, without str() this would be a pathlib class
                image = cv2.imread(imagePath)
#                 print('before image resize',np.shape(image))
                image = resize(image, (512, 1024))
#                 image = np.expand_dims(image, axis=0)
                ims.append(image)
            
#           Do a random mirror over x axis 
            if reflect[batch_size*i]:
                print(np.shape(ims))
                ims=np.flip(ims, axis = 1)
                labels=np.flip(labels, axis = 1)
            ims = np.expand_dims(image, axis=0)

#             #do a random rotation !!!!!!!!!!Does not do anything yet!!!!!!!!!!!
#             for k in range(num_rotations[batch_size*i]):
#                 r =  np.rot90(r, axes = (1,2)) 
#                 label = np.rot90(label, axes = (1,2))
 
#             print(np.shape(ims),np.shape(labels))
            yield( [ims, labels])

#built network

input_im =keras.engine.Input( shape = [512,1024,3], dtype = 'float32' )

l0 = keras.layers.convolutional.Conv2D( filters=64, kernel_size= (3,3),padding="same",     activation = 'relu' )(input_im)
l0 = keras.layers.convolutional.Conv2D( filters=64, kernel_size= (3,3),padding="same",     activation = 'relu' )(l0)

l1 = keras.layers.AvgPool2D(pool_size = (2,2))(l0)
l1 = keras.layers.convolutional.Conv2D( filters=128, kernel_size= (3,3),padding="same",     activation = 'relu' )(l1)
l1 = keras.layers.convolutional.Conv2D( filters=128, kernel_size= (3,3),padding="same",     activation = 'relu' )(l1)

l2 = keras.layers.AvgPool2D(pool_size = (2,2))(l1)
l2 = keras.layers.convolutional.Conv2D( filters=256, kernel_size= (3,3),padding="same",     activation = 'relu' )(l2)
l2 = keras.layers.convolutional.Conv2D( filters=256, kernel_size= (3,3),padding="same",     activation = 'relu' )(l2)

l3 = keras.layers.AvgPool2D(pool_size = (2,2))(l2)
l3 = keras.layers.convolutional.Conv2D( filters=512, kernel_size= (3,3),padding="same",     activation = 'relu' )(l3)
l3 = keras.layers.convolutional.Conv2D( filters=512, kernel_size= (3,3),padding="same",     activation = 'relu' )(l3)

l4 = keras.layers.AvgPool2D(pool_size = (2,2))(l3)
l4 = keras.layers.convolutional.Conv2D( filters=1024, kernel_size= (3,3),padding="same",     activation = 'relu' )(l4)
l4 = keras.layers.convolutional.Conv2D( filters=1024, kernel_size= (3,3),padding="same",     activation = 'relu' )(l4)


l3_up = keras.layers.convolutional.Conv2DTranspose(filters = 512 , kernel_size=(3,3) ,strides = (2, 2), padding="same")(l4)
l3_up = keras.layers.concatenate([l3,l3_up])
l3_up = keras.layers.convolutional.Conv2D( filters=512, kernel_size= (3,3),padding="same",     activation = 'relu' )(l3_up)
l3_up = keras.layers.convolutional.Conv2D( filters=512, kernel_size= (3,3),padding="same",     activation = 'relu' )(l3_up)

l2_up = keras.layers.convolutional.Conv2DTranspose(filters = 256 , kernel_size=(3,3) ,strides = (2, 2), padding="same")(l3_up)
l2_up = keras.layers.concatenate([l2,l2_up])
l2_up = keras.layers.convolutional.Conv2D( filters=256, kernel_size= (3,3),padding="same",     activation = 'relu' )(l2_up)
l2_up = keras.layers.convolutional.Conv2D( filters=256, kernel_size= (3,3),padding="same",     activation = 'relu' )(l2_up)

l1_up = keras.layers.convolutional.Conv2DTranspose(filters = 128 , kernel_size=(3,3) ,strides = (2, 2), padding="same")(l2_up)
l1_up = keras.layers.concatenate([l1,l1_up])
l1_up = keras.layers.convolutional.Conv2D( filters=128, kernel_size= (3,3),padding="same",     activation = 'relu' )(l1_up)
l1_up = keras.layers.convolutional.Conv2D( filters=128, kernel_size= (3,3),padding="same",     activation = 'relu' )(l1_up)

l0_up = keras.layers.convolutional.Conv2DTranspose(filters = 64 , kernel_size=(3,3) ,strides = (2, 2), padding="same")(l1_up)
l0_up = keras.layers.concatenate([l0,l0_up])
l0_up = keras.layers.convolutional.Conv2D( filters=64, kernel_size= (3,3),padding="same",     activation = 'relu' )(l0_up)
l0_up = keras.layers.convolutional.Conv2D( filters=64, kernel_size= (3,3),padding="same",     activation = 'relu' )(l0_up)

output = keras.layers.convolutional.Conv2D( filters=classes, kernel_size= (3,3),padding="same",     activation = 'softmax' )(l0_up)

model = keras.models.Model(inputs = input_im, outputs = output)

opt = keras.optimizers.adam( lr= 0.0001 , decay = 0,  clipnorm = 0.3 )
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics = ["accuracy"])

# #load a previous model
# model = keras.models.load_model( path +'model/model_rotations_3')


#train
for epoch in range(epochs):
    print(epoch)
    model.fit_generator(generator = generator(), steps_per_epoch = len(files_labels), epochs = 1 )

    if epoch % save_each == 0:
        name = path / (str(model_name)+str(epoch))
        model.save(name)

