# USAGE
# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64

# import the necessary packages
from keras.models import load_model
import argparse
from pathlib import Path
from imutils import paths
import pickle
import cv2
import sys

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image we are going to classify")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained Keras model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to label binarizer")
# ap.add_argument("-w", "--width", type=int, default=28,
# 	help="target spatial dimension width")
# ap.add_argument("-e", "--height", type=int, default=28,
# 	help="target spatial dimension height")
# ap.add_argument("-f", "--flatten", type=int, default=-1,
# 	help="whether or not we should flatten the image")
# args = vars(ap.parse_args())

# targetsize=16
# model_path=Path('/Users/datalab1/Lars/output/modelsimple500.model')
# label_path=Path('/Users/datalab1/Lars/output/modelsimple500.pickle')

targetsize=32
model_path=Path('/Users/datalab1/Lars/output/modelfullpyimagesearch.model')
label_path=Path('/Users/datalab1/Lars/output/modelfullpyimagesearch.pickle')


dataset_path=Path('/Users/datalab1/Lars/test')
flatten=False
imagePaths = sorted(list(paths.list_images(dataset_path)))
correct=[51,17,32,31,37,27,26,48,47,51,54,4,56,18,1,11,12,0,17,17]
score=0

for counter,imagePath in enumerate(imagePaths):
    print(imagePath,counter)

    # load the input image and resize it to the target spatial dimensions
#     image = cv2.imread(args["image"])
    image = cv2.imread(imagePath)
    output = image.copy()
    image = cv2.resize(image, (targetsize, targetsize))

    # check to see if we should flatten the image and add a batch
    # dimension
    if flatten==True:
        image = image.flatten()
        image = image.reshape((1, image.shape[0]))

    # otherwise, we must be working with a CNN -- don't flatten the
    # image, simply add the batch dimension
    else:
        image = image.reshape((1, image.shape[0], image.shape[1],
            image.shape[2]))

    # load the model and label binarizer
    print("[INFO] loading network and label binarizer...")
    model = load_model(model_path)
    lb = pickle.loads(open(label_path, "rb").read())

    # make a prediction on the image
    preds = model.predict(image)

    # find the class label index with the largest corresponding
    # probability
    i = preds.argmax(axis=1)[0]
    print(i)
    if i==correct[counter]:
        print('TRUE!!')
        score+=1
    else:
        print('FALSE')
    #print(preds)
    label = lb.classes_[i]

    # draw the class label + probability on the output image
    text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)
print(score)
