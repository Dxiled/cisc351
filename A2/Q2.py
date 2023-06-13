# Python version 3.7.9
# TensorFlow version 2.4.1
# SciPy is a dependancy for image transformations used to augment training data
# SciPy version 1.6.1

# All image data is stored in a directory 'images' in the same directory as this file. 
# Change the following variable to change the path to the image directory
IMAGEPATH = 'images/'

# Training time is about 7-10 minutes; using verbose to show progress and accuracy/loss.
# Change the following variable to 0 to hide this information.
VERBOSE = 1

# Change the following variable to change the number of training epochs. 
EPOCHS = 5

# Importing relevant libraries
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from random import shuffle

# Dictionary containing 1-hot representations of the labels
LABELDICT = {
    'Topwear':      [1,0,0,0,0,0,0,0,0,0,0,0,0],
    'Bottomwear':   [0,1,0,0,0,0,0,0,0,0,0,0,0],
    'Innerwear':    [0,0,1,0,0,0,0,0,0,0,0,0,0],
    'Bags':         [0,0,0,1,0,0,0,0,0,0,0,0,0],
    'Watches':      [0,0,0,0,1,0,0,0,0,0,0,0,0],
    'Jewellery':    [0,0,0,0,0,1,0,0,0,0,0,0,0],
    'Eyewear':      [0,0,0,0,0,0,1,0,0,0,0,0,0],
    'Wallets':      [0,0,0,0,0,0,0,1,0,0,0,0,0],
    'Shoes':        [0,0,0,0,0,0,0,0,1,0,0,0,0],
    'Sandal':       [0,0,0,0,0,0,0,0,0,1,0,0,0],
    'Makeup':       [0,0,0,0,0,0,0,0,0,0,1,0,0],
    'Fragrance':    [0,0,0,0,0,0,0,0,0,0,0,1,0],
    'Others':       [0,0,0,0,0,0,0,0,0,0,0,0,1]
}

# Helper method to get the path of an image given its line in the CSV file
def getPath(line):
    return IMAGEPATH + line.split('\t')[0] + '.jpg'

# Helper method to get the 1-hot representation of the label of an image given
# its line in the CSV file
def labelImage(line):
    return LABELDICT.get(line.split('\t')[1])

# Loads all image data given a CSV file
def loadData(file):
    data = []
    f = open(file, 'r')
    lines = f.read().split('\n')
    for line in lines[1:]:
        try:
            label = labelImage(line)
            path = getPath(line)
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((80, 60), Image.ANTIALIAS)
            data.append([np.array(img), label])
        except:
            pass
    shuffle(data)
    return data

# Loads all image data in the training file
def loadTraining():
    return loadData('train.csv')

# Loads all image data in the testing file
def loadTesting():
    return loadData('test.csv')

if __name__ == '__main__':
    # Clears any previous tensorflow session to ensure there is no interference
    tf.keras.backend.clear_session()
    
    # Creates a model containing a normalize [0,255] -> [0,1] layer, 3 layers of 
    # 2d convolution, a flatten layer, three layers of hidden neurons, and a 1-hot prediction layer
    model = tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
                                        tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation=tf.nn.relu, input_shape=(80, 60, 1)),
                                        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                        tf.keras.layers.Conv2D(64, kernel_size = (3, 3), activation=tf.nn.relu, input_shape=(80, 60, 1)),
                                        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                        tf.keras.layers.Conv2D(128, kernel_size = (3, 3), activation=tf.nn.relu, input_shape=(80, 60, 1)),
                                        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(13, activation=tf.nn.softmax)])
    
    # Creates an ImageDataGenerator to generate additional images for training
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    
    # Compiles the model for categorical crossentropy, with an adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    # Loads the training data
    trainData = loadTraining()
    
    # Prepares the training data
    trainImages = tf.convert_to_tensor(np.array([np.array(i[0]) for i in trainData]).reshape(-1, 80, 60, 1))
    trainLabels = tf.convert_to_tensor(np.array([i[1] for i in trainData]).reshape(-1, 13))
    
    datagen.fit(trainImages)
    
    # Fits the model to the training data
    model.fit(datagen.flow(trainImages, trainLabels), batch_size = 50, epochs = EPOCHS, verbose=VERBOSE)
    
    # Loads the testing data
    testData = loadTesting()
    
    # Prepares the testing data
    testImages = np.array([i[0] for i in testData]).reshape(-1, 80, 60, 1)
    testLabels = np.array([i[1] for i in testData]).reshape(-1, 13)
    
    # Evaluates the model based on the testing data
    loss, acc = model.evaluate(testImages, testLabels, verbose=0)
    
    # Displays accuracy and loss values
    print('loss: %f' %loss)
    print('accuracy: %f%%' %(acc * 100))
    
    # Displays the model summary
    model.summary()