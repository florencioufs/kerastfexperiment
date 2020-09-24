#Este código é uma modificação do código desenvolvidos por Gazar (2018). Disponível em: https://medium.com/@mgazar/lenet-5-in-9-lines-of-code-using-keras-ac99294c8086



from requests import get

def download_file(url, file_name):
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

download_file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte.gz')
download_file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte.gz')
download_file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz')
download_file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz')



%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)


import gzip
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
#import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path,'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                        .reshape(length, 784) \
                        .reshape(length, 28, 28, 1)
        
    return features, labels

train = {}
test = {}

train['features'], train['labels'] = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

"""### Split training data into training and validation"""

validation = {}
train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)




# Pad images with 0s
train['features']      = np.pad(train['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
validation['features'] = np.pad(validation['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
test['features']       = np.pad(test['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
#print("Updated Image Shape: {}".format(train['features'][0].shape))

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=120, activation='relu'),
  tf.keras.layers.Dense(units=84, activation='relu'),
  tf.keras.layers.Dense(units=10, activation = 'softmax')
 ])

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

EPOCHS = 10
BATCH_SIZE = 100

X_train, y_train = train['features'], tf.keras.utils.to_categorical(train['labels'])
X_validation, y_validation = validation['features'], tf.keras.utils.to_categorical(validation['labels'])

train_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)


steps_per_epoch = X_train.shape[0]//BATCH_SIZE
validation_steps = X_validation.shape[0]//BATCH_SIZE

#from time import time

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
for i in range(100):
  start = time.time()
  model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, 
                    validation_data=validation_generator, validation_steps=validation_steps, 
                    shuffle=True)
  end = time.time()
  print(f'{end-start:.6f}')

for i in range(100):  
  start = time.time()
  score = model.evaluate(test['features'], tf.keras.utils.to_categorical(test['labels']))
  end = time.time()
  print(f'{end-start:.6f}')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
