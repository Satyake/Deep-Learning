import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
#ConvX2
classifier=Sequential()
shape=(128,128,3)
classifier.add(Convolution2D(34,(3,3),input_shape=shape,activation='relu'))
classifier.add(Convolution2D(34,(3,3),padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(25,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(25,(3,3),padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#conv3s
classifier.add(Convolution2D(16,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(16,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(16,(3,3),padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(10,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(10,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(10,(3,3),padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D(14,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(14,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(14,(3,3),padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(3,activation='softmax'))

classifier.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'J:/Chexray/chest-xray-pneumonia/chest_xray/chest_xray/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'J:/Chexray/chest-xray-pneumonia/chest_xray/chest_xray/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')


classifier.fit_generator(
        train_generator,
        steps_per_epoch=171,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=11)

import numpy as np 
from keras.preprocessing import image
UK=image.load_img('G:/PETERLIU/stair1.jpg',target_size=(128,128))
UK=image.img_to_array(UK)
UK=np.expand_dims(UK,axis=0)
result=classifier.predict(UK)
train_generator.class_indices