import time

from numpy import extract

start = time.time()

import os
import json

os.system('cls')

print("Modules importation :\n")
print(f"{'    Standard modules' :-<50}", end="")
import pathlib
import numpy as np
import matplotlib.pyplot as plt

print(" Done\n")

print(f"{'    Tensorflow modules' :-<50}", end="")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

print(" Done\n")

####################################################################################################
#                                          LOADING DATA                                            #
####################################################################################################
print('Loading data : \n')

data_path = os.path.join('dataset')

data_dir = pathlib.Path(f'{data_path}/training')
val_dir = pathlib.Path(f'{data_path}/validation')

image_count_dat = len(list(data_dir.glob('*/*.*')))
print(f'    Dataset images    : {image_count_dat}')
image_count_val = len(list(val_dir.glob('*/*.*')))
print(f'    Validation images : {image_count_val}')


####################################################################################################
#                                       PREPROCESSING DATA                                         #
####################################################################################################
print('Preprocessing data :\n')

batch_size = 128
img_height = 150
img_width = 150

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=710,
    image_size=(img_height, img_width),
    interpolation = "bilinear",
    labels = "inferred",
    label_mode = "int",
    class_names = None,
    color_mode = "rgb",
    batch_size=batch_size)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    validation_split=0.2,
    subset="validation",
    seed=710,
    image_size=(img_height, img_width),
    interpolation = "bilinear",
    labels = "inferred",
    label_mode = "int",
    class_names = None,
    color_mode = "rgb",
    batch_size=batch_size)

class_names = train_data.class_names
iter_train = iter(train_data)
iter_val = iter(val_data)
first_train = next(iter_train)
print(f'\n    Class names : {class_names}')

index = first_train[1].numpy()

for i in range(batch_size):
    image = first_train[0][i]
    plt.figure()
    plt.imshow(image.numpy().astype(np.int64))
    plt.title(class_names[index[i]])

# Create a dictionary with initial values of 0
pokemon_data = {name: index for index, name in enumerate(class_names, start=0)}

# Save the dictionary to a JSON file
with open('pokemon_indices.json', 'w') as json_file:
    json.dump(pokemon_data, json_file, indent= 4)

print("JSON file 'pokemon_indices.json' created")

####################################################################################################
#                                          NEURAL NETWORK                                          #
####################################################################################################
nb_classes = 903
model_name = "pokemon_recognition.h5"

if os.path.exists(model_name):
    model = load_model(model_name)
else:
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(nb_classes, activation='softmax')        
    ])

opt = keras.optimizers.legacy.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.build((None, img_height, img_width, 3))
print(model.summary())

logdir = f'logs/{model_name}'

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=logdir,
                                                   embeddings_data=train_data)


model.fit(train_data, validation_data=val_data, epochs = 10, callbacks=[tensorboard_callback])


####################################################################################################
#                                              Testing                                             #
####################################################################################################

first_val = next(iter_val)

pred = model.predict(first_val[0])

image = first_val[0][0]

score = tf.nn.softmax(pred[0])

plt.figure()
plt.imshow(image.numpy().astype(np.int64))
plt.title(class_names[np.argmax(score)])

####################################################################################################
print(f'\nProcessing complete (time : {round(time.time() - start, 4)}s)')
plt.show()

if os.path.exists(model_name):
    os.remove(model_name)

model.save(model_name)