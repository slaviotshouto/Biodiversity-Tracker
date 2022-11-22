# %%
import numpy as np
%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random

# %%
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dirs = ["train", "valid", "test", "images to test"]
path = Path.cwd()
train_dir = str(path) + "\\" + dirs[0]
val_dir = str(path) + "\\" + dirs[1]
test_dir = str(path) + "\\" + dirs[2]

batch_size = 32

def generate(dataset):
    gen = ImageDataGenerator( rescale = 1.0/255. )
    x = gen.flow_from_directory(
        directory = dataset,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        target_size=(224,224)
    )
    return x

train_data = generate(train_dir)
val_data = generate(val_dir)
test_data = generate(test_dir)

# %%
# Build the model
def construct(model_name):
    base_model = model_name
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape =(224,224,3), name = "input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling_layer")(x)
    outputs = tf.keras.layers.Dense(400, activation = "softmax", name = "output_layer")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Compile and fit the model
def fitness(model, epochs, epoch_steps):
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
        metrics = ["accuracy"]
    )

    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    
    history = model.fit(
        train_data,
        epochs = epochs,
        steps_per_epoch = epoch_steps,
        validation_data = val_data,
        validation_steps = len(val_data),
        callbacks=[early]
    )
    return history, model.evaluate(test_data)

# %%
V2 = tf.keras.applications.MobileNetV2(include_top= False,)
model_V2  = construct(V2)
steps_per_epoch = len(train_data)
eval_V2  = fitness(model_V2 , 30, steps_per_epoch )
print('Accuracy of MobileNetV2: ', eval_V2[1][1])

# %%
#model_V2.save('good_V2_model.h5')

# %%
#check random picture
idx=random.randint(0,400)
file = test_dir + "\\" + str(names[idx]) + "\\1.jpg"
pred_and_plot(model_V2, file, names)


# %%
#check 100 pictures

names = os.listdir(dirs[0])
n = evaluation_100(model_V2)
print( n, 'good guesses out of 100' )

# %%



