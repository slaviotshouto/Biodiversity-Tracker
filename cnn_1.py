# %%
import numpy as np
%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# %%
dirs = ["train", "valid", "test", "images to test"]
names = os.listdir(dirs[0])

trains = []
vals = []

for i in range( len(names) ):
    trains.append( dirs[0] + "/" + names[i] )
    vals.append( dirs[1] + "/" + names[i] )

# %%
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dirs = ["train", "valid", "test", "images to test"]
path = Path.cwd()
train_dir = str(path) + "/" + dirs[0]
val_dir = str(path) + "/" + dirs[1]
test_dir = str(path) + "/" + dirs[2]

batch_size = 32

train_gen = ImageDataGenerator( rescale = 1.0/255. )
val_gen = ImageDataGenerator( rescale = 1.0/255. )
test_gen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_gen.flow_from_directory(
    directory = train_dir,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224,224)
)
val_generator = val_gen.flow_from_directory(
    directory = val_dir,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224,224)
)
test_generator = test_gen.flow_from_directory(
    directory = test_dir,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='categorical',
    target_size=(224,224)
)
print( type(train_generator[0]), len(train_generator[0][0]), len(train_generator[1][0]) )

# %%
# Building application model.

nn = tf.keras.applications.EfficientNetB7()
nn.trainable = False

inputs = tf.keras.layers.Input(shape =(224,224,3), name = "Input_Layer")

x = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)
x = nn(inputs)
#x = tf.keras.layers.MaxPooling2D(name = "Global_Average_Pooling_Layer")(x)
outputs = tf.keras.layers.Dense(400, activation = "softmax", name = "Output_Layer")(x)

model = tf.keras.Model( inputs, outputs )

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']
nn.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
)
model.compile()
history = nn.fit(
    train_generator,
    epochs=10,
    steps_per_epoch = len(train_generator),
    validation_data = val_generator
)


# %%
# Building sequential model. I'm not sure what are the differences between layers.

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (224, 224 ,3) ),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = "relu"),
])

#cnn.add( tf.keras.layers.Dense(550,activation="relu") )
#cnn.add( tf.keras.layers.Dropout(0.1,seed = 2019) )
#cnn.add( tf.keras.layers.Dense(400,activation="relu") )
#cnn.add( tf.keras.layers.Dropout(0.3,seed = 2019) )
#cnn.add( tf.keras.layers.Dense(250,activation="relu") )
#cnn.add( tf.keras.layers.Dropout(0.5,seed = 2019) )
#cnn.add( tf.keras.layers.Dense(5,activation = "softmax") )


# %%
cnn.summary()

# %%
from tensorflow.keras.optimizers import RMSprop,SGD,Adam

optimizer = Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']

cnn.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
)


# %%
fitted_cnn = cnn.fit(
    train_generator,
    validation_data = val_generator,
    steps_per_epoch = 150//batch_size,
    epochs = 30,
    validation_steps = 5,
    verbose = 2
)

# %%
