import shutil

import tensorflow as tf
import os
import uvicorn as uvicorn

from pathlib import Path
from sys import platform
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
path = Path.cwd()

if "win" in platform:
    directory_slashes = "\\"
else:
    directory_slashes = "//"

model_V2 = tf.keras.models.load_model('cnn_weights.h5')


@app.post('/analyze_hook')
def analyze_hook(file: UploadFile = File(...)):
    prediction_json = {}

    try:
        with open('temp.zip', 'wb') as buffer:
            temp_zip_loc = buffer.name
            shutil.copyfileobj(file.file, buffer)
            zip_location = os.path.join(os.getcwd(), 'temp')
            shutil.unpack_archive(buffer.name, zip_location, 'zip')

            prediction_json = predict_from_zip()
    finally:
        # Cleanup
        for file in os.listdir(os.path.join(os.getcwd(), 'temp')):
            os.remove(os.path.join(os.path.join(os.getcwd(), 'temp'), file))
        os.remove(temp_zip_loc)

        return prediction_json


def load_and_prep_image(filename, img_shape=224):
    img = tf.io.read_file(filename)  # read image
    img = tf.image.decode_image(img)  # decode the image to a tensor
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


# Singular check
def predict(model, filename, class_names):
    img = load_and_prep_image(filename)
    prediction = model.predict(tf.expand_dims(img, axis=0))

    if len(prediction[0]) > 1:  # check for multi-class
        pred_class = class_names[prediction.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(prediction)[0][0])]  # if only one output, round

    return pred_class


def predict_from_zip():
    temp_folder = str(path) + f"{directory_slashes}temp"
    cl_names = os.listdir(os.getcwd() +
                          directory_slashes + 'cnn_models' +
                          directory_slashes + 'evaluation_models' +
                          directory_slashes + 'test')
    results_dic = {}
    files = os.listdir(temp_folder)

    for file in files:
        my_file = temp_folder + f"{directory_slashes}{file}"
        my_pred = predict(model_V2, my_file, cl_names)
        if my_pred not in results_dic:
            results_dic[my_pred] = 1
        else:
            results_dic[my_pred] += 1

    return results_dic


if __name__ == "__main__":
    uvicorn.run(app, port=8000, log_level="info")
