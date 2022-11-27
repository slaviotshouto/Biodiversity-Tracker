import shutil
from typing import Dict, Optional

import tensorflow as tf
import os
import uvicorn as uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File

app = FastAPI()
path = Path.cwd()

# Load the model from the weights file
model_V2 = tf.keras.models.load_model('cnn_weights.h5')


@app.post('/analyze_hook')
def analyze_hook(file: UploadFile = File(...)) -> Dict[str, int]:
    """
    The hook that is responsible for receiving a zip file object of time SpooledTempFile.
    it listens on port the port defined at the bottom of the file, once a file has been sent
    unzip it, make predictions on the jpg images, return those predictions and cleanup
    :param file: the file that is
    :return: dictionary object that is return like a json to the flask app
    """
    prediction_json: Dict[str, int] = {}

    try:
        with open('temp.zip', 'wb') as buffer:
            # Open the file received, use shutil to store it and then unzip
            # Even though the file received by FastAPI is a temp file, storing it is necessary for unzipping
            temp_zip_loc: str = buffer.name
            shutil.copyfileobj(file.file, buffer)
            zip_location = os.path.join(os.getcwd(), 'temp')
            shutil.unpack_archive(buffer.name, zip_location, 'zip')

            # Make predictions based on the images, the directory is always the same
            prediction_json = predict_from_zip()
    finally:
        # Perform general cleanup, by first removing the stored images and later removing the zip file itself
        for file in os.listdir(os.path.join(os.getcwd(), 'temp')):
            os.remove(os.path.join(os.path.join(os.getcwd(), 'temp'), file))
        os.remove(temp_zip_loc)

        return prediction_json


def load_and_prep_image(filename, img_shape: Optional = 224):
    """
    Resize the image to the proper shape for predictions
    :param filename: the directory of the file
    :param img_shape: Optional paramether, regarding the size of the img it will be resized to
    :return: resized image
    """
    img = tf.io.read_file(filename)  # read image
    img = tf.image.decode_image(img)  # decode the image to a tensor
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


# Singular check
def predict(model, filename, class_names) -> str:
    """
    :param model: the loaded model with the proper weights
    :param filename: the exact path of images
    :param class_names: classes which can be predicted for each img
    :return: the name of the class predicted for the provided img
    """
    img = load_and_prep_image(filename)
    prediction = model.predict(tf.expand_dims(img, axis=0))

    if len(prediction[0]) > 1:  # check for multi-class
        prediction_class = class_names[prediction.argmax()]  # if more than one output, take the max
    else:
        prediction_class = class_names[int(tf.round(prediction)[0][0])]  # if only one output, round

    return prediction_class


def predict_from_zip() -> Dict[str, int]:
    """
    Function used to get all the current img files paths and pass them to the prediction function
    It counts each occurrence of a class in a dictionary, which it later returns
    :return: dictionary of the encountered classes
    """
    temp_folder = os.path.join(str(path), 'temp')
    cl_names = os.listdir(os.path.join(os.getcwd(), 'cnn_models', 'evaluation_models', 'test'))
    results_dic = {}
    files = os.listdir(temp_folder)

    for file in files:
        file_path = os.path.join(temp_folder, file)
        prediction = predict(model_V2, file_path, cl_names)
        if prediction not in results_dic:
            results_dic[prediction] = 1
        else:
            results_dic[prediction] += 1

    return results_dic


if __name__ == "__main__":
    uvicorn.run(app, port=8000, log_level="info")
