import tensorflow as tf
import os
import pandas as pd
from pathlib import Path
from sys import platform

import uvicorn as uvicorn
from fastapi import FastAPI

app = FastAPI()
path = Path.cwd()

if "win" in platform:
    directory_slashes = "\\"
else:
    directory_slashes = "//"

model_V2 = tf.keras.models.load_model('cnn_weights.h5')


@app.post('/analyze_hook')
def analyze_hook(req: dict):
    print("THAT'S GREAT", req)


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
    results_list = []
    files = os.listdir(temp_folder)
    print(files)

    for file in files:
        my_file = temp_folder + f"{directory_slashes}{file}"
        my_pred = predict(model_V2, my_file, cl_names)
        if my_pred not in results_dic:
            results_dic[my_pred] = 1
        else:
            results_dic[my_pred] += 1

    for k, v in results_dic.items():
        results_list.append([k.capitalize(), v])

    df = pd.DataFrame(results_list)
    csv = df.to_csv('results.csv', index=False)


if __name__ == "__main__":
    uvicorn.run(app, port=8000, log_level="info")