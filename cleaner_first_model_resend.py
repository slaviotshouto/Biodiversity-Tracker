# %%
# Importing packages
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import splitfolders

# %%
# Opening data
import os

# walk through the directory and list the numbers of files
for dirpath, dirnames, filenames in os.walk("/kaggle/input/100-bird-species"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")

# List of names
names = os.listdir("train")
num_of_bird_groups = len(os.listdir("train"))
print(num_of_bird_groups)

names_test = os.listdir("test")
x = len(os.listdir("test"))
print(x)

# %%
dirs = ["train", "valid", "test", "images to test"]
print(len(names))
print(len(os.listdir(dirs[0])))
i = 0
target_class = names[i]
target_folder = dirs[0] + "/" + target_class
print((os.listdir(target_folder), 1)[0][i])
print(os.listdir(target_folder))

# %%
# Building list of all paths
alles = []
for i in range(len(names)):
    target_class = names[i]
    target_folder = dirs[0] + "/" + target_class
    alles.append(os.listdir(target_folder))
arr = np.array(alles)

alles_test = []
for i in range(len(names_test)):
    target_class = names_test[i]
    target_folder = dirs[2] + "/" + target_class
    alles_test.append(os.listdir(target_folder))
arr_test = np.array(alles_test)

# %%
# Transform jpg into array
from PIL import Image
from numpy import asarray

# %%
target_class = names[0]
target_folder = dirs[0] + "/" + target_class
image = Image.open(target_folder + "/" + arr[0][0])
image.resize((30, 30))
data = asarray(image)
print(data.shape)

# %%
# Building actual model
# I am checking if the assumptions is good - if the accuracy will rise as n rises we are good to
# extend the model to whole dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

iterations = []
scores = []

cut = 50
step = 5

for n_iterations in range(1, 30):

    iterations.append(n_iterations)

    # set X_train as an array of first n pictures of every bird
    # here: transform pictures into arrays
    X = []
    for n in range(n_iterations):
        for i in range(len(arr)):
            target_class = names[i]
            target_folder = dirs[0] + "/" + target_class
            image = Image.open(target_folder + "/" + arr[i][n])
            data = asarray(image)
            X.append(data)

    X_arr = np.array(X)
    np.mean(X_arr, axis=0)

    # transform an array from 4D to 2D
    combined = []
    for i in range(X_arr.shape[0]):
        x = []
        for j in range(0, X_arr.shape[1], step):
            for k in range(0, X_arr.shape[2], step):
                for l in range(X_arr.shape[3]):
                    x.append(X_arr[i][j][k][l])
        combined.append(x)

    X_train = np.array(combined)

    # set X_test as an array one picture of every bird from test set
    Xt = []
    for i in range(len(arr)):
        target_class = names_test[i]
        target_folder = dirs[2] + "/" + target_class
        image = Image.open(target_folder + "/" + arr_test[i][0])
        data = asarray(image)
        Xt.append(data)

    X_arrt = np.array(Xt)

    newt = []
    for i in range(X_arrt.shape[0]):
        x = []
        for j in range(0, X_arrt.shape[1], step):
            for k in range(0, X_arrt.shape[2], step):
                for l in range(X_arr.shape[3]):
                    x.append(X_arr[i][j][k][l])
        newt.append(x)

    X_test = np.array(newt)

    print("Iteration", n_iterations, ": data ready")

    # set y_train as an array of bird names repeated n_iteration times
    y = names.copy()
    for i in range(n_iterations - 1):
        y.extend(names)
    y_train = np.array(y)

    # set y_train as an array of bird names
    y2 = names_test.copy()
    y_test = np.array(y2)

    # test the model
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    # random.shuffle(X_test)
    y_pred = knn.predict(X_test)

    # evaluate the model
    score = accuracy_score(y_test, y_pred)
    scores.append(score)

    print("Iteration", n_iterations, ": training done with anaccuracy of", score)

# %%
import matplotlib.pyplot as plt

for i in range(len(scores)):
    print(iterations[i], "iterations --> accuracy score", scores[i])

plt.plot(iterations, scores)
plt.xlabel("number of bird photos")
plt.ylabel("accuracy")

# %%
# the same preprocessing, cheking every 5 pixel in phtoto

n_iterations = 5
step = 5

X = []
for i in range(1):
    for n in range(n_iterations):
        for i in range(len(arr)):
            target_class = names[i]
            target_folder = dirs[0] + "/" + target_class
            image = Image.open(target_folder + "/" + arr[i][n])
            data = asarray(image)
            X.append(data)

    X_arr = np.array(X)

    # transform an array from 4D to 2D
    combined = []
    for i in range(X_arr.shape[0]):
        x = []
        for j in range(0, X_arr.shape[1], step):
            for k in range(0, X_arr.shape[2], step):
                for l in range(X_arr.shape[3]):
                    x.append(X_arr[i][j][k][l])
        combined.append(x)

    X_train = np.array(combined)

    # set X_test as an array one picture of every bird from test set
    Xt = []
    for i in range(len(arr)):
        target_class = names_test[i]
        target_folder = dirs[2] + "/" + target_class
        image = Image.open(target_folder + "/" + arr_test[i][0])
        data = asarray(image)
        Xt.append(data)

    X_arrt = np.array(Xt)

    newt = []
    for i in range(X_arrt.shape[0]):
        x = []
        for j in range(0, X_arrt.shape[1], step):
            for k in range(0, X_arrt.shape[2], step):
                for l in range(X_arr.shape[3]):
                    x.append(X_arr[i][j][k][l])
        newt.append(x)

    X_test = np.array(newt)

    print("Iteration", n_iterations, ": data ready")

    # set y_train as an array of bird names repeated n_iteration times
    y = names.copy()
    for i in range(n_iterations - 1):
        y.extend(names)
    y_train = np.array(y)

    # set y_train as an array of bird names
    y2 = names_test.copy()
    y_test = np.array(y2)

# %%
# Logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Logistic regreession accuracy:", accuracy_score(y_test, y_pred))

# %%
"""
from sklearn.svm import LinearSVC
svm = LinearSVC().fit(X_train,y_train)
y_pred p svm.predict(X_test)
print( "SVM accuracy:", accuracy_score(y_test, y_pred) )
"""

# %%
# SVM
from sklearn.svm import LinearSVC

svm = LinearSVC().fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("SVM accuracy:", accuracy_score(y_test, y_pred))

# %%
# Random Forest  using 5-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid_rf = {'max_depth': [1, 3, 5, 7, 9]}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

y_pred = grid_search_rf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, y_pred))

# %%
# random check
from math import floor
import random

random = floor(random.random() * 400)
X_train = np.array(combined)

# set X_test as an array one picture of every bird from test set
target_class = names_test[random]
target_folder = dirs[2] + "/" + target_class
image = Image.open(target_folder + "/" + arr_test[random][0])
print(names_test[random])
image.show()
data = asarray(image)
Xt.append(data)

X_arrt = np.array(Xt)

newt = []
for i in range(X_arrt.shape[0]):
    x = []
    for j in range(0, X_arrt.shape[1], step):
        for k in range(0, X_arrt.shape[2], step):
            for l in range(X_arr.shape[3]):
                x.append(X_arr[i][j][k][l])
    newt.append(x)

X_test = np.array(newt)

print("Iteration", n_iterations, ": data ready")

# set y_train as an array of bird names repeated n_iteration times
y = names.copy()
y_train = np.array(y)

# set y_train as an array of bird names
y2 = names_test[random]
y_test = np.array(y2)

# Showing random picture
y_lr = lr.predict(X_test)
print("Logistic Regression prediction: ", y_lr)
y_svm = lr.predict(X_test)
print("SVM prediction: ", y_svm)
y_rf = grid_search_rf.predict(X_test)
print("SVM prediction: ", y_rf)
