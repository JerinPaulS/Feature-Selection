import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from Py_FS.filter import PCC as FS
import collections

def read_data():
    dataset = pd.read_csv("/home/jerinpaul/Documents/Study Material/DISS1/Dataset/data.csv")
    columns_to_remove = ['Unnamed: 32', 'id']
    dataset.drop(columns_to_remove, axis=1, inplace=True)
    dataset["diagnosis"] = (dataset["diagnosis"] == 'M').astype('int')

    data = dataset.drop("diagnosis", axis=1)
    target = dataset["diagnosis"]
    return data, target

def preprocess(data, target):
    d = preprocessing.normalize(data)
    X = pd.DataFrame(d)

    X = X.to_numpy()
    y = target.to_numpy()
    return data, target

def select_features(data, ranks, features, threshold):
    index_feature_mapping = collections.defaultdict(str)
    index_vals = []
    for index, val in enumerate(ranks):
        index_vals.append([val, index])
        index_feature_mapping[index] = features[index]
    index_vals.sort()
    features_picked = []
    index = 0
    while threshold:
        features_picked.append(index_feature_mapping[index_vals[index][1]])
        index += 1
        threshold -= 1
    new_data = data.filter(features_picked, axis=1)
    print("Picked Features: ", features_picked)
    print(new_data.head)
    return new_data

def model_creation(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=20, random_state=4)
    #print(X_train.shape[1])
    model=tf.keras.models.Sequential([
        tf.keras.layers.Dense(64,activation='elu',input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64,activation='elu'),
        tf.keras.layers.Dense(64,activation='elu'),
        tf.keras.layers.Dense(2,activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #fit model on train dataset
    model.fit(X_train,y_train,epochs=1750)
    predicted = model.predict(X_test)
    predicted = np.argmax(predicted, axis = 1)
    print(model.evaluate(X_test, y_test))

    confusion_matrix = metrics.confusion_matrix(y_test, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    cm_display.plot()
    plt.show()

    actual_values, predicted_values = y_test, predicted

    print("Accuracy score",accuracy_score(actual_values, predicted_values))
    print("Precision score",precision_score(actual_values, predicted_values, pos_label=1))
    print("Recall score",recall_score(actual_values, predicted_values))
    print("F1 score",f1_score(actual_values, predicted_values))

features = {0: "radius_mean", 1: "texture_mean", 2:	"perimeter_mean", 3: "area_mean", 4: "smoothness_mean", 5: "compactness_mean", 6: "concavity_mean",
            7: "concave points_mean", 8: "symmetry_mean", 9: "fractal_dimension_mean", 10: "radius_se", 11: "texture_se", 12: "perimeter_se", 13: "area_se",
            14: "smoothness_se", 15: "compactness_se", 16: "concavity_se", 17: "concave_points_se", 18: "symmetry_se", 19: "fractal_dimension_se",
            20: "radius_worst", 21: "texture_worst", 22: "perimeter_worst", 23: "area_worst", 24: "smoothness_worst", 25: "compactness_worst",
            26:"concavity_worst", 27: "concave_points_worst", 28: "symmetry_worst", 29: "fractal_dimension_worst"}

data, target = read_data()
data, target = preprocess(data, target)

#print(data.head, data.shape, target.shape, target.head)
solution = FS(data, target)
#print(solution.scores)
rankings = solution.ranks
print(len(solution.ranks))
#print(solution.ranked_features)

new_data = select_features(data, solution.ranks, features, 11)
model_creation(new_data, target)
###############################################################################################
