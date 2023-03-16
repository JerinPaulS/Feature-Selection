import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import preprocessing
import collections
from CS_Wrapper_FS import CS as CS_FS
from MA_Wrapper_FS import MA as MA_FS
from GA_Wrapper_FS import GA as GA_FS
from GWO_Wrapper_FS import GWO as GWO_FS
from PSO_Wrapper_FS import PSO as PSO_FS
from WOA_Wrapper_FS import WOA as WOA_FS

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

def CSFeatureSelection(data, target, features):
    solution = CS_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def MAFeatureSelection(data, target, features):
    solution = MA_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def GAFeatureSelection(data, target, features):
    solution = GA_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def GWOFeatureSelection(data, target, features):
    solution = GWO_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def PSOFeatureSelection(data, target, features):
    solution = PSO_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def WOAFeatureSelection(data, target, features):
    solution = WOA_FS(num_agents=50, max_iter=500, train_data=data, train_label=target, save_conv_graph=True)
    features_picked = select_features(solution.best_agent, features)
    return features_picked

def select_features(new_features, features):
    features_picked = []
    for index, val in enumerate(new_features):
        if val:
            features_picked.append(features[index])
    return features_picked

def voting_features(features, result_mapping, count):
    count_mapping = collections.defaultdict(int)

    for feature in features:
        for key in result_mapping:
            if features[feature] in result_mapping[key]:
                count_mapping[feature] += 1
    features_picked = []
    for feature in count_mapping.keys():
        if count_mapping[feature] >= count:
            features_picked.append(features[feature])
    return features_picked


###############################################################################################

def WrapperFeatureSelection():
    features = {0: "radius_mean", 
                1: "texture_mean", 
                2:	"perimeter_mean", 
                3: "area_mean", 
                4: "smoothness_mean", 
                5: "compactness_mean", 
                6: "concavity_mean",
                7: "concave points_mean", 
                8: "symmetry_mean", 
                9: "fractal_dimension_mean", 
                10: "radius_se", 
                11: "texture_se", 
                12: "perimeter_se", 
                13: "area_se",
                14: "smoothness_se", 
                15: "compactness_se", 
                16: "concavity_se", 
                17: "concave_points_se", 
                18: "symmetry_se", 
                19: "fractal_dimension_se",
                20: "radius_worst", 
                21: "texture_worst", 
                22: "perimeter_worst", 
                23: "area_worst", 
                24: "smoothness_worst", 
                25: "compactness_worst",
                26:"concavity_worst", 
                27: "concave_points_worst", 
                28: "symmetry_worst", 
                29: "fractal_dimension_worst"}

    data, target = read_data()
    data, target = preprocess(data, target)

    result_mapping = collections.defaultdict(list)

    result_mapping["GA"] = GAFeatureSelection(data, target, features)
    result_mapping["GWO"] = GWOFeatureSelection(data, target, features)
    result_mapping["PSO"] = PSOFeatureSelection(data, target, features)
    result_mapping["WOA"] = WOAFeatureSelection(data, target, features)

    feature_set = voting_features(features, result_mapping, 2)
    print(feature_set)
    return data.filter(feature_set, axis=1), target
##################################################################
#GA': ['radius_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'concave_points_se', 'radius_worst', 'compactness_worst', 'concavity_worst'], 
#GWO': ['radius_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean', 'texture_se', 'compactness_se', 'texture_worst', 'perimeter_worst', 'compactness_worst'], 
#PSO': ['compactness_mean', 'concavity_mean', 'compactness_se', 'fractal_dimension_se', 'concave_points_worst'], 
#WOA': ['texture_se', 'radius_worst', 'texture_worst']#
####################################################################