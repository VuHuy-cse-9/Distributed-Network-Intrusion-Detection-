from models.NewOnlineAdaboost import NewOnlineAdaboost
from DataGenerator.DataGenerator import get_full_data, get_part_data
from utils import detection_rate, false_alarm_rate
from visualize import plot_multi_norm_data, plot_global_history, roc_curve_plot

import json
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.svm import SVC
import os.path

import options


class Trainer:
    def __init__(self, options):
        self.opt = options
        
        #Get train dataset:
        self.data={}
        if self.opt.train_mode == "full":
            self.data["X_train"], self.data["Y_train"], \
                self.data["labels_idx"], self.data["labels"], self.data["feature_names"] = \
                get_full_data(self.opt.path_train, 
                         self.opt.log_transform,
                         self.opt.cate2cont)
        elif self.opt.train_mode == "part":
            self.data["X_train"], self.data["Y_train"], \
                self.data["labels_idx"], self.data["labels"], self.data["feature_names"] = \
                get_part_data(self.opt.path_train, 
                         self.opt.log_transform,
                         self.opt.cate2cont)
        elif self.opt.train_mode == "random":
            raise Exception("Feature haven't implemented!")
        
        #Get test dataset:
        self.data["X_test"], self.data["Y_test"], _, _, _ = \
                get_full_data(self.opt.path_test, 
                         self.opt.log_transform,
                         self.opt.cate2cont)
        
        #Prepare model:
        self.model = NewOnlineAdaboost(
            self.opt.r,
            self.opt.p,
            self.opt.P,
            self.opt.gamma,
            self.opt.beta,
            self.opt.n_components,
            self.opt.T,
            self.opt.default_std
        )
        
        if self.opt.cate2cont == "integer":
            self.n_features = 41
            self.n_labels = 5
        elif self.opt.cate2cont == "remove":
            self.n_features = 32
            self.n_labels = 5
            
        self.model.build(n_labels=self.n_labels, n_features=self.n_features) 
        
    def train(self):
        self.model.fit(self.data["X_train"], self.data["Y_train"])
        
    def eval(self):
        Y_roc_test = np.array(np.clip(self.data['Y_test'], -1, 1), dtype=np.float32)
        y_score = self.model.predict_score(self.data['X_test'])
        y_predict = np.sign(y_score)
        tpr = detection_rate(Y_roc_test, y_predict)
        fpr = false_alarm_rate(Y_roc_test, y_predict)
        
        if not os.path.exists(self.opt.path_result + f"model{self.opt.nodeid}/"):
            os.mkdir(self.opt.path_result + f"model{self.opt.nodeid}/")
            
        #Save roc_curve
        roc_curve_plot(
            y_test=Y_roc_test, 
            y_score = y_predict, 
            path_dir = self.opt.path_result + f"model{self.opt.nodeid}/")
        
        #Save plot curve
        strong_gmms = self.model.strong_gmms
        means = np.array([strong_gmms[i].means for i in range(self.n_features)])
        stds = np.array([strong_gmms[i].stds for i in range(self.n_features)])
        plot_multi_norm_data(self.data["X_test"], 
                            Y_roc_test, 
                            means, 
                            stds, 
                            label_indexs=self.data["labels_idx"], 
                            label_names=self.data["labels"], 
                            feature_names=self.data["feature_names"],
                            path_dir=self.opt.path_result + f"model{self.opt.nodeid}/",
                            n_components=self.opt.n_components)
        print(f">> tpr {tpr * 100}%, fpr {fpr * 100}%")

        
    def save(self):
        model_dict = self.model.to_dict()
        model_dict["node"] = self.opt.nodeid
        params = json.dumps(model_dict)
        #Save model parameter
        if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/"):
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "para.json", "w") as outfile:
                outfile.write(params)
        else:
            os.mkdir(self.opt.path_save_dir + f"model{self.opt.nodeid}")
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "para.json", "x") as outfile:
                outfile.write(params)
        hyperatmeter = {
            "r": self.opt.r,
            "p": self.opt.p,
            "P": self.opt.P,
            "gamma": self.opt.gamma,
            "beta": self.opt.beta,
            "components": self.opt.n_components,
            "T":self.opt.T,
            "std": self.opt.default_std,
            "train_mode": self.opt.train_mode,
            "path_train": self.opt.path_train,
            "log_transform": self.opt.log_transform,
            "cate2cont": self.opt.cate2cont,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
        }
        hyperjson = json.dumps(hyperatmeter)
        if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/"):
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "hyper.json", "w") as outfile:
                outfile.write(hyperjson)
        else:
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "hyper.json", "x") as outfile:
                outfile.write(hyperjson)
        
    def summary_dataset(self):
        #Summerize data:
        print("=====================TRAIN DATA SUMMARY=========================")
        print(f"Number of samples {self.data['X_train'].shape[0]}, Number of features: {self.data['X_train'].shape[1]}")
        print(f"Train: Number of normal {np.sum(self.data['Y_train'] == 1)}, Number of attack {np.sum(self.data['Y_train'] == -1)}")
        print(f"Number of labels: {np.unique(self.data['labels_idx'])}")
        print(f"Number of : {len(self.data['Y_train'][self.data['Y_train'] == -1])}")
        print(f"Number of : {len(self.data['Y_train'][self.data['Y_train'] == -2])}")
        print(f"Number of : {len(self.data['Y_train'][self.data['Y_train'] == -3])}")
        print(f"Number of : {len(self.data['Y_train'][self.data['Y_train'] == -4])}")
        
        print("=====================TEST DATA SUMMARY=========================")
        print(f"Number of samples {self.data['X_test'].shape[0]}, Number of features: {self.data['X_test'].shape[1]}")
        print(f"Train: Number of normal {np.sum(self.data['Y_test'] == 1)}, Number of attack {np.sum(self.data['Y_test'] <= -1)}")
        print(f"Number of labels: {np.unique(self.data['Y_test']).shape[0]}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -1])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -2])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -3])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -4])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -5])}") #Some thing not right here
        print("=================================================================")