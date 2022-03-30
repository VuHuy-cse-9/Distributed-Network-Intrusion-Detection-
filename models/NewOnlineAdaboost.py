"""Training GMMs using online Adaboost
"""
import numpy as np
from models.GMM import OnlineGMM
import hyper
from utils import detection_rate, false_alarm_rate, send_model, send_data
import json
from tqdm import tqdm
from visualize import plot_multi_norm_data
import argparse
import time

class NewOnlineAdaboost():
    def __init__(self):
        self.r = hyper.r
        self.p = hyper.p
        self.P = hyper.P
        self.gamma = hyper.gamma
        self.beta = hyper.beta
        self.eta = 1e-5
        self.n_components = hyper.n_components
        self.T = 0.5

    def build(self, n_labels, n_features):
        self.N_classifiers = n_features   #Depend on number of feature
        self.N_labels = n_labels
        self.S_normal, self.S_attack = 0, 0               #Use in equation (21)
        self.Omega = 1                               #Use in step 5
        self.lamda_sc = np.zeros((self.N_classifiers, ))  #Use in step 3, 4
        self.lamda_sw = np.zeros((self.N_classifiers, ))  
        self.C = np.zeros((self.N_classifiers, ))         #Use in step 3, 4, 5
        self.v  = np.zeros((self.N_classifiers,))         #combined classification rate v_t, equation (23)
        self.epsilon = np.zeros((self.N_classifiers, ))   
        self.lamda = 0                               #Sample weight, using in equation (22)
        self.alphas = np.ones((self.N_classifiers,))
        self.strong_gmms = None
        self.gmms = [None] * self.N_classifiers
        for i in np.arange(self.N_classifiers):
            self.gmms[i] = OnlineGMM(hyper.std, n_components=self.n_components, T=self.T)
            self.gmms[i].build(n_labels=self.N_labels)
        
    def fit(self, X_train, Y_train):
        predicts = np.zeros((self.N_classifiers,))
        for X, y in tqdm(zip(X_train, Y_train), total=X_train.shape[0]): #REMEBER TO N_TRAINX2   
            for i in range(self.N_classifiers):
                predicts[i] =  self.gmms[i].predict(X[i])
                # print(f">> match: {y==self.gmms[i].predict(X[i])}, y = {y}, predict = {self.gmms[i].predict(X[i])}")
                # print(f">> alphas: {self.alphas}")
            #Optimize this
            if y == 1:
                self.S_normal += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_normal * self.r
            else:
                self.S_attack += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_attack * (1- self.r)
            #print(f">>lamda: {self.lamda}")
            #Step 2: Calculate the combined classification weight error
            self.epsilon = (self.lamda_sw) / np.maximum(self.lamda_sc + self.lamda_sw, self.eta)
            #print(f"epsilon: {self.epsilon}")
            self.v = np.maximum((1 - self.p) * self.epsilon - self.p * np.sign(y) * predicts, self.eta)
            #Step 3: Update classifier
            #Step 3.1: Sort list classifier:
            sort_index = np.argsort(self.v)
            sort_strong_index = np.argwhere(self.v[sort_index] <= 0.5)
            strong_index = np.squeeze(sort_index[sort_strong_index], axis=1)
            weak_index = np.squeeze(np.argwhere(self.v > 0.5), axis=1)
            
            #Step 3.3.1: Calculate number of iterations:
            P_iter = np.array(self.P * np.exp(-self.gamma*(self.v - np.min(self.v))), dtype=np.int32)
            #print(f">> P_iter: {P_iter}")
            #Step 3.3.2: Update strong sub-set
            if strong_index.size != 0:
                for index in strong_index:
                    for _ in range(P_iter[index]):
                        self.gmms[index].fit(X[index], y)
                        #print(f">> match: {y==self.gmms[i].predict(X[i])}, y = {y}, predict = {self.gmms[i].predict(X[i])}")
                        if np.sign(y) == self.gmms[index].predict(X[index]):
                            self.C[index] += 1   
                            self.lamda_sc[index] += self.lamda
                            self.lamda = self.lamda * ((1 - 2*self.p) / np.maximum(2*(1 - self.v[index]), self.eta))
                        else:
                            self.lamda_sw[index] += self.lamda
                            self.lamda = self.lamda * ((1 + 2*self.p) / np.maximum(2*self.v[index], self.eta))
            #print(f">>lamda: {self.lamda}")
            #Step 3.3.3: Update weak subset
            if weak_index.size != 0:
                for index in weak_index:
                    #print(f">> match: {y==self.gmms[i].predict(X[i])}, y = {y}, predict = {self.gmms[i].predict(X[i])}")
                    if np.sign(y) == predicts[index]:
                        self.C[index] += 1
                        self.lamda_sc[index] += self.lamda
                    else:
                        self.lamda_sw[index] += self.lamda
            # print(f"lamda_sc: {self.lamda_sc}")
            # print(f"lamda_sw: {self.lamda_sw}")
            # print(f">>lamda: {self.lamda}")

            #Step 4: Update Omega, contribution from previous classifier
            if self.strong_gmms is not None:
                if np.sign(y) == np.sign(np.sum(
                    [self.alphas[i] * self.strong_gmms[i].predict(X[i]) for i in range(self.N_classifiers)])):
                    self.Omega += 1
            #Step 5: Construct strong classifier
            #Step 5.1: Calculate ensemble weight
            #MODIFY: Alphas has to greater than 0
            self.alphas = np.maximum(self.beta * np.log(np.maximum((1 - self.epsilon) / (self.epsilon + self.eta), self.eta)) \
                    + (1 - self.beta) * np.log(np.maximum(self.C / (self.Omega), self.eta)), 0)
            #Step 5.2: Normalize ensemble weight
            if np.sum(self.alphas) != 0:
                self.alphas = self.alphas / np.sum(self.alphas)
            else:
                self.alphas = np.ones((self.N_classifiers,)) * self.eta
            #Strong classifier:
            self.strong_gmms = self.gmms.copy()
            
            
    def evaluate(self, X_test, Y_test):
        dtrs = []
        fars = []
        N_test = X_test.shape[0]
        
        #Global evaluate:
        predicts = []
        for i in range(self.N_classifiers):
            predicts.append(self.alphas[i] * self.strong_gmms[i].predict(X_test[:, i]))
        predicts = np.transpose(predicts, (1, 0))

        #Checking 
        if predicts.shape != (N_test, self.N_classifiers):
            raise Exception(f"Shape of global predict is not right: {predicts.shape}")
        predicts = np.sign(np.sum(self.alphas * predicts, axis=1))

        #Global result
        dtrs.append(detection_rate(Y_test, predicts))
        fars.append(false_alarm_rate(Y_test, predicts))
        
        #Local evaluate:
        for local_index in range(self.N_classifiers):
            predicts = self.strong_gmms[local_index].predict(X_test[:, local_index])
            dtrs.append(detection_rate(Y_test, predicts))
            fars.append(false_alarm_rate(Y_test, predicts))
        return np.array(dtrs), np.array(fars)