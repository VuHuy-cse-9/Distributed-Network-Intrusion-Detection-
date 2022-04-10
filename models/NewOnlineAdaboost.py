"""Training GMMs using online Adaboost
"""
import numpy as np
from models.GMM import OnlineGMM
import hyper
from utils import detection_rate, false_alarm_rate, send_model
import json
from tqdm import tqdm
import argparse
import time
import copy


class NewOnlineAdaboost():
    def __init__(self):
        self.r = hyper.r
        self.p = hyper.p
        self.P = hyper.P
        self.gamma = hyper.gamma
        self.beta = hyper.beta
        self.eta = 1e-9
        self.n_components = hyper.n_components
        self.T = hyper.T

    def build(self, n_labels, n_features):
        self.N_classifiers = n_features   #Depend on number of feature
        self.N_labels = n_labels
        self.S_normal, self.S_attack = 0, 0               #Use in equation (21)
        self.Omega = 1                               #Use in step 5
        self.lamda_sc = np.ones((self.N_classifiers, ), dtype=np.float64) * self.eta  #Use in step 3, 4
        self.lamda_sw = np.ones((self.N_classifiers, ), dtype=np.float64) * self.eta
        self.C = np.ones((self.N_classifiers, ))         #Use in step 3, 4, 5
        self.v  = np.zeros((self.N_classifiers,))         #combined classification rate v_t, equation (23)
        self.epsilon = np.zeros((self.N_classifiers, ))   
        self.lamda = 0                               #Sample weight, using in equation (22)
        self.alphas = np.ones((self.N_classifiers,), dtype=np.float64)
        self.strong_gmms = None
        self.gmms = [] 
        for i in np.arange(self.N_classifiers):
            gmm = OnlineGMM(hyper.std, n_components=self.n_components, T=self.T)
            gmm.build(n_labels=self.N_labels)
            self.gmms.append(gmm)
        self.gmms = np.array(self.gmms)

        
    def fit(self, X_train, Y_train):
        predicts = np.zeros((self.N_classifiers,))
        global_count = 0
        for X, y in tqdm(zip(X_train, Y_train), total=X_train.shape[0]):  
            for i in range(self.N_classifiers):
                predicts[i] =  self.gmms[i].predict(X[i])
            if y == 1:
                self.S_normal += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_normal * self.r
            else:
                self.S_attack += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_attack * (1- self.r)
            #Step 2: Calculate the combined classification weight error
            self.epsilon = self.lamda_sw / (self.lamda_sc + self.lamda_sw)
            self.v = np.maximum((1 - self.p) * self.epsilon - self.p * np.sign(y) * predicts, 0)
            #Step 3: Update classifier
            #Step 3.1: Sort list classifier:
            sort_index = np.argsort(self.v)
            sort_strong_index = np.argwhere(self.v[sort_index] <= 0.5)
            strong_index = np.squeeze(sort_index[sort_strong_index], axis=1)
            weak_index = np.squeeze(np.argwhere(self.v > 0.5), axis=1)
            
            #Step 3.3.1: Calculate number of iterations:
            P_iter = np.array(np.round(self.P * np.exp(-self.gamma*(self.v - np.min(self.v)))), dtype=np.int32)
            #Step 3.3.2: Update strong sub-set
            # print(">> strong index:")
            # for index in strong_index:
            #     print(f">> {index}: alpha {self.alphas[index]} C {self.C[index]} Omega {self.Omega} right {(np.sign(y) == predicts)[index]} epsilon {self.epsilon[index]} - v {self.v[index]} - P {P_iter[index]} - lam_src {self.lamda_sc[index]} - lam_sw {self.lamda_sw[index]}")
                
            # print(f">> weak index: {weak_index}")
            # for index in weak_index:
            #     print(f">> {index}: alpha {self.alphas[index]} C {self.C[index]} Omega {self.Omega} right {(np.sign(y) == predicts)[index]} epsilon {self.epsilon[index]} - v {self.v[index]} - P {P_iter[index]} - lam_src {self.lamda_sc[index]} - lam_sw {self.lamda_sw[index]}")
            if strong_index.size != 0:   
                for index in strong_index:
                    for _ in range(P_iter[index]):
                        self.gmms[index].fit(X[index], y)
                        if np.sign(y) == self.gmms[index].predict(X[index]):
                            self.C[index] += 1   
                            self.lamda_sc[index] += self.lamda
                            self.lamda = self.lamda * (1 - 2*self.p) / 2*(1 - self.v[index])
                        else:
                            self.lamda_sw[index] += self.lamda
                            self.lamda = np.minimum(self.lamda * ((1 + 2*self.p) / np.maximum(2*self.v[index], self.eta)), 100)
            
            #Step 3.3.3: Update weak subset
            if weak_index.size != 0:
                for index in weak_index:
                    if np.sign(y) == predicts[index]:
                        self.C[index] += 1
                        self.lamda_sc[index] += self.lamda
                    else:
                        self.lamda_sw[index] += self.lamda
            
            #Step 3.3.3+: Normalize lamda_sw and lamda_src to avoid overflow:
            # sum_lamda = self.lamda_sc + self.lamda_sw
            # self.lamda_sc = self.lamda_sc / sum_lamda
            # self.lamda_sw = self.lamda_sw / sum_lamda

            #Step 4: Update Omega, contribution from previous classifier
            if self.strong_gmms is not None:
                if np.sign(y) == np.sign(np.sum(
                    [self.alphas[i] * self.strong_gmms[i].predict(X[i]) for i in range(self.N_classifiers)])):
                    self.Omega += 1
                    
            #Step 5: Construct strong classifier
            #Step 5.1: Calculate ensemble weight
            self.alphas = np.maximum(self.beta * np.log(np.maximum((1 - self.epsilon) / self.epsilon, self.eta)) \
                    + (1 - self.beta) * np.log(self.C / self.Omega), 0)
            
            #Step 5.2: Normalize ensemble weight
            if np.sum(self.alphas) != 0:
                self.alphas = self.alphas / np.sum(self.alphas)
            else:
                #print("All alpha is zeros")
                self.alphas = np.ones((self.N_classifiers,))
            #Strong classifier:
            # if self.strong_gmms is not None:
            #     count = 0
            #     for i in range(self.N_classifiers):
            #         if (np.abs(self.strong_gmms[i].means - self.gmms[i].means) < 1e-12).all():
            #             count += 1
            #     if count == self.N_classifiers:
            #         global_count += 1
            #         if global_count == 10:
            #             raise Exception("Strong gmms shallow copy gmms")
            self.strong_gmms = copy.deepcopy(self.gmms)
            
    def evaluate(self, X_test, Y_test):
        y_score = predict_score(X_test)
        predicts = np.sign(y_score)

        dtrs = detection_rate(Y_test, predicts)
        fars = false_alarm_rate(Y_test, predicts)

        return dtrs, fars
    
    def predict_score(self, X_test):
        N_test = X_test.shape[0]
        
        #Global evaluate:
        predicts = []
        #print("================================")
        #print(f"X_test: {X_test[:,0].shape}")
        for i in range(self.N_classifiers):
            predicts.append(self.alphas[i] * self.strong_gmms[i].predict(X_test[:, i]))
        predicts = np.transpose(predicts, (1, 0))

        #Checking 
        if predicts.shape != (N_test, self.N_classifiers):
            raise Exception(f"Shape of global predict is not right: {predicts.shape}")
        return np.sum(self.alphas * predicts, axis=1)
    
    def set_params(self, strong_gmms, alphas, N_classifiers):
        self.N_classifiers = N_classifiers
        self.strong_gmms = strong_gmms
        self.alphas = alphas
