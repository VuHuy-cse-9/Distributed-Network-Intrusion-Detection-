"""Training GMMs using online Adaboost
"""
import numpy as np
from GMMs import gmms
from GMMs import hyper
from utils import detection_rate, false_alarm_rate, send_model, load_fake_data, send_data
import json
from tqdm import tqdm
from visualize import plot_multi_norm_data
import argparse

class NewOnlineAdaboost():
    def __init__(self):
        self.N_classifiers = hyper.N_features   #Depend on number of feature
        self.r = hyper.r
        self.p = hyper.p
        self.P = hyper.P
        self.gamma = hyper.gamma
        self.beta = hyper.beta
        self.eta = 1e-5
        self.n_components = hyper.n_components
        self.T = 0.5
        self.N_labels = 2
        #self.N_data_send_count = hyper.N_data_local_send
        # self.N_normal_send = N_data_send_count / 2
        # self.N_attack_send = N_data_send_count / 2

    def build(self):
        self.S_normal, self.S_attack = 0, 0               #Use in equation (21)
        self.Omega = 1                               #Use in step 5
        self.lamda_sc = np.zeros((self.N_classifiers, ))  #Use in step 3, 4
        self.lamda_sw = np.zeros((self.N_classifiers, ))  
        self.C = np.zeros((self.N_classifiers, ))         #Use in step 3, 4, 5
        self.v  = np.zeros((self.N_classifiers,))         #combined classification rate v_t, equation (23)
        self.epsilon = np.zeros((self.N_classifiers, ))   
        self.lamda = 0                               #Sample weight, using in equation (22)
        #Initialize classifier
        self.alphas = np.ones((self.N_classifiers,))
        self.strong_gmms = None
        self.gmms = [gmms.OnlineGMM(hyper.std, n_components=self.n_components, T=self.T)] * self.N_classifiers
        for gmm in self.gmms:
            gmm.build(n_labels=self.N_labels)
        
    def fit(self, X_train, Y_train):
        for x, y in tqdm(zip(X_train, Y_train), total=X_train.shape[0]): #REMEBER TO N_TRAINX2
        #Step 1: Update number of normal and attack sample
        #        and initialize weight for new sample
            # if N_data_send_count > 0:
            #     #Share data.
            #     if y == 1 and N_normal_send > 0:
            #         send_data(x.tolist(), int(y))
            #         N_normal_send -= 1
            #         N_data_send_count -= 1
            #     if y < 0 and N_attack_send > 0:
            #         send_data(x.tolist(), int(y))
            #         N_attack_send -= 1
            #         N_data_send_count -= 1
            
            if y == 1:
                self.S_normal += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_normal * self.r
            else:
                self.S_attack += 1
                self.lamda = (self.S_normal + self.S_attack) / self.S_attack * (1- self.r)
        #Step 2: Calculate the combined classification weight error
            for i in range(self.N_classifiers):
                self.epsilon[i] = (self.lamda_sw[i]) / np.maximum(self.lamda_sc[i] + self.lamda_sw[i], self.eta)
                self.v[i] = np.maximum((1 - self.p) * self.epsilon[i] - self.p * np.sign(y) * self.gmms[i].predict(x[i]), self.eta)
        #Step 3: Update classifier
            #Step 3.1: Sort list classifier:
            sort_index = np.argsort(self.v)
            sort_strong_index = np.argwhere(self.v[sort_index] <= 0.5)
            strong_index = np.squeeze(sort_index[sort_strong_index], axis=1)
            
            weak_index = np.squeeze(np.argwhere(self.v > 0.5), axis=1)
            
            #Step 3.3.1: Calculate number of iterations:
            P_iter = np.array(self.P * np.exp(-self.gamma*(self.v - np.min(self.v))), dtype=np.int32)
            
            #Step 3.3.2: Update strong sub-set
            if strong_index.size != 0:
                for index in strong_index:
                    count_correct = 0
                    count_wrong = 0
                    for _ in range(P_iter[index]):
                        self.gmms[index].fit(x[index], y) #Only using x
                        if np.sign(y) == self.gmms[index].predict(x[index]):
                            count_correct+= 1
                            self.C[index] += 1   
                            self.lamda_sc[index] += self.lamda
                            self.lamda = self.lamda * ((1 - 2*self.p) / np.maximum(2*(1 - self.v[index]), self.eta))
                        else:
                            count_wrong += 1
                            self.lamda_sw[index] += self.lamda
                            self.lamda = self.lamda * ((1 + 2*self.p) / np.maximum(2*self.v[index], self.eta))

            #Step 3.3.3: Update weak subset
            if weak_index.size != 0:
                for index in weak_index:
                    for _ in range(P_iter[index]):
                        if np.sign(y) == self.gmms[index].predict(x[index]):
                            self.C[index] += 1
                            self.lamda_sc[index] += self.lamda
                        else:
                            self.lamda_sw[index] += self.lamda


        #Step 4: Update Omega, contribution from previous classifier
            if self.strong_gmms is not None:
                if np.sign(y) == np.sign(np.sum([self.alphas[i] * self.gmms[i].predict(x[i]) for i in range(self.N_classifiers)])):
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