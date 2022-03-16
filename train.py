"""Training GMMs using online Adaboost
"""
import numpy as np
from GMMs import gmms
from GMMs import hyper
from utils import detection_rate, false_alarm_rate

#Hyper:
N_classifiers = 7
r = hyper.r
p = hyper.p
P = hyper.P
gamma = hyper.gamma
beta = hyper.beta
eta = 1e-5
#Load data:
N_labels = 2
N_train = 1000
N_test = 50
print(">> Loading dataset ...")
#train dataset
X_train = np.random.normal(0, 1, (N_train, N_classifiers)) #7 samples, 7 attributes
Y_train = np.array((np.mean(X_train, axis=1) > -0.5), dtype=np.float64)
Y_train[np.squeeze(np.argwhere(Y_train == 0))] -= 1
#Test dataset
X_test = np.random.normal(0, 1, (N_test, N_classifiers)) #7 samples, 7 attributes
Y_test = np.array(np.mean(X_test, axis=1) > -0.5, dtype=np.float64)
Y_test[np.squeeze(np.argwhere(Y_test == 0))] -= 1
#Step 0: Initlialize weight
print(">> Initializing ...")
S_normal, S_attack = 0, 0               #Use in equation (21)
Omega = 1                               #Use in step 5
lamda_sc = np.zeros((N_classifiers, ))  #Use in step 3, 4
lamda_sw = np.zeros((N_classifiers, ))  
C = np.zeros((N_classifiers, ))         #Use in step 3, 4, 5
v  = np.zeros((N_classifiers,))         #combined classification rate v_t, equation (23)
lamda = 0                               #Sample weight, using in equation (22)
epsilon = np.zeros((N_classifiers, ))   
#Initialize classifier
alphas = np.zeros((N_classifiers,))
strong_gmms = None
gmms = [gmms.OnlineGMM(hyper.std, T=1)] * N_classifiers
for gmm in gmms:
    gmm.build(n_labels=N_labels)

print(">> Training")
for x, y in zip(X_train, Y_train):
#Step 1: Calculate weight sample
    #Step 1.1: Update number of normal and attack sample
    if y == 1:
        S_normal += 1
    else:
        S_attack += 1
    #Step 1.2: Initialize weight for new sample
    
    if y == 1:
        lamda = (S_normal + S_attack) / S_normal * r
    else:
        lamda = (S_normal + S_attack) / S_attack * (1- r)
#Step 2: Calculate the combined classification weight error
    for i in range(N_classifiers):
        epsilon[i] = (lamda_sc[i]) / np.maximum(lamda_sc[i] + lamda_sw[i], eta)
        v[i] = (1 - epsilon[i]) - p * np.sign(y) * gmms[i].predict(x)
#Step 3: Update classifier
    #Step 3.1: Sort list classifier:
    sort_index = np.argsort(v)
    #Step 3.2: Separate sub-set classifiers
    strong_index = np.argwhere(v < 0.5)
    weak_index = np.argwhere(v >= 0.5)
    #Step 3.3.1: Calculate number of iterations:
    P_iter = np.array(P * np.exp(-gamma*(v - np.min(v))), dtype=np.int32)
    #Step 3.3.2: Update strong sub-set
    if strong_index.size != 0:
        for index in strong_index[0]:
            for _ in range(P_iter[index]):
                gmms[index].fit(x[index]) #Only using x
                if np.sign(y) == gmms[index].predict(x):
                    C[index] += 1
                    lamda_sc[index] += lamda
                    lamda = lamda * ((1 - 2*p) / np.maximum(2*(1 - v[index]), eta))
                else:
                    lamda_sw[index] += lamda
                    lamda = lamda * ((1 + 2*p) / np.maximum(2*v[index], eta))
    #Step 3.3.3: Update weak subset
    if weak_index.size != 0:
        for index in weak_index[0]:
            for _ in range(P_iter[index]):
                gmms[index].fit(x[index]) #Only using x
                if np.sign(y) == gmms[index].predict(x):
                    C[index] += 1
                    lamda_sc[index] += lamda
                else:
                    lamda_sw[index] += lamda

#Step 4: Update Omega, contribution from previous classifier
    if strong_gmms is not None:
        if np.sign(y) == np.sign(np.sum([alphas[i] * gmms[i].predict(x) for i in range(N_classifiers)])):
            Omega += 1

#Step 5: Construct strong classifier
    #Step 5.1: Calculate ensemble weight
    # print(f"epsilon: {epsilon}")
    # print(f"C: {C}")
    # print(f"Omega: {Omega}")
    alphas = beta * np.log(np.maximum((1 - epsilon), eta) / np.maximum(epsilon, eta)) \
            + (1 - beta) * np.log(np.maximum(C, eta) / np.maximum(Omega, eta))
    #Step 5.2: Normalize ensemble weight
    # print(f"left: {(1 - epsilon) / epsilon + eta}")
    # print(f"right: {np.log(C / Omega + eta)}")
    # print(f"alphas: {alphas}")
    alphas = alphas / np.maximum(np.sum(alphas), eta)
    #Strong classifier:
    strong_gmms = gmms.copy()
    #np.sign(np.sum([alphas[i] * gmms[i](x) for i in range(N_classifiers)]))
    
#Evaluate
predicts = []
for x in X_test:
    predicts.append(np.sign(np.sum([alphas[i] * strong_gmms[i].predict(x) for i in range(N_classifiers)])))
        
print(f"labels: {Y_test}")
print(f"predicts: {predicts}")

dtr = detection_rate(Y_test, predicts)
flr = false_alarm_rate(Y_test, predicts)

print(f">> detection rate: {dtr}")
print(f">> false alarm rate: {flr}")