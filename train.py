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
n_components = 16
#Load data:
N_labels = 2
N_train = 25000
N_test = 1000
print(">> Loading dataset ...")

#train dataset
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_train, N_classifiers))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_train, N_classifiers))
X_train = np.concatenate((X0, X1))
Y_train = np.concatenate((np.ones((N_train, ), dtype=np.int32),
                    -1*np.ones((N_train, ), dtype=np.int32)))

#Test dataset
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_test, N_classifiers))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_test, N_classifiers))
X_test = np.concatenate((X0, X1), axis=0)
Y_test = np.concatenate((np.ones((N_test, ), dtype=np.int32), 
                        -1*np.ones((N_test, ), dtype=np.int32)))

#Summerize data:
print(">> Train")
print(f"Number of normal: {np.sum(Y_train == 1)}")
print(f"Number of attack: {np.sum(Y_train == -1)}")

print(">> Test")
print(f"Number of normal: {np.sum(Y_test == 1)}")
print(f"Number of attack: {np.sum(Y_test == -1)}")

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
alphas = np.ones((N_classifiers,))
strong_gmms = None
gmms = [gmms.OnlineGMM(hyper.std, n_components=n_components, T=1)] * N_classifiers
for gmm in gmms:
    gmm.build(n_labels=N_labels)

print(">> Training")
for x, y in zip(X_train, Y_train):
#Step 1: Update number of normal and attack sample
#        and initialize weight for new sample
    #print("======================================")
    #print(f">> Class: {y}")
    if y == 1:
        S_normal += 1
        lamda = (S_normal + S_attack) / S_normal * r
    else:
        S_attack += 1
        lamda = (S_normal + S_attack) / S_attack * (1- r)
    # print(f"S_normal: {S_normal}")
    # print(f"S_attack: {S_attack}")
    # print(f"lamda1: {lamda}")
#Step 2: Calculate the combined classification weight error
    for i in range(N_classifiers):
        epsilon[i] = (lamda_sw[i]) / np.maximum(lamda_sc[i] + lamda_sw[i], eta)
        v[i] = np.maximum((1 - p) * epsilon[i] - p * np.sign(y) * gmms[i].predict(x[i]), eta)
    #print(f">> sign y: {np.sign(y)}")
#Step 3: Update classifier
    #Step 3.1: Sort list classifier:
    #BUG: HAVEN'T SORT STRONG INDEX
    sort_index = np.argsort(v)
    sort_strong_index = np.argwhere(v[sort_index] <= 0.5)
    #print(f">> sort strong index: {sort_strong_index}")
    strong_index = np.squeeze(sort_index[sort_strong_index], axis=1)
    # for index in strong_index:
    #     print(f">> strong index: {index}, v: {v[index]}")
    
    weak_index = np.squeeze(np.argwhere(v > 0.5), axis=1)
    # print(f"strong_index: {strong_index}")
    # print(f"weak_index: {weak_index}")
    
    #Step 3.3.1: Calculate number of iterations:
    P_iter = np.array(P * np.exp(-gamma*(v - np.min(v))), dtype=np.int32)
    
    #Step 3.3.2: Update strong sub-set
    # print(f"lamda_sc1: {lamda_sc}")
    # print(f"lamda_sw1: {lamda_sw}")
    # print(f"epsilon: {epsilon}")
    # print(f"v: {v}")
    # print(f"P_iter: {P_iter}")
    
    if strong_index.size != 0:
        #print(f">> Strong index:")
        for index in strong_index:
            count_correct = 0
            count_wrong = 0
            for _ in range(P_iter[index]):
                gmms[index].fit(x[index], y) #Only using x
                if np.sign(y) == gmms[index].predict(x[index]):
                    count_correct+= 1
                    C[index] += 1   
                    lamda_sc[index] += lamda
                    lamda = lamda * ((1 - 2*p) / np.maximum(2*(1 - v[index]), eta))
                else:
                    count_wrong += 1
                    lamda_sw[index] += lamda
                    lamda = lamda * ((1 + 2*p) / np.maximum(2*v[index], eta))
            #print(f"index: {index}, lamda: {lamda}, correct: {count_correct}, wrong: {count_wrong}")

    #Step 3.3.3: Update weak subset
    #print(f"weak_index: {weak_index}")
    if weak_index.size != 0:
        #print(f">> Weak index:")
        for index in weak_index:
            for _ in range(P_iter[index]):
                if np.sign(y) == gmms[index].predict(x[index]):
                    #print("Correct")
                    C[index] += 1
                    lamda_sc[index] += lamda
                else:
                    #print("Wrong")
                    lamda_sw[index] += lamda
    #print(f"lamda2: {lamda}")


#Step 4: Update Omega, contribution from previous classifier
    if strong_gmms is not None:
        if np.sign(y) == np.sign(np.sum([alphas[i] * gmms[i].predict(x[i]) for i in range(N_classifiers)])):
            Omega += 1
    #print(f"lamda_sc2: {lamda_sc}")
    #print(f"lamda_sw2: {lamda_sw}")
#Step 5: Construct strong classifier
    #Step 5.1: Calculate ensemble weight
    #print(f"C: {C}")
    #print(f"Omega: {Omega}")
    alphas = beta * np.log(np.maximum((1 - epsilon) / (epsilon + eta), eta)) \
            + (1 - beta) * np.log(np.maximum(C / (Omega), eta))
    #Step 5.2: Normalize ensemble weight
    #print("=============================================")
    alphas = alphas / np.sum(alphas)
    #print(f">> alphas: {alphas}")
    #Strong classifier:
    strong_gmms = gmms.copy()
    #np.sign(np.sum([alphas[i] * gmms[i](x) for i in range(N_classifiers)]))
    
#print(f">> len: {len(Y_test)} labels: {Y_test}")

#GLOBAL Evaluate
predicts = (np.sign(np.sum([alphas[i] * strong_gmms[i].predict(X_test[:, i]) for i in range(N_classifiers)])))
        
#print(f">> len: {len(predicts)} global predicts: {predicts}")

dtr = detection_rate(Y_test, predicts)
flr = false_alarm_rate(Y_test, predicts)

print(">> Global:")
print(f">> detection rate: {dtr}")
print(f">> false alarm rate: {flr}")

#Local Evaluate:
for local_index in range(N_classifiers):
    predicts = []
    for x in X_test:
        predicts.append(strong_gmms[local_index].predict(X_test[:, local_index]))
    dtr = detection_rate(Y_test, predicts)
    flr = false_alarm_rate(Y_test, predicts)
    print(f">> model: {local_index}")
    print(f"Detection rate: {dtr}")
    print(f"False alarm rate: {flr}")
    #print(f">> len: {len(predicts)} global predicts: {predicts}")